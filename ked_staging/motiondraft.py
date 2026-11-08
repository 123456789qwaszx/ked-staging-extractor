"""모션 초안 — 어휘로 설명 못 한 제자리 움직임을 ked-motion-tool 저작 원본(*.motion-authoring.json)으로 낸다.

P3의 지나가는 움직임 찾기(vocab.TransientFinder)가 런타임 모션(motions.json)으로도 설명하지 못한 인물을
"unexplained"로 남긴다. 여기서는 그 줄의 프레임마다 그 인물 층을 평행 이동해 보며 영상과 가장 잘 맞는 자리를
찾고(앞 프레임 자리에서 언덕 내려가기 + 포물선으로 반 화소 다듬기), 그 궤적을 모션 툴이 읽는 모양으로 적는다.

  채널  CharacterPortrait_Track_Move_X · _Y (position) — 런타임 리그의 평행 이동 축
  값    화면 px ÷ (깊이 배율 × 카메라 배율), y는 위가 + (리그 로컬 px)
  끝    restore — 제자리 움직임이다(정착 화면은 그대로)

평행 이동만 잰다 — 회전 · 배율은 범위 밖이다. 값의 주인은 모션 툴이다: 초안을 열어 다듬고 AI 보조로 프리셋을
만든 뒤 motions.json으로 내보낸다.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

AUTHORING_SCHEMA = "motion-authoring/2"
TRACK_X = "CharacterPortrait_Track_Move_X"
TRACK_Y = "CharacterPortrait_Track_Move_Y"
MARGIN = 40                  # 층을 넓혀 그리는 둘레(분석 px) — 이 안에서만 움직임을 찾는다
QUIET = 0.75                 # 리그 px — 이보다 작은 값은 움직임이 아니다
SIMPLIFY = 0.75              # 키 줄이기 허용(리그 px)


def _robust(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(np.abs(a - b).mean(2), 60.0)


def measure(oracle, catalog, painter, mask: np.ndarray, history: Sequence[Sequence[dict]], line: Sequence[dict],
            frames: Sequence[Tuple[float, np.ndarray]], slot: str) -> Optional[dict]:
    """줄 시간 축의 프레임마다 slot 층이 예측 자리에서 화면 몇 px 옮겨졌나 → {"times", "dx", "dy"(캔버스 px),
    "depthScale", "cameraScale"}. 그 슬롯이 보이지 않으면 None."""
    times = [t for t, _ in frames]
    snaps = oracle.timeline(list(history) + [list(line)], times)["snapshots"]
    est = (0, 0)
    out_dx, out_dy, used_t = [], [], []
    depth_scale, camera_scale = 1.0, 1.0
    for (t, frame), sn in zip(frames, snaps):
        below = np.zeros(frame.shape, np.float32)
        front: List[Tuple[np.ndarray, np.ndarray]] = []
        layer = None
        for s in sn.slots:
            a_val = float(s.get("alpha", 0.0))
            if not s.get("quad") or a_val <= 0.003:
                continue
            entry = catalog.background(s.get("sprite") or "") if s.get("background") else \
                catalog.portrait(s.get("character") or "", s.get("variant") or "a", s.get("emotion") or "")
            if entry is None:
                continue
            quad = np.asarray(s["quad"], np.float64)
            if s["key"] == slot:
                color, alpha = painter.layer(entry, quad, bool(s.get("toolMirror")), margin=MARGIN)
                layer = (color, alpha * a_val)
                depth_scale = float(s.get("depthScale") or 1.0)
                continue
            color, alpha = painter.layer(entry, quad, bool(s.get("toolMirror")))
            a = (alpha * a_val)[:, :, None]
            if layer is None:
                below = color * a + below * (1 - a)
            else:
                front.append((color, a))
        if layer is None:
            continue
        camera_scale = float((sn.shot or {}).get("cameraScale") or 1.0)
        h, w = frame.shape[:2]
        lc, la = layer
        ys, xs = np.nonzero(la > 0.05)
        if len(xs) == 0:
            continue
        # 대조 영역: 층이 닿을 수 있는 자리(둘레 포함)를 화면 좌표로
        y0, y1 = max(0, ys.min() - 2 * MARGIN), min(h, ys.max() + 1)
        x0, x1 = max(0, xs.min() - 2 * MARGIN), min(w, xs.max() + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        weight = mask[y0:y1, x0:x1]
        target = frame[y0:y1, x0:x1]
        cache: Dict[Tuple[int, int], float] = {}

        def err(dx: int, dy: int) -> float:
            if (dx, dy) in cache:
                return cache[(dx, dy)]
            if abs(dx) > MARGIN or abs(dy) > MARGIN:
                return float("inf")
            # 층을 (dx, dy) 옮긴 그림 = 넓힌 캔버스의 (MARGIN - dx, MARGIN - dy)에서 자른 것
            sy, sx = MARGIN - dy + y0, MARGIN - dx + x0
            c = lc[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
            a = la[sy:sy + (y1 - y0), sx:sx + (x1 - x0)][:, :, None]
            comp = c * a + below[y0:y1, x0:x1] * (1 - a)
            for fc, fa in front:
                fa_ = fa[y0:y1, x0:x1]
                comp = fc[y0:y1, x0:x1] * fa_ + comp * (1 - fa_)
            e = float((_robust(target, comp) * weight).sum())
            cache[(dx, dy)] = e
            return e

        cx, cy = est
        for _ in range(4 * MARGIN):
            best = min(((err(cx + i, cy + j), cx + i, cy + j) for j in (-1, 0, 1) for i in (-1, 0, 1)))
            if (best[1], best[2]) == (cx, cy):
                break
            cx, cy = best[1], best[2]
        est = (cx, cy)

        def vertex(em: float, e0: float, ep: float) -> float:
            den = em - 2 * e0 + ep
            return 0.0 if not np.isfinite(den) or den <= 1e-9 else float(np.clip(0.5 * (em - ep) / den, -0.5, 0.5))

        fx = cx + vertex(err(cx - 1, cy), err(cx, cy), err(cx + 1, cy))
        fy = cy + vertex(err(cx, cy - 1), err(cx, cy), err(cx, cy + 1))
        used_t.append(t)
        out_dx.append(fx / painter.scale)
        out_dy.append(fy / painter.scale)
    if not used_t:
        return None
    return {"times": used_t, "dx": out_dx, "dy": out_dy, "depthScale": depth_scale, "cameraScale": camera_scale}


def _rdp(t: Sequence[float], v: Sequence[float], tol: float) -> List[int]:
    """Ramer–Douglas–Peucker — 남길 점 번호."""
    def rec(i: int, j: int) -> List[int]:
        if j <= i + 1:
            return [i, j]
        ti, tj, vi, vj = t[i], t[j], v[i], v[j]
        worst, k = -1.0, -1
        for m in range(i + 1, j):
            line_v = vi + (vj - vi) * (t[m] - ti) / max(tj - ti, 1e-9)
            d = abs(v[m] - line_v)
            if d > worst:
                worst, k = d, m
        if worst <= tol:
            return [i, j]
        return rec(i, k)[:-1] + rec(k, j)
    return rec(0, len(t) - 1) if len(t) > 1 else [0]


def _keys(t: Sequence[float], v: Sequence[float]) -> List[dict]:
    keep = _rdp(t, v, SIMPLIFY)
    tt = [t[i] for i in keep]
    vv = [v[i] for i in keep]
    vv[0] = vv[-1] = 0.0          # restore — 처음과 끝은 제자리
    keys = []
    for k in range(len(tt)):
        if 0 < k < len(tt) - 1:
            slope = (vv[k + 1] - vv[k - 1]) / max(tt[k + 1] - tt[k - 1], 1e-6)
        else:
            slope = 0.0
        keys.append({"t": round(float(tt[k]), 4), "v": round(float(vv[k]), 3), "inTangent": round(float(slope), 3),
                     "outTangent": round(float(slope), 3), "tangent": "free", "broken": False})
    return keys


def to_motion(name: str, sample: dict) -> Optional[dict]:
    """measure 결과 → 모션 툴 MotionAsset(JSON). 움직임이 없으면 None."""
    k = max(1e-6, float(sample["depthScale"]) * float(sample["cameraScale"]))
    t = np.asarray(sample["times"], np.float64)
    x = np.asarray(sample["dx"], np.float64) / k
    y = -np.asarray(sample["dy"], np.float64) / k
    moving = np.nonzero((np.abs(x) > QUIET) | (np.abs(y) > QUIET))[0]
    if len(moving) == 0:
        return None
    a, b = max(0, moving[0] - 1), min(len(t) - 1, moving[-1] + 1)
    t, x, y = t[a:b + 1] - t[a], x[a:b + 1], y[a:b + 1]
    channels = []
    if np.abs(x).max() > QUIET:
        channels.append({"node": TRACK_X, "kind": "position", "x": {"keys": _keys(t, x)}})
    if np.abs(y).max() > QUIET:
        channels.append({"node": TRACK_Y, "kind": "position", "y": {"keys": _keys(t, y)}})
    return {"name": name, "rig": "character", "duration": round(float(t[-1]), 4), "end": "restore",
            "previewMode": "relative", "channels": channels}


def peak(motion: dict) -> Tuple[float, float]:
    """모션 초안의 x · y 최대 크기(부호 포함, 리그 px)."""
    out = [0.0, 0.0]
    for ch in motion.get("channels", []):
        for i, axis in enumerate(("x", "y")):
            keys = (ch.get(axis) or {}).get("keys", [])
            if keys:
                v = max((k["v"] for k in keys), key=abs)
                if abs(v) > abs(out[i]):
                    out[i] = float(v)
    return out[0], out[1]


def write_authoring(path: str, motions: Sequence[dict]) -> str:
    doc = {"schema": AUTHORING_SCHEMA, "motions": list(motions)}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path
