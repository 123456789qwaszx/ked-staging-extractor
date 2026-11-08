"""다른 게임 — 원본 스프라이트 · 배경으로 카탈로그를 만든다(원본 모드).

그 게임의 원본 그림을 구할 수 있으면 영상에서 인물을 배우지 않는다(규칙 1 — 원본이 있으면 원본으로). 영상에서 재는
것은 "게임이 원본을 화면 어디에 어떤 배율로 그렸나"뿐이다. 결과는 catalog-build와 같은 카탈로그 + 튜닝 겹본이라
extract를 그대로 돌린다.

원본 폴더
  <이름>_<변형>_<표정>.png   예: Kasimira_a_14.png → 인물 kasimira · 변형 a · 표정 14 (' (1)' 같은 사본은 무시)
  BG_<이름>.png             예: BG_mine_inside.png → 배경 원본

  1. 줄마다 정착 화면(scan의 frames/)에서 원본 스프라이트를 SIFT 특징점으로 찾는다 — 닮음 변환(배율 · 평행 이동,
     회전 없음)을 RANSAC으로 맞추고, 맞은 점이 MIN_INLIERS 이상이면 그 인물이 섰다. 대사창 아래는 보지 않는다.
  2. 인물마다 가장 자주 선 자리(배율 1% · 4px 안)를 기준으로, catalog-build와 같은 규칙으로 그림을 만든다:
     높이 = 기준 자리 사각형 높이 × 영상 배율, 가장 자주 선 자리가 가장 가까운 place_*에 정확히 오게 여백을 두른다.
     원본은 몸 전체 · 표정 전부가 있으므로 채우기 · 얼굴 바꾸기가 없다.
  3. 배경: scan의 배경 무리마다 판(plate)과 원본 배경을 SIFT로 맞대어, 원본을 그 장면의 화면 구도로 편다
     (게임은 원본 배경의 일부를 확대해 보여 준다). 인물이 박히지 않은 깨끗한 판이 된다. 맞는 원본이 없는 무리
     (메뉴 · 타이틀)는 scan의 판을 그대로 쓴다.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .catalog import Background, Catalog, Portrait
from .catalogdraft import DEPTH_TRY, PLACES, DraftEditor, _premultiplied_warp, _slot, write_overlay
from .imageio import read_image, write_image

SPRITE = re.compile(r"^(?P<name>.+?)_(?P<variant>[A-Za-z])_(?P<emotion>\d{1,2})(?P<copy> \(\d+\))?\.png$")
BACKGROUND = re.compile(r"^BG_(?P<key>.+)\.png$", re.IGNORECASE)
FEATURE_HEIGHT = 1000        # 특징점을 뽑을 때 원본을 이 높이로 줄인다
MIN_INLIERS = 25             # 이만큼 맞으면 그 인물이 섰다
MAX_ROTATION = 2.0           # 도 — 원본은 돌려 그리지 않는다
BG_MIN_INLIERS = 40
BG_MAX_ERROR = 15.0          # 편 원본과 정착 화면의 무대 차 중앙값이 이보다 크면 다른 그림이다(맞으면 5 안팎)


def ident(text: str) -> str:
    """런타임 키 규칙(영문 소문자 · 숫자 · _)으로."""
    s = re.sub(r"[^0-9a-z_]+", "_", text.strip().lower()).strip("_")
    return s if s and s[0].isalpha() else "x_" + s


def read_folder(folder: str) -> Tuple[Dict[Tuple[str, str], Dict[str, str]], Dict[str, str]]:
    """원본 폴더 → ({(인물, 변형): {표정: 경로}}, {배경 키: 경로})."""
    sprites: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
    backgrounds: Dict[str, str] = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        b = BACKGROUND.match(name)
        if b:
            backgrounds[ident(b.group("key"))] = path
            continue
        m = SPRITE.match(name)
        if not m or m.group("copy"):
            continue
        sprites[(ident(m.group("name")), m.group("variant").lower())][f"{int(m.group('emotion')):02d}"] = path
    return dict(sprites), backgrounds


class Features:
    """한 그림의 SIFT 특징점(줄인 크기에서) — 원래 크기 좌표로 돌려 쓴다."""

    def __init__(self, sift, img: np.ndarray, mask: Optional[np.ndarray] = None, height: Optional[int] = None):
        self.scale = 1.0 if height is None else height / float(img.shape[0])
        small = img if self.scale == 1.0 else cv2.resize(img, None, fx=self.scale, fy=self.scale,
                                                         interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small[:, :, :3], cv2.COLOR_BGR2GRAY)
        m = None
        if mask is not None:
            m = mask if self.scale == 1.0 else cv2.resize(mask, (g.shape[1], g.shape[0]),
                                                          interpolation=cv2.INTER_NEAREST)
        self.kp, self.desc = sift.detectAndCompute(g, m)
        self.pts = np.float32([k.pt for k in self.kp]) / self.scale if self.kp else np.zeros((0, 2), np.float32)


def similarity(src: Features, dst: Features, matcher, min_inliers: int) -> Optional[Tuple[float, float, float, int]]:
    """src(원래 좌표) → dst(원래 좌표) 닮음 변환 → (배율, x, y, 맞은 점). 회전이 있거나 모자라면 None."""
    if src.desc is None or dst.desc is None or len(src.kp) < 4 or len(dst.kp) < 4:
        return None
    pairs = [a for a, b in (p for p in matcher.knnMatch(src.desc, dst.desc, k=2) if len(p) == 2)
             if a.distance < 0.75 * b.distance]
    if len(pairs) < min_inliers:
        return None
    s = np.float32([src.pts[a.queryIdx] for a in pairs])
    d = np.float32([dst.pts[a.trainIdx] for a in pairs])
    m, inl = cv2.estimateAffinePartial2D(s, d, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if m is None or inl is None:
        return None
    n = int(inl.sum())
    rot = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    if n < min_inliers or abs(rot) > MAX_ROTATION:
        return None
    return float(np.hypot(m[0, 0], m[1, 0])), float(m[0, 2]), float(m[1, 2]), n


def locate(frames: Sequence[np.ndarray], bottom: int, sprites: Dict[Tuple[str, str], Dict[str, str]],
           log: Callable[[str], None] = print) -> List[List[dict]]:
    """줄마다 선 인물 → [[{character, variant, scale, x, y, inliers}]] — 인물마다 가장 잘 맞는 변형 하나."""
    sift = cv2.SIFT_create(nfeatures=6000)
    matcher = cv2.BFMatcher()
    refs: Dict[Tuple[str, str], Features] = {}
    for key, emos in sorted(sprites.items()):
        img = read_image(emos[sorted(emos)[0]], alpha=True)
        refs[key] = Features(sift, img, (img[:, :, 3] > 128).astype(np.uint8) * 255, FEATURE_HEIGHT)
    out: List[List[dict]] = []
    for i, frame in enumerate(frames):
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[:bottom] = 255
        f = Features(sift, frame.astype(np.uint8) if frame.dtype != np.uint8 else frame, mask)
        best: Dict[str, dict] = {}
        for (name, variant), ref in refs.items():
            hit = similarity(ref, f, matcher, MIN_INLIERS)
            if hit and (name not in best or hit[3] > best[name]["inliers"]):
                best[name] = {"character": name, "variant": variant, "scale": hit[0], "x": hit[1], "y": hit[2],
                              "inliers": hit[3]}
        out.append(sorted(best.values(), key=lambda h: h["x"]))
        if out[-1]:
            log(f"  줄 {i:3d}: " + ", ".join(f"{h['character']}/{h['variant']} ×{h['scale']:.3f} @({h['x']:.0f},{h['y']:.0f})"
                                            for h in out[-1]))
    return out


def stance(hits: Sequence[dict]) -> dict:
    """가장 자주 선 자리 — 배율 1% · 4px 안을 같은 자리로 본다."""
    best = None
    for h in hits:
        near = [g for g in hits if abs(g["scale"] / h["scale"] - 1) < 0.01 and abs(g["x"] - h["x"]) <= 4
                and abs(g["y"] - h["y"]) <= 4]
        if best is None or len(near) > len(best):
            best = near
    return {"scale": float(np.median([g["scale"] for g in best])), "x": float(np.median([g["x"] for g in best])),
            "y": float(np.median([g["y"] for g in best])), "lines": len(best), "seen": len(hits)}


def _view(original: np.ndarray, hit, size: Tuple[int, int]) -> np.ndarray:
    s, x, y = hit[0], hit[1], hit[2]
    m = np.float32([[s, 0, x], [0, s, y]])
    return cv2.warpAffine(original[:, :, :3], m, size, flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _robust_error(view: np.ndarray, frame: np.ndarray, bottom: int) -> float:
    """무대에서 차의 중앙값 — 인물이 덮은 자리가 절반 아래면 배경의 차만 남는다."""
    return float(np.median(np.abs(view[:bottom].astype(np.float32) - frame[:bottom].astype(np.float32)).max(2)))


def fit_background(original_feat: "Features", original: np.ndarray, frames: Sequence[np.ndarray],
                   feats: Sequence["Features"], bottom: int, matcher) -> Optional[Tuple[np.ndarray, int, float, int]]:
    """원본 배경을 그 무리 줄들의 화면 구도로 편다. 줄마다 맞대어 가장 많은 줄이 같은 구도(배율 1% · 4px)를 쓴다.
    → (편 그림, 그 구도의 줄 수, 차, 맞은 점). 맞지 않으면 None."""
    found = []
    for frame, f in zip(frames, feats):
        hit = similarity(original_feat, f, matcher, BG_MIN_INLIERS)
        if hit is None:
            continue
        view = _view(original, hit, (frame.shape[1], frame.shape[0]))
        err = _robust_error(view, frame, bottom)
        if err <= BG_MAX_ERROR:
            found.append((hit, err))
    if not found:
        return None
    best = None
    for h, _e in found:
        near = [(g, e) for g, e in found if abs(g[0] / h[0] - 1) < 0.01 and abs(g[1] - h[1]) <= 4
                and abs(g[2] - h[2]) <= 4]
        if best is None or len(near) > len(best):
            best = near
    hit = (float(np.median([g[0] for g, _ in best])), float(np.median([g[1] for g, _ in best])),
           float(np.median([g[2] for g, _ in best])), int(max(g[3] for g, _ in best)))
    view = _view(original, hit, (frames[0].shape[1], frames[0].shape[0]))
    return view, len(best), float(np.median([e for _, e in best])), hit[3]


def build_from_originals(folder: str, draft_path: str, out_dir: str, runtime_tuning: str,
                         oracle_command: Optional[Sequence[str]] = None,
                         log: Callable[[str], None] = print) -> Tuple[str, str]:
    """원본 폴더 + scan 초안(줄 정착 화면 · 배경 판) → out_dir/catalog.json · tuning/. → (카탈로그, 튜닝 폴더)."""
    from .observe import depth_presets
    from .oracle import Oracle

    sprites, bg_files = read_folder(folder)
    if not sprites:
        raise ValueError(f"원본 스프라이트(<이름>_<변형>_<표정>.png)가 없습니다: {folder}")
    ed = DraftEditor.load(draft_path)
    d = ed.data
    vw, vh = (int(x) for x in d["videoSize"])
    bottom = int(d["stageBottom"])
    lines = d.get("lines", [])
    frames = [read_image(ed.abspath(l["frame"])) for l in lines]
    log(f"원본: 인물 {len({k[0] for k in sprites})}명 · 그림 {sum(len(v) for v in sprites.values())}장 · "
        f"배경 {len(bg_files)}장 / 줄 {len(frames)}개")
    hits = locate(frames, bottom, sprites, log)
    os.makedirs(out_dir, exist_ok=True)
    report = {"lines": [{"index": l["index"], "cast": h} for l, h in zip(lines, hits)], "characters": {},
              "backgrounds": {}}

    # 배경
    cat = Catalog("originals", os.path.abspath(out_dir))
    bg_dims: Dict[str, Tuple[int, int]] = {}
    sift, matcher = cv2.SIFT_create(nfeatures=8000), cv2.BFMatcher()
    originals = {k: read_image(p) for k, p in bg_files.items()}
    ofeats = {k: Features(sift, img, None, FEATURE_HEIGHT) for k, img in originals.items()}
    stage = np.zeros((vh, vw), np.uint8)
    stage[:bottom] = 255
    used: Dict[str, int] = defaultdict(int)
    for b in ed.backgrounds:
        if b.get("excluded"):
            continue
        plate = read_image(ed.abspath(b["plate"]))
        members = [i for i in b.get("lines", []) if 0 <= i < len(frames)]
        gframes = [frames[i] for i in members]
        gfeats = [Features(sift, f, stage) for f in gframes]
        best = None
        for k, img in originals.items():
            fit = fit_background(ofeats[k], img, gframes, gfeats, bottom, matcher) if gframes else None
            if fit and (best is None or (fit[1], -fit[2]) > (best[1][1], -best[1][2])):
                best = (k, fit)
        if best is None:
            key, view, how = b["key"], plate, "판(원본 없음)"
        else:
            used[best[0]] += 1
            key = best[0] if used[best[0]] == 1 else f"{best[0]}_{used[best[0]]}"
            view = best[1][0]
            how = f"원본 {best[0]} (줄 {best[1][1]}/{len(gframes)}이 같은 구도, 차 중앙값 {best[1][2]:.1f})"
        rel = os.path.join("backgrounds", f"{key}.png")
        write_image(os.path.join(out_dir, rel), view)
        cat.backgrounds.append(Background(key, rel, (view.shape[1], view.shape[0])))
        bg_dims[key] = (view.shape[1], view.shape[0])
        report["backgrounds"][b["key"]] = {"key": key, "from": how, "lines": b.get("lines", [])}
        log(f"  배경 {b['key']} → {key}: {how}")

    # 인물
    seen: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for hs in hits:
        for h in hs:
            seen[(h["character"], h["variant"])].append(h)
    depths_have = depth_presets(runtime_tuning)
    depth_try = [x for x in DEPTH_TRY if x in depths_have] or depths_have[:1]
    tuning = os.path.join(out_dir, "tuning")
    write_overlay(runtime_tuning, tuning, {(c, v, "01"): (100, 100) for c, v in seen}, bg_dims)
    plans = []
    with Oracle(tuning, command=oracle_command) as oracle:
        v = vh / float(oracle.height)
        for (c, var), hs in sorted(seen.items()):
            st = stance(hs)
            emos = sprites[(c, var)]
            img = read_image(emos[sorted(emos)[0]], alpha=True)
            ys, xs = np.nonzero(img[:, :, 3] > 8)
            k = st["scale"]
            top = (st["y"] + k * ys.min()) / v
            x0, x1 = (st["x"] + k * xs.min()) / v, (st["x"] + k * (xs.max() + 1)) / v
            depth = depth_try[-1]
            for cand in depth_try:
                if _slot(oracle, c, var, "center", "bust", cand)["rect"][1] <= top - 2:
                    depth = cand
                    break
            centers = {}
            for p in PLACES:
                r = _slot(oracle, c, var, p, "bust", depth)["rect"]
                centers[p] = (r[0] + r[2] / 2.0, r)
            place = min(PLACES, key=lambda p: abs(centers[p][0] - (x0 + x1) / 2.0))
            cx, rect = centers[place]
            H = int(round(rect[3] * v))
            half = max(abs(x0 - cx), abs(x1 - cx)) * v + 4
            W = int(min(2 * np.ceil(half), np.floor(H * rect[2] / rect[3])))
            plans.append(((c, var), st, place, depth, (W, H)))
            report["characters"][f"{c}/{var}"] = dict(st, place=place, depth=depth, size=[W, H])
            log(f"  {c}/{var}: 줄 {st['seen']}개에 섬 · 기준 자리 place_{place} · bust · {depth} · 원본 ×{k:.3f} · 그림 {W}×{H}")
    dims = {(c, var, e): size for (c, var), _s, _p, _d, size in plans for e in sprites[(c, var)]}
    write_overlay(runtime_tuning, tuning, dims, bg_dims)
    with Oracle(tuning, command=oracle_command) as oracle:
        v = vh / float(oracle.height)
        for (c, var), st, place, depth, (W, H) in plans:
            s = _slot(oracle, c, var, place, "bust", depth, str(int(sorted(sprites[(c, var)])[0])))
            qx, qy, qw, qh = s["bounds"]
            k = st["scale"] * W / qw / v                    # 원본 화소 → 그림 화소
            m = np.float32([[k, 0, (st["x"] / v - qx) * W / qw], [0, k, (st["y"] / v - qy) * H / qh]])
            for e, path in sorted(sprites[(c, var)].items()):
                out = _premultiplied_warp(read_image(path, alpha=True), m, (W, H))
                rel = os.path.join("portraits", c, var, f"{e}.png")
                write_image(os.path.join(out_dir, rel), out)
                ys, xs = np.nonzero(out[:, :, 3] > 8)
                box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) \
                    if len(xs) else (0, 0, 0, 0)
                cat.portraits.append(Portrait(c, var, e, rel, (W, H), box))
    unseen = sorted(f"{c}/{var}" for c, var in sprites if (c, var) not in seen)
    if unseen:
        log(f"  영상에서 찾지 못해 뺀 원본: {', '.join(unseen)}")
    report["unseen"] = unseen
    path = os.path.join(out_dir, "catalog.json")
    cat.save(path)
    with open(os.path.join(out_dir, "originals-report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    log(f"초상 {len(cat.portraits)} · 배경 {len(cat.backgrounds)} → {path}")
    return path, tuning
