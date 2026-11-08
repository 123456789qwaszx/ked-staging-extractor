"""평가 — 추출한 .yarn을 오라클로 다시 접어 정답 대본과 줄마다 비교한다(P1 통과 기준: 상태 일치 95% 이상).

줄 짝짓기: 추출 줄 머리와 정답 줄 머리를 시간 순서대로 짝짓는다(차이 ≤ 0.25초).
상태 일치(줄마다): 보이는 배경 그림이 같고, 보이는 인물 집합이 같고, 인물마다 변형 · 표정 · 반전이 같고
화면 사각형(bounds)의 네 값이 모두 허용(기본 6px) 안이다. 그리기 순서와 시간은 보지 않는다(P2 · P3).
대사: 화자 일치, 본문 완전 일치(띄어쓰기 무시), 글자 정확도(편집 거리).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .oracle import Oracle, Snapshot, route_lines

STATE_PASS = 0.95


def _visible(snap: Snapshot) -> Tuple[Optional[str], Dict[str, dict]]:
    bg = None
    chars: Dict[str, dict] = {}
    for s in snap.visible_slots(0.5):
        if s.get("background"):
            bg = s.get("sprite")
        elif s.get("character"):
            chars[s["character"]] = s
    return bg, chars


def compare_state(truth: Snapshot, got: Snapshot, tolerance: float = 6.0) -> List[str]:
    """다른 점 목록(비면 일치)."""
    problems: List[str] = []
    tb, tc = _visible(truth)
    gb, gc = _visible(got)
    if tb != gb:
        problems.append(f"배경 {tb} ≠ {gb}")
    for c in sorted(set(tc) | set(gc)):
        if c not in gc:
            problems.append(f"{c} 없음")
            continue
        if c not in tc:
            problems.append(f"{c} 더 있음")
            continue
        t, g = tc[c], gc[c]
        for axis in ("variant", "emotion"):
            if t.get(axis) != g.get(axis):
                problems.append(f"{c} {axis} {t.get(axis)} ≠ {g.get(axis)}")
        if bool(t.get("toolMirror")) != bool(g.get("toolMirror")):
            problems.append(f"{c} 반전 {t.get('toolMirror')} ≠ {g.get('toolMirror')}")
        tbx, gbx = t.get("bounds"), g.get("bounds")
        if tbx and gbx:
            d = max(abs(a - b) for a, b in zip(tbx, gbx))
            if d > tolerance:
                problems.append(f"{c} 자리 {d:.0f}px 어긋남")
    return problems


def _norm(text: str) -> str:
    return "".join((text or "").split())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def match_lines(truth_heads: Sequence[float], got_heads: Sequence[float], window: float = 0.25
                ) -> List[Tuple[int, int]]:
    pairs = []
    j = 0
    for i, th in enumerate(truth_heads):
        while j < len(got_heads) and got_heads[j] < th - window:
            j += 1
        if j < len(got_heads) and abs(got_heads[j] - th) <= window:
            pairs.append((i, j))
            j += 1
    return pairs


def evaluate(truth: dict, staging: dict, oracle: Oracle, tolerance: float = 6.0) -> dict:
    truth_lines = truth["lines"]
    truth_snaps = [l["snapshot"] for l in route_lines(oracle.fold_yarn(truth["route"]))]
    got_lines = staging["lines"]
    got_snaps = [l["snapshot"] for l in route_lines(oracle.fold_yarn([staging["files"]["yarn"]]))]
    if len(got_snaps) != len(got_lines):
        raise ValueError(f"추출 .yarn의 줄 수({len(got_snaps)})가 staging.json({len(got_lines)})과 다릅니다")
    fps = float(truth.get("fps", 30.0))
    pairs = match_lines([l["head"] for l in truth_lines], [l["head"] for l in got_lines])
    rows = []
    state_ok = text_ok = speaker_ok = 0
    chars = errors = 0
    head_err = []
    for ti, gi in pairs:
        t, g = truth_lines[ti], got_lines[gi]
        problems = compare_state(truth_snaps[ti], got_snaps[gi], tolerance)
        state_ok += not problems
        tt, gt = _norm(t["body"]), _norm(g.get("text", ""))
        text_ok += tt == gt
        speaker_ok += _norm(t["speaker"]) == _norm(g.get("speaker", ""))
        chars += len(tt)
        errors += edit_distance(tt, gt)
        head_err.append((g["head"] - t["head"]) * fps)
        rows.append({"truth": ti, "got": gi, "headFrames": round(head_err[-1], 2), "state": problems,
                     "text": None if tt == gt else {"truth": t["body"], "got": g.get("text", "")}})
    n = len(truth_lines)
    matched = {ti for ti, _ in pairs}
    report = {
        "truthLines": n,
        "gotLines": len(got_lines),
        "matched": len(pairs),
        "missed": [i for i in range(n) if i not in matched],
        "extra": [j for j in range(len(got_lines)) if j not in {gj for _, gj in pairs}],
        "headFramesMax": round(max((abs(e) for e in head_err), default=0.0), 2),
        "stateMatch": state_ok / n if n else 0.0,
        "speakerMatch": speaker_ok / n if n else 0.0,
        "textExact": text_ok / n if n else 0.0,
        "charAccuracy": 1.0 - errors / chars if chars else 0.0,
        "lines": rows,
    }
    report["pass"] = report["stateMatch"] >= STATE_PASS
    return report


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── P2: 시간 · 이징 ───────────────────────────────────────────

TIMING_PASS_DURATION = 0.9   # 길이 오차 2fr 안에 든 흐름 비율
TIMING_PASS_EASE = 0.8       # 움직임 흐름의 이징 일치 비율
_ALPHA = {"fade_in", "fade_out", "fade", "bg_fade_in", "bg_fade_out"}


def _who(snapshot: Snapshot) -> Dict[str, str]:
    """슬롯 키 → 누구(인물 이름 · 배경은 'bg')."""
    out = {}
    for s in snapshot.slots:
        out[s["key"]] = "bg" if s.get("background") else (s.get("character") or s["key"])
    return out


def _tweens(schedule: Sequence[dict], before: Snapshot, after: Snapshot) -> Dict[Tuple[str, str], List[dict]]:
    """시간표에서 보이는 흐름만 — (누구, 'alpha' | 'move') → 항목들."""
    who_b, who_a = _who(before), _who(after)
    vis_b = {s["key"] for s in before.visible_slots(0.5)}
    vis_a = {s["key"] for s in after.visible_slots(0.5)}
    out: Dict[Tuple[str, str], List[dict]] = {}
    for e in schedule:
        if e.get("wait") or float(e.get("duration", 0)) <= 0 or not (e.get("tween") or e.get("fade")):
            continue
        key = (e.get("args") or [""])[0]
        kind = "alpha" if e["name"] in _ALPHA else "move"
        if kind == "move" and (key not in vis_b or key not in vis_a):
            continue
        who = who_a.get(key) or who_b.get(key) or key
        out.setdefault((who, kind), []).append(e)
    return out


def evaluate_timing(truth: dict, staging: dict, oracle: Oracle, tolerance_frames: float = 2.0) -> dict:
    fps = 24.0
    truth_lines = truth["lines"]
    if not all("schedule" in l for l in truth_lines):
        raise ValueError("정답에 시간표가 없습니다 — 시험 녹화를 오라클 시간 축으로 만드세요(scenario --timed)")
    truth_snaps = [l["snapshot"] for l in route_lines(oracle.fold_yarn(truth["route"]))]
    got_lines = staging["lines"]
    got_cmds = [list(staging["header"]) + list(got_lines[0]["commands"])] + [list(l["commands"]) for l in got_lines[1:]]
    got_snaps = [l["snapshot"] for l in route_lines(oracle.fold_yarn([staging["files"]["yarn"]]))]
    pairs = match_lines([l["head"] for l in truth_lines], [l["head"] for l in got_lines])
    rows = []
    total = dur_ok = start_ok = 0
    ease_total = ease_ok = 0
    for ti, gi in pairs:
        t_before = truth_snaps[ti - 1] if ti else Snapshot([], {}, [])
        g_before = got_snaps[gi - 1] if gi else Snapshot([], {}, [])
        want = _tweens(truth_lines[ti]["schedule"], t_before, truth_snaps[ti])
        got_schedule = oracle.timeline(got_cmds[:gi + 1], [])["schedule"]
        have = _tweens(got_schedule, g_before, got_snaps[gi])
        for (who, kind), entries in want.items():
            for e in entries:
                total += 1
                cands = have.get((who, kind), [])
                if not cands:
                    rows.append({"truth": ti, "who": who, "kind": kind, "command": e["name"], "problem": "없음"})
                    continue
                g = min(cands, key=lambda c: abs(float(c["duration"]) - float(e["duration"])))
                dd = round((float(g["duration"]) - float(e["duration"])) * fps, 3)
                ds = round((float(g["start"]) - float(e["start"])) * fps, 3)
                dur_ok += abs(dd) <= tolerance_frames
                start_ok += abs(ds) <= tolerance_frames
                same_ease = kind == "alpha" or (g.get("ease") or "OutCubic").lower() == (e.get("ease") or "OutCubic").lower()
                if kind == "move":
                    ease_total += 1
                    ease_ok += same_ease
                if abs(dd) > tolerance_frames or abs(ds) > tolerance_frames or not same_ease:
                    rows.append({"truth": ti, "who": who, "kind": kind, "command": e["name"],
                                 "want": [round(float(e["start"]) * fps), round(float(e["duration"]) * fps), e.get("ease")],
                                 "got": [round(float(g["start"]) * fps), round(float(g["duration"]) * fps), g.get("ease")]})
    report = {
        "tweens": total,
        "durationWithin": dur_ok / total if total else 1.0,
        "startWithin": start_ok / total if total else 1.0,
        "easeMatch": ease_ok / ease_total if ease_total else 1.0,
        "moveTweens": ease_total,
        "problems": rows,
    }
    report["pass"] = report["durationWithin"] >= TIMING_PASS_DURATION and report["easeMatch"] >= TIMING_PASS_EASE
    return report


# ── P3: 남은 어휘 ─────────────────────────────────────────────

VOCAB_PASS = 0.9
_SHOT_NAMES = {"shot_to", "shot_zoom", "shot_track", "shot_reset", "shot_focus_to"}


def families(commands: Sequence[dict], header: bool = False) -> set:
    """줄 커맨드 → 어휘 갈래(카메라 · 모션 이름 · 슬라이드 · 캐릭터 비주얼 · 화면 효과). 머리의 샷 초기화는 뺀다."""
    out = set()
    for c in commands:
        n = c["name"]
        if n in _SHOT_NAMES:
            if header and n == "shot_reset":
                continue
            out.add("shot")
        elif n == "motion" and len(c.get("args", [])) > 1:
            out.add(f"motion:{c['args'][1]}")
        elif n in ("slide_in", "slide_out"):
            out.add(n)
        elif n.startswith("char_visual"):
            out.add(n)
        elif n.startswith("screen_") and not n.endswith("_clear") and not n.endswith("_reset"):
            out.add(n)
    return out


def evaluate_vocab(truth: dict, staging: dict) -> dict:
    truth_lines = truth["lines"]
    got_lines = staging["lines"]
    pairs = match_lines([l["head"] for l in truth_lines], [l["head"] for l in got_lines])
    want_n = got_n = hit = 0
    rows = []
    for ti, gi in pairs:
        w = families(truth_lines[ti]["commands"], header=ti == 0)
        g = families(got_lines[gi]["commands"])
        want_n += len(w)
        got_n += len(g)
        hit += len(w & g)
        if w != g:
            rows.append({"truth": ti, "missing": sorted(w - g), "extra": sorted(g - w)})
    report = {"wanted": want_n, "found": got_n, "hit": hit,
              "recall": hit / want_n if want_n else 1.0, "precision": hit / got_n if got_n else 1.0,
              "problems": rows}
    report["pass"] = report["recall"] >= VOCAB_PASS and report["precision"] >= VOCAB_PASS
    return report


# ── P4: 모르는 카탈로그 — 이름을 맞춰 보고 견준다 ──────────────────

UNKNOWN_IOU = 0.85        # 무대 안에 보이는 그림 모양(α)이 이만큼 겹쳐야 자리가 같다


def _visible_masks(snap: Snapshot, catalog, painter, rows: int):
    """보이는 인물마다 (인물, 변형, 표정, 무대 안 α>0.5 가림) — 분석 배율, 대사창 위만."""
    out = []
    for s in snap.visible_slots(0.5):
        if s.get("background") or not s.get("character") or not s.get("quad"):
            continue
        entry = catalog.portrait(s["character"], s.get("variant") or "a", s.get("emotion") or "")
        if entry is None:
            continue
        _, alpha = painter.layer(entry, np.asarray(s["quad"], np.float64), bool(s.get("toolMirror")))
        m = alpha > 0.5
        m[rows:] = False
        if m.sum() >= 30:
            out.append((s["character"], s.get("variant") or "a", s.get("emotion") or "", m))
    return out


def evaluate_unknown(truth: dict, staging: dict, oracle: Oracle, truth_catalog, got_catalog,
                     stage_bottom: float, min_iou: float = UNKNOWN_IOU) -> dict:
    """영상에서 배운 카탈로그(이름이 c01 · 01 …)로 뽑은 결과를 정답과 견준다(P4).

    두 카탈로그의 그림은 여백 · 크기가 달라 화면 사각형으로는 못 견준다. 줄마다 대사창 위에 실제로 보이는 인물 모양(α)을
    그려 가장 많이 겹치는 것끼리 짝짓고, 짝을 모아 다수결로 이름 대응(뽑은 인물 → 정답 인물, (뽑은 인물, 표정) →
    (정답 인물, 표정), 뽑은 배경 → 정답 배경)을 정한다. 순도: 그 대응을 따르는 짝의 비율.
    줄 일치: 대응으로 바꿔 읽은 배경 · 인물 · 표정이 같고, 짝마다 모양이 min_iou 이상 겹친다."""
    from collections import Counter, defaultdict
    from .observe import ANALYSIS_SCALE, Painter
    size = (int(round(oracle.width * ANALYSIS_SCALE)), int(round(oracle.height * ANALYSIS_SCALE)))
    rows = int(stage_bottom * ANALYSIS_SCALE)
    tpaint, gpaint = Painter(truth_catalog, size, ANALYSIS_SCALE), Painter(got_catalog, size, ANALYSIS_SCALE)
    truth_snaps = [l["snapshot"] for l in route_lines(oracle.fold_yarn(truth["route"]))]
    got_snaps = [l["snapshot"] for l in route_lines(oracle.fold_yarn([staging["files"]["yarn"]]))]
    pairs = match_lines([l["head"] for l in truth["lines"]], [l["head"] for l in staging["lines"]])

    def iou(a, b):
        return float((a & b).sum()) / max(1, int((a | b).sum()))

    matched_rows = []
    who, face, bgs = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for ti, gi in pairs:
        tb, _ = _visible(truth_snaps[ti])
        gb, _ = _visible(got_snaps[gi])
        if gb is not None:
            bgs[gb][tb] += 1
        tp = _visible_masks(truth_snaps[ti], truth_catalog, tpaint, rows)
        gp = _visible_masks(got_snaps[gi], got_catalog, gpaint, rows)
        used = set()
        links = []
        for g in gp:
            best = None
            for k, t in enumerate(tp):
                if k in used:
                    continue
                v = iou(g[3], t[3])
                if v > 0.3 and (best is None or v > best[0]):
                    best = (v, k)
            if best is not None:
                used.add(best[1])
                t = tp[best[1]]
                links.append((g, t, best[0]))
                who[g[0]][t[0]] += 1
                face[(g[0], g[1], g[2])][(t[0], t[1], t[2])] += 1
        matched_rows.append((ti, gi, tb, gb, tp, gp, links))
    char_map = {g: c.most_common(1)[0][0] for g, c in who.items()}
    face_map = {g: c.most_common(1)[0][0] for g, c in face.items()}
    bg_map = {g: c.most_common(1)[0][0] for g, c in bgs.items()}
    total = sum(sum(c.values()) for c in who.values())
    identity = sum(c.most_common(1)[0][1] for c in who.values()) / total if total else 0.0
    expression = sum(c.most_common(1)[0][1] for c in face.values()) / total if total else 0.0
    ok = 0
    rows = []
    for ti, gi, tb, gb, tp, gp, links in matched_rows:
        problems = []
        if (bg_map.get(gb) if gb is not None else None) != tb:
            problems.append(f"배경 {tb} ≠ {gb}")
        if len(tp) != len(gp) or len(links) != len(tp):
            problems.append(f"인물 수 {len(tp)} ≠ {len(gp)}")
        for g, t, v in links:
            if face_map.get((g[0], g[1], g[2])) != (t[0], t[1], t[2]):
                problems.append(f"{t[0]}/{t[1]}/{t[2]} ≠ {g[0]}/{g[1]}/{g[2]}")
            if v < min_iou:
                problems.append(f"{t[0]} 모양 겹침 {v:.2f}")
        ok += not problems
        rows.append({"truth": ti, "got": gi, "state": problems})
    n = len(truth["lines"])
    ious = [v for *_, links in matched_rows for _, _, v in links]
    report = {"truthLines": n, "matched": len(pairs), "lineMatch": ok / n if n else 0.0,
              "meanIoU": round(float(np.mean(ious)), 4) if ious else 0.0,
              "identityPurity": round(identity, 4), "expressionPurity": round(expression, 4),
              "characters": {g: t for g, t in sorted(char_map.items())},
              "expressions": {"/".join(g): "/".join(t) for g, t in sorted(face_map.items())},
              "backgrounds": {g: t for g, t in sorted(bg_map.items())},
              "lines": rows}
    report["pass"] = identity >= 1.0 and expression >= 0.95 and report["lineMatch"] >= 0.9
    return report
