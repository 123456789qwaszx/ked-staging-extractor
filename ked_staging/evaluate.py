"""평가 — 추출한 .yarn을 오라클로 다시 접어 정답 대본과 줄마다 비교한다(P1 통과 기준: 상태 일치 95% 이상).

줄 짝짓기: 추출 줄 머리와 정답 줄 머리를 시간 순서대로 짝짓는다(차이 ≤ 0.25초).
상태 일치(줄마다): 보이는 배경 그림이 같고, 보이는 인물 집합이 같고, 인물마다 변형 · 표정 · 반전이 같고
화면 사각형(bounds)의 네 값이 모두 허용(기본 6px) 안이다. 그리기 순서와 시간은 보지 않는다(P2 · P3).
대사: 화자 일치, 본문 완전 일치(띄어쓰기 무시), 글자 정확도(편집 거리).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

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
