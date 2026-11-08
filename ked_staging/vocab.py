"""남은 어휘(P3) — 줄 안에서만 보이고 정착 화면에는 남지 않는 움직임을 찾는다.

정착 화면 관측(P1)과 시간 재기(P2)가 끝난 줄을 오라클 시간 축으로 다시 그려 영상과 프레임마다 견준다.
슬롯 자리에서 '지나가는' 오차가 크면 후보 커맨드를 끼워 보고(합성으로 해석하기) 가장 잘 맞는 것을 고른다.

- 제자리 모션(restore — 예: nod): 보인 채 그대로인 슬롯. 모션 이름 × 시작(0~1초, 3프레임 간격, 늦으면 pause).
- 슬라이드: 등장(fade_in)하며 미끄러져 들어오면 slide_in, 퇴장(fade_out)하며 나가면 slide_out —
  방향 × 거리(u) 후보, 길이 · 이징은 그 줄의 페이드와 같다고 본다(샘플 대본 관례).

commit 모션(예: bow)은 정착 화면에 자세가 남으므로 관측(P1)이 자리 후보로 찾는다.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .catalog import Catalog
from .observe import Painter, robust_error
from .oracle import Oracle
from .timing import compose

TRANSIENT_ERROR = 6.0       # 슬롯 자리 평균 오차가 정착 때보다 이만큼 크면 '지나가는 움직임'을 의심한다
SLIDE_HINT_GAIN = 0.15      # 대표 슬라이드가 오차를 이만큼이라도 줄이면 제대로 찾아본다
ACCEPT_GAIN = 0.3           # 후보가 오차를 이 비율 이상 줄여야 받아들인다
SLIDE_DIRECTIONS = ("left", "right")
SLIDE_DISTANCES = (1, 2, 3, 4, 6, 8, 12)
MOTION_STARTS_FR = range(0, 25, 3)


@dataclass
class Finding:
    slot: str
    kind: str              # "motion" | "slide_in" | "slide_out" | "unexplained"
    commands: List[dict]   # 끼운 커맨드(모션은 필요하면 pause 포함)
    gain: float
    note: str = ""
    line: Optional[List[dict]] = None   # 줄을 통째로 고친 경우(슬라이드: 페이드 자리 · 길이까지)


class TransientFinder:
    def __init__(self, oracle: Oracle, catalog: Catalog, painter: Painter, mask: np.ndarray,
                 motions: Sequence[dict]):
        self.oracle = oracle
        self.catalog = catalog
        self.painter = painter
        self.mask = mask
        self.restores = [m for m in motions if m["end"] == "restore"]

    def _errors(self, history, line, frames, regions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """줄 시간 축의 예측과 프레임마다 슬롯 자리 평균 오차 → 슬롯 → (프레임 수,)"""
        times = [t for t, _ in frames]
        snaps = self.oracle.timeline(list(history) + [list(line)], times)["snapshots"]
        out = {k: np.zeros(len(frames)) for k in regions}
        for i, ((_, f), sn) in enumerate(zip(frames, snaps)):
            img, _ = compose(sn, self.catalog, self.painter)
            err = robust_error(f, img)
            for k, reg in regions.items():
                out[k][i] = float(err[reg].mean()) if reg.any() else 0.0
        return out

    def _regions(self, history, line, frames, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        snaps = self.oracle.timeline(list(history) + [list(line)], [0.0, frames[-1][0]])["snapshots"]
        regions = {k: np.zeros(self.mask.shape, bool) for k in keys}
        for sn in snaps:
            for slot in sn.slots:
                if slot["key"] in regions and slot.get("quad"):
                    entry = self.catalog.portrait(slot.get("character") or "", slot.get("variant") or "a",
                                                  slot.get("emotion") or "")
                    if entry is None:
                        continue
                    _, alpha = self.painter.layer(entry, np.asarray(slot["quad"], np.float64), bool(slot.get("toolMirror")))
                    regions[slot["key"]] |= alpha > 0.5
        # 슬롯 자리를 넓혀(움직임이 지나가는 둘레) 대사창 위로 자른다.
        import cv2
        for k in keys:
            regions[k] = cv2.dilate(regions[k].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & (self.mask > 0)
        return regions

    def find(self, history: Sequence[Sequence[dict]], line: Sequence[dict], frames: Sequence[Tuple[float, np.ndarray]],
             static: Sequence[str], appearing: Sequence[str], leaving: Sequence[str]) -> List[Finding]:
        if not frames or not (static or appearing or leaving):
            return []
        keys = list(dict.fromkeys(list(static) + list(appearing) + list(leaving)))
        regions = self._regions(history, line, frames, keys)
        base = self._errors(history, line, frames, regions)
        dt = (frames[1][0] - frames[0][0]) if len(frames) > 1 else 1 / 30
        # 줄 머리는 반 프레임 안에서만 안다 — 큰 회전 · 빠른 움직임은 그 차이에도 크게 틀린다. 그래서 견줄 때마다
        # 프레임 시각을 ±반 프레임 안에서 맞춰 본다(가장 잘 맞는 시각의 오차를 쓴다).
        lags = [i * dt / 4 for i in range(-2, 3)]

        floors = {k: float(np.median(base[k][-max(3, len(base[k]) // 5):])) for k in keys}

        def excess(key, e) -> float:
            """정착 바닥(압축 · 잡음)을 뺀 오차 합 — 긴 정착 구간이 움직임 구간의 차이를 묽히지 않게."""
            return float(np.clip(e - floors[key], 0, None).sum())

        def best_over_lags(cand_line, key) -> Tuple[float, np.ndarray]:
            out = None
            for lag in lags:
                e = self._errors(history, cand_line, [(t + lag, f) for t, f in frames], {key: regions[key]})[key]
                c = excess(key, e)
                if out is None or c < out[0]:
                    out = (c, e)
            return out

        findings: List[Finding] = []
        for key in keys:
            if self._transient(base[key]) < TRANSIENT_ERROR:
                if key in static or not self._slide_hint(history, line, frames, key, regions[key], base[key],
                                                          "slide_in" if key in appearing else "slide_out",
                                                          floors[key]):
                    continue
            total0, e0 = best_over_lags(line, key)
            if key in static and self._transient(e0) < TRANSIENT_ERROR:
                continue   # 시각만 어긋났다
            if key in static:
                trials = []
                for m in self.restores:
                    for st in MOTION_STARTS_FR:
                        add = ([{"name": "pause", "args": [f"{st}fr"]}] if st else []) + \
                              [{"name": "motion", "args": [key, m["name"]]}]
                        cand = self._insert(line, add, "motion")
                        e = self._errors(history, cand, frames, {key: regions[key]})[key]
                        trials.append((excess(key, e), add))
                trials.sort(key=lambda x: x[0])
                best = None
                for _, add in trials[:3]:
                    total, _ = best_over_lags(self._insert(line, add, "motion"), key)
                    if best is None or total < best[0]:
                        best = (total, add)
                gain = 1.0 - best[0] / max(total0, 1e-6) if best else 0.0
                if best and gain >= ACCEPT_GAIN:
                    findings.append(Finding(key, "motion", best[1], gain))
                    line = self._insert(line, best[1], "motion")
                    continue
                findings.append(Finding(key, "unexplained", [], gain,
                                        f"{key}: 줄 안에서 지나가는 움직임을 어휘로 설명하지 못했습니다"))
            else:
                slide = self._slide(history, line, frames, key, regions[key],
                                    "slide_in" if key in appearing else "slide_out", floors[key])
                if slide is not None:
                    _, kind, new_line = slide
                    total, _ = best_over_lags(new_line, key)
                    gain = 1.0 - total / max(total0, 1e-6)
                    if gain >= ACCEPT_GAIN:
                        findings.append(Finding(key, kind, [c for c in new_line if c["name"] == kind], gain,
                                                line=new_line))
                        line = new_line
                        continue
                findings.append(Finding(key, "unexplained", [], 0.0,
                                        f"{key}: 등 · 퇴장하며 지나가는 움직임을 어휘로 설명하지 못했습니다"))
        return findings

    def _slide_hint(self, history, line, frames, key, region, base_e, kind, floor: float = 0.0) -> bool:
        """등 · 퇴장은 페이드가 움직임을 가려 지나가는 오차가 작다 — 대표 슬라이드 둘(좌 · 우 4u)로 먼저 떠본다."""
        fade_name = "fade_in" if kind == "slide_in" else "fade_out"
        fade = next((c for c in line if c["name"] == fade_name and c["args"][:1] == [key]), None)
        if fade is None:
            return False
        dur = fade["args"][1] if len(fade["args"]) > 1 else "14fr"
        total0 = float(np.clip(base_e - floor, 0, None).sum())
        for d in SLIDE_DIRECTIONS:
            cand = self._insert(line, [{"name": kind, "args": [key, d, "4u", dur]}], kind)
            e = self._errors(history, cand, frames, {key: region})[key]
            if float(np.clip(e - floor, 0, None).sum()) < total0 * (1 - SLIDE_HINT_GAIN):
                return True
        return False

    @staticmethod
    def _transient(e: np.ndarray) -> float:
        """지나가는 움직임의 크기 — 처음 · 끝보다 가운데가 얼마나 더 틀리나(처음부터 틀린 것은 시간 어긋남이지 움직임이 아니다)."""
        if len(e) == 0:
            return 0.0
        settled = float(np.median(e[-max(3, len(e) // 5):]))
        return float(e.max() - max(settled, float(e[0])))

    def _slide(self, history, line, frames, key: str, region: np.ndarray, kind: str, floor: float = 0.0):
        """등 · 퇴장하는 슬롯의 슬라이드 — 방향 × 거리 × 길이를 찾는다. 길이는 같은 슬롯의 페이드와 같게 둔다(샘플 관례).
        → (오차 합, 종류, 슬라이드를 넣고 페이드 길이도 맞춘 줄) 또는 None"""
        fade_name = "fade_in" if kind == "slide_in" else "fade_out"
        fi = next((i for i, c in enumerate(line) if c["name"] == fade_name and c["args"][:1] == [key]), None)
        if fi is None:
            return None
        cur = line[fi]["args"][1] if len(line[fi]["args"]) > 1 else "14fr"
        try:
            cur_fr = int(round(float(cur[:-2]))) if cur.endswith("fr") else 14
        except ValueError:
            cur_fr = 14

        # 시간 재기가 페이드 앞에 끼운 대기는 슬라이드가 섞인 투명도를 잘못 잰 것일 수 있다 — 페이드를 줄 머리 무리로 옮긴다.
        rest = [copy.deepcopy(c) for i, c in enumerate(line) if i != fi]
        while rest and rest[-1]["name"] == "pause":
            rest.pop()
        head_end = next((i for i, c in enumerate(rest) if c["name"] == "pause"), len(rest))

        def build(direction: str, dist: int, frames_: int) -> List[dict]:
            slide = {"name": kind, "args": [key, direction, f"{dist}u", f"{frames_}fr"]}
            fade = {"name": fade_name, "args": [key, f"{frames_}fr"]}
            return rest[:head_end] + [slide, fade] + rest[head_end:]

        def cost(cand: List[dict]) -> float:
            return float(np.clip(self._errors(history, cand, frames, {key: region})[key] - floor, 0, None).sum())

        coarse = sorted({cur_fr, 8, 12, 16, 20})
        best = None
        for d in SLIDE_DIRECTIONS:
            for dist in SLIDE_DISTANCES:
                for fr in coarse:
                    c = build(d, dist, fr)
                    total = cost(c)
                    if best is None or total < best[0]:
                        best = (total, d, dist, fr)
        _, d, dist, fr = best
        for fr2 in range(max(2, fr - 3), fr + 4):
            if fr2 == fr:
                continue
            total = cost(build(d, dist, fr2))
            if total < best[0]:
                best = (total, d, dist, fr2)
        total, d, dist, fr = best
        return total, kind, build(d, dist, fr)

    @staticmethod
    def _insert(line: Sequence[dict], add: Sequence[dict], kind: str) -> List[dict]:
        """모션은 줄 끝에(대기는 그 앞), 슬라이드는 같은 슬롯의 페이드 바로 앞에 끼운다."""
        out = [copy.deepcopy(c) for c in line]
        if kind in ("slide_in", "slide_out"):
            key = add[0]["args"][0]
            for i, c in enumerate(out):
                if c["name"] in ("fade_in", "fade_out") and c["args"][:1] == [key]:
                    return out[:i] + [copy.deepcopy(a) for a in add] + out[i:]
        return out + [copy.deepcopy(a) for a in add]


def apply_findings(line: Sequence[dict], findings: Sequence[Finding]) -> List[dict]:
    """찾은 커맨드를 줄에 끼운다. 슬라이드는 같은 슬롯 페이드의 길이도 슬라이드와 같게 맞춘다."""
    out = [copy.deepcopy(c) for c in line]
    for f in findings:
        if f.line is not None:
            out = [copy.deepcopy(c) for c in f.line]
            continue
        if not f.commands:
            continue
        if f.kind in ("slide_in", "slide_out"):
            slide = f.commands[0]
            fade_name = "fade_in" if f.kind == "slide_in" else "fade_out"
            for c in out:
                if c["name"] == fade_name and c["args"][:1] == [f.slot]:
                    c["args"] = [f.slot, slide["args"][3]]
        out = TransientFinder._insert(out, f.commands, f.kind)
    return out
