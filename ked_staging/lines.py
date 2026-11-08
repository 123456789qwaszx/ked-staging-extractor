"""줄 경계 — 대사창 본문 자리의 글자(잉크)가 지워지는 순간을 줄 머리로 잡는다.

런타임은 줄 머리에 커맨드 · 대사창 전환 · 타자기를 한꺼번에 시작한다(LinePresentationFlow). 그래서 화면에서
줄 머리가 가장 또렷하게 보이는 곳은 본문이다 — 앞 줄 글자가 한꺼번에 사라지고 새 글자가 하나씩 늘어난다.

- 잉크: 본문 · 이름 자리에서 주변보다 확 밝거나 어두운 가는 획(top-hat · black-hat). 글자색을 가정하지 않는다.
- 지움(clear): 앞 프레임 잉크의 절반 넘게가 사라짐 → 그 프레임이 줄 머리.
- 등장(onset): 한동안 비어 있던 본문에 잉크가 생김 → 줄 머리. 바로 앞에 대사창 판이 나타난 프레임이 있으면 그쪽.
- 다 찍힘(typed): 줄 안에서 잉크가 마지막으로 늘어난 프레임.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .dialogue import BoxLayout

WORK_SCALE = 0.5          # 분석 배율
INK_THRESHOLD = 60        # 획 대비(0..255)
CLEAR_RATIO = 0.5         # 앞 잉크 중 이만큼 사라지면 지움
MIN_INK_FRACTION = 0.0015  # 본문 자리 대비 "글자가 있다"고 볼 잉크 비율
PANEL_JUMP = 10.0          # 대사창 판이 나타났다고 볼 평균 밝기 변화
MIN_GROWTH = 20            # "글자가 늘었다"고 볼 새 잉크(분석 배율 px) — 잡음 · 무대 전이를 거른다


@dataclass
class LineSpan:
    head: float
    typed: float
    end: float
    kind: str          # "clear" | "onset"
    ink: int           # 다 찍혔을 때 잉크 픽셀 수(분석 배율)

    def to_json(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _crop(gray: np.ndarray, rect, scale: float) -> np.ndarray:
    x0, y0, x1, y1 = (int(round(v * scale)) for v in rect)
    return gray[max(0, y0):y1, max(0, x0):x1]


def ink_mask(region: np.ndarray, stroke: int = 7, threshold: int = INK_THRESHOLD) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (stroke, stroke))
    top = cv2.morphologyEx(region, cv2.MORPH_TOPHAT, k)
    black = cv2.morphologyEx(region, cv2.MORPH_BLACKHAT, k)
    return np.maximum(top, black) > threshold


@dataclass
class InkTrack:
    times: List[float]
    ink: np.ndarray        # 프레임마다 잉크 수
    removed: np.ndarray    # 앞 프레임 잉크 중 사라진 수
    added: np.ndarray      # 새로 생긴 수
    panel: np.ndarray      # 대사창 판(글자 자리 밖) 평균 밝기 변화
    area: int              # 본문 자리 픽셀 수


def track_ink(video: str, layout: BoxLayout, scale: float = WORK_SCALE, log=None) -> InkTrack:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"영상을 열지 못했습니다: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    box = layout.scaled(w, h)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stroke = max(3, int(round(box.line_font * scale * 0.2)) | 1)
    dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    px0, py0, px1, py1 = box.panel()
    times, ink, removed, added, panel = [], [], [], [], []
    prev_mask: Optional[np.ndarray] = None
    prev_masks: List[np.ndarray] = []
    prev_panel: Optional[np.ndarray] = None
    index = 0
    area = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and index > 0:
            t = index / fps
        sub = frame[int(py0):int(np.ceil(py1)), int(px0):int(np.ceil(px1))]
        small = cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        off = (px0, py0, px0, py0)
        rects = [tuple(r - o for r, o in zip(box.line, off))]
        if box.name:
            rects.append(tuple(r - o for r, o in zip(box.name, off)))
        masks = [ink_mask(_crop(gray, r, scale), stroke) for r in rects]
        mask = np.concatenate([m.ravel() for m in masks])
        area = masks[0].size
        if prev_mask is None:
            rem = add = 0
        else:
            grown_prev = np.concatenate([cv2.dilate(m.astype(np.uint8), dil).ravel() for m in prev_masks])
            grown_cur = np.concatenate([cv2.dilate(m.astype(np.uint8), dil).ravel() for m in masks])
            rem = int((prev_mask & (grown_cur == 0)).sum())
            add = int((mask & (grown_prev == 0)).sum())
        g = gray.astype(np.float32)
        panel.append(0.0 if prev_panel is None else float(np.abs(g - prev_panel).mean()))
        prev_panel = g
        prev_mask, prev_masks = mask, masks
        times.append(t)
        ink.append(int(mask.sum()))
        removed.append(rem)
        added.append(add)
        index += 1
    cap.release()
    if log:
        log(f"대사창 잉크 추적: 프레임 {index}")
    return InkTrack(times, np.asarray(ink), np.asarray(removed), np.asarray(added), np.asarray(panel), area)


def detect_lines(track: InkTrack, min_gap: float = 0.25) -> List[LineSpan]:
    """잉크 기록 → 줄 구간들."""
    n = len(track.times)
    if n == 0:
        return []
    present = max(12, int(track.area * MIN_INK_FRACTION))
    heads: List[Tuple[int, str]] = []
    empty_run = 0
    for i in range(1, n):
        before = track.ink[i - 1]
        if before >= present and track.removed[i] > CLEAR_RATIO * before:
            heads.append((i, "clear"))
            empty_run = 0
            continue
        if track.ink[i] < present:
            empty_run += 1
            continue
        if empty_run >= 2:
            # 비어 있던 본문에 글자가 생겼다 — 판이 막 나타난 프레임이 있으면 그것이 머리다.
            j = i
            lo = max(1, i - 12)
            jumps = [k for k in range(lo, i + 1) if track.panel[k] >= PANEL_JUMP]
            if jumps:
                j = jumps[0]
            if not heads or track.times[j] - track.times[heads[-1][0]] >= min_gap:
                heads.append((j, "onset"))
        empty_run = 0
    # 너무 붙은 머리(지움 직후 흔들림)는 앞의 것만 둔다.
    merged: List[Tuple[int, str]] = []
    for i, kind in heads:
        if merged and track.times[i] - track.times[merged[-1][0]] < min_gap:
            continue
        merged.append((i, kind))
    spans: List[LineSpan] = []
    for k, (i, kind) in enumerate(merged):
        stop = merged[k + 1][0] if k + 1 < len(merged) else n
        # 줄이 끝나는 곳: 다음 머리, 또는 글자가 모두 사라진 곳(대사창이 닫힘)
        end_i = stop
        for j in range(i + 1, stop):
            if track.ink[j - 1] >= present and track.ink[j] < present and track.removed[j] > CLEAR_RATIO * track.ink[j - 1]:
                end_i = j
                break
        grow = [j for j in range(i, end_i) if track.added[j] >= MIN_GROWTH and track.ink[j] >= present]
        typed_i = grow[-1] if grow else i
        if max(track.ink[i:end_i], default=0) < present:
            continue  # 글자가 끝내 안 생긴 구간 — 줄이 아니다
        step = (track.times[-1] - track.times[0]) / max(1, n - 1) if n > 1 else 0.0
        end_t = track.times[end_i] if end_i < n else track.times[-1] + step
        # 줄 머리는 앞 프레임과 이 프레임 사이 어딘가다 — 반 프레임 앞으로 둔다(기댓값). 시간 재기의 시작이 덜 치우친다.
        spans.append(LineSpan(track.times[i] - 0.5 * step, track.times[typed_i], end_t, kind,
                              int(track.ink[typed_i])))
    return spans
