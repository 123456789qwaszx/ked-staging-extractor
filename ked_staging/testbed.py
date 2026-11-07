"""시험장 — 원본 .yarn을 아는 런타임 녹화로 오라클과 (나중에) 추출기를 잰다.

P0에서 하는 일:
  1. 녹화 경로(재생 순서의 yarn 파일들)를 오라클로 접어 "보이는 상태"마다 예측 화면을 만든다.
  2. 영상 프레임과 예측 화면을 대조해 상태마다 영상의 구간을 찾는다(단조 정렬 — 순서는 대본이 안다).
  3. 상태마다 가장 잘 맞는 프레임에서 슬롯별 어긋남(px)을 잰다 → P0 통과 기준(1px 이내).

대사창은 예측하지 않으므로 대조에서 가린다(surface-layout의 대사창 높이 + 여유).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .catalog import Catalog
from .oracle import Oracle, Snapshot, route_lines
from .render import composite, coverages, render, render_layers

SCHEMA = "ked-staging-testbed/1"
ANALYSIS_WIDTH = 480          # 정렬용 축소 폭(px)
SAMPLE_FPS = 8.0              # 정렬용 표본 주기
MIN_MEASURE_PIXELS = 4000     # 슬롯 어긋남을 재려면 보이는 그림이 이만큼은 있어야 한다
EDGE_ERODE = 8                # 측정 영역 가장자리에서 깎는 폭(px)


# ── 매니페스트 ──────────────────────────────────────────────────

@dataclass
class Recording:
    video: str
    route: List[str]
    note: str = ""


@dataclass
class Manifest:
    path: str
    runtime: str
    recordings: List[Recording]

    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{SCHEMA} 형식이 아닙니다: {data.get('schema')!r}")
        base = os.path.dirname(os.path.abspath(path))
        runtime = os.path.normpath(os.path.join(base, data["runtime"]))
        recs = []
        for r in data["recordings"]:
            recs.append(Recording(
                video=os.path.normpath(os.path.join(base, r["video"])),
                route=[os.path.normpath(os.path.join(runtime, p)) for p in r["route"]],
                note=r.get("note", "")))
        return cls(os.path.abspath(path), runtime, recs)


# ── 보이는 상태 ─────────────────────────────────────────────────

@dataclass
class VisualState:
    """같아 보이는 연속 줄 묶음. 대사창을 가리면 이 줄들은 화면으로 구별할 수 없다."""
    first: int
    last: int
    snapshot: Snapshot
    lines: List[dict]


def _signature(snapshot: Snapshot) -> Tuple:
    sig = []
    for s in snapshot.visible_slots():
        quad = tuple(round(v, 1) for p in s["quad"] for v in p)
        ident = s.get("sprite") if s.get("background") else (s.get("character"), s.get("variant"), s.get("emotion"))
        sig.append((s["key"], ident, quad, round(s.get("alpha", 1.0), 2), bool(s.get("toolMirror"))))
    return tuple(sig)


def visual_states(lines: Sequence[dict]) -> List[VisualState]:
    states: List[VisualState] = []
    for i, line in enumerate(lines):
        snap: Snapshot = line["snapshot"]
        if states and _signature(states[-1].snapshot) == _signature(snap):
            states[-1].last = i
            states[-1].lines.append(line)
        else:
            states.append(VisualState(i, i, snap, [line]))
    return states


def merge_identical(states: List[VisualState], predictions: List[np.ndarray], mask: np.ndarray,
                    tolerance: float = 1.0) -> Tuple[List[VisualState], List[np.ndarray]]:
    """예측 화면이 사실상 같은 연속 상태를 하나로 합친다(대조 영역에서 픽셀 차이가 tolerance 이하).

    같은 그림을 쓰는 표정 교체(견본 초상은 표정 그림이 모두 같다)나 보이지 않는 pose처럼, 키가 바뀌어도
    화면이 같으면 영상으로는 구별할 수 없다.
    """
    out_s: List[VisualState] = []
    out_p: List[np.ndarray] = []
    w = mask > 0
    for s, p in zip(states, predictions):
        if out_p and float(np.abs(out_p[-1] - p)[w].max()) <= tolerance:
            last = out_s[-1]
            last.last = s.last
            last.lines.extend(s.lines)
            continue
        out_s.append(VisualState(s.first, s.last, s.snapshot, list(s.lines)))
        out_p.append(p)
    return out_s, out_p


# ── 대사창 가림 ─────────────────────────────────────────────────

def box_top_ratio(tuning_dir: str, preset: str = "bottom", margin: float = 0.06) -> float:
    """대사창이 차지하는 아래쪽 비율(0..1). surface-layout의 줄 · 이름 영역 위 끝 + 여유."""
    path = os.path.join(tuning_dir, "presets", "surface-layout.json")
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        body = data.get("MonoBehaviour", data)
        for entry in body.get("entries", []):
            if entry.get("key") == preset:
                tops = [entry["lineAnchorMax"]["y"]]
                if entry.get("useName"):
                    tops.append(entry["nameAnchorMax"]["y"])
                return min(1.0, max(tops) + margin)
    except (OSError, KeyError, ValueError):
        pass
    return 0.4


def stage_mask(width: int, height: int, box_ratio: float) -> np.ndarray:
    """대조에 쓰는 영역(1) — 아래쪽 대사창 띠를 뺀다."""
    mask = np.ones((height, width), np.float32)
    mask[int(round(height * (1.0 - box_ratio))):, :] = 0.0
    return mask


# ── 정렬 ───────────────────────────────────────────────────────

def _prep(img: np.ndarray, width: int) -> np.ndarray:
    h = int(round(img.shape[0] * width / img.shape[1]))
    small = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def state_weights(predictions: Sequence[np.ndarray], mask: np.ndarray, floor: float = 2.0) -> np.ndarray:
    """상태끼리 다른 픽셀에 무게를 싣는다 — 표정 교체처럼 작은 변화도 정렬이 놓치지 않게.

    무게 = 대조 영역 × (상태 예측들 사이의 표준편차 + floor). 모든 상태에서 같은 배경은 거의 무시된다.
    """
    stack = np.stack(predictions, 0)
    w = (stack.std(0) + floor) * mask
    return w / max(float(w.sum()), 1e-6)


def emission(frames: Sequence[np.ndarray], predictions: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """(프레임, 상태)별 가중 제곱 오차의 음수 — 클수록 잘 맞는다."""
    out = np.empty((len(frames), len(predictions)), np.float32)
    for fi, f in enumerate(frames):
        for si, p in enumerate(predictions):
            d = f - p
            out[fi, si] = -float((weights * d * d).sum())
    return out


def sample_video(path: str, fps: float = SAMPLE_FPS, width: int = ANALYSIS_WIDTH
                 ) -> Tuple[List[float], List[np.ndarray], float, Tuple[int, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"영상을 열지 못했습니다: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    step = max(1, int(round(src_fps / fps)))
    times, frames = [], []
    index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                times.append(index / src_fps)
                frames.append(_prep(frame, width))
        index += 1
    cap.release()
    return times, frames, src_fps, size


def align(similarity: np.ndarray, outside: np.ndarray, skip_penalty: Optional[float] = None) -> List[int]:
    """프레임마다 상태 번호(-1 = 첫 상태 전 · len = 끝 뒤)를 정한다. 상태 순서는 바뀌지 않는다.

    similarity: (프레임 수, 상태 수), outside: 프레임마다 "어느 상태도 아님"(검은 화면 · 타이틀)의 점수.
    상태 하나를 건너뛰는 것(너무 짧게 지나간 줄)은 벌점을 받고 허용한다.
    """
    F, S = similarity.shape
    if skip_penalty is None:
        spread = float(np.median(similarity.max(1) - np.median(similarity, 1)))
        skip_penalty = 3.0 * max(spread, 1e-6)
    states = S + 2                      # 0 = 전, 1..S = 상태, S+1 = 뒤
    score = np.full((F, states), -np.inf)
    back = np.zeros((F, states), np.int32)

    def gain(f: int, s: int) -> float:
        return float(outside[f]) if s in (0, S + 1) else float(similarity[f, s - 1])

    score[0, 0] = gain(0, 0)
    score[0, 1] = gain(0, 1)
    for f in range(1, F):
        for s in range(states):
            best, arg = score[f - 1, s], s
            if s >= 1 and score[f - 1, s - 1] > best:
                best, arg = score[f - 1, s - 1], s - 1
            if s >= 2 and score[f - 1, s - 2] - skip_penalty > best:
                best, arg = score[f - 1, s - 2] - skip_penalty, s - 2
            score[f, s] = best + gain(f, s)
            back[f, s] = arg
    s = int(np.argmax(score[F - 1, S:S + 2]) + S)  # 마지막 상태나 그 뒤에서 끝난다
    path = [0] * F
    for f in range(F - 1, -1, -1):
        path[f] = s
        s = back[f, s]
    return [p - 1 for p in path]


# ── 어긋남 재기 ─────────────────────────────────────────────────

@dataclass
class SlotOffset:
    key: str
    dx: float
    dy: float
    scale: float
    pixels: int
    correlation: float

    @property
    def distance(self) -> float:
        return float(np.hypot(self.dx, self.dy))


def measure_slot(predicted: np.ndarray, actual: np.ndarray, cover: np.ndarray, region_mask: np.ndarray,
                 key: str) -> Optional[SlotOffset]:
    """예측과 실제를 슬롯이 보이는 영역에서만 맞춰 본다(ECC, 아핀). dx · dy는 실제가 예측보다 얼마나 옮겨졌나(px)."""
    m = ((cover > 0.95) & (region_mask > 0)).astype(np.uint8)
    # 경계를 깎는다 — 앞뒤 슬롯이 몇 px 어긋나도 그 가장자리가 이 슬롯의 측정에 섞이지 않게.
    m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EDGE_ERODE + 1, 2 * EDGE_ERODE + 1)))
    pixels = int(m.sum())
    if pixels < MIN_MEASURE_PIXELS:
        return None
    ys, xs = np.nonzero(m)
    pad = 48
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + pad + 1)
    tpl = cv2.cvtColor(predicted[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    inp = cv2.cvtColor(actual[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    crop_mask = m[y0:y1, x0:x1]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    # 이동은 평행 이동 모형으로 잰다 — 넓고 밋밋한 배경에서 아핀 모형은 배율과 이동을 서로 바꿔 먹는다.
    try:
        cc, shift = cv2.findTransformECC(tpl, inp, np.eye(2, 3, dtype=np.float32), cv2.MOTION_TRANSLATION,
                                         criteria, crop_mask, 5)
    except cv2.error:
        return SlotOffset(key, float("nan"), float("nan"), float("nan"), pixels, 0.0)
    # 배율은 참고값 — 이동을 초기값으로 아핀을 한 번 더 맞춘다.
    try:
        _, affine = cv2.findTransformECC(tpl, inp, shift.copy(), cv2.MOTION_AFFINE, criteria, crop_mask, 5)
        scale = float(np.sqrt(abs(np.linalg.det(affine[:, :2]))))
    except cv2.error:
        scale = float("nan")
    return SlotOffset(key, float(shift[0, 2]), float(shift[1, 2]), scale, pixels, float(cc))


# ── 한 녹화 전체 ────────────────────────────────────────────────

@dataclass
class StateResult:
    index: int
    lines: Tuple[int, int]
    first_text: str
    start: Optional[float]
    end: Optional[float]
    best_time: Optional[float]
    score: float
    offsets: List[SlotOffset] = field(default_factory=list)


def run_recording(rec: Recording, oracle: Oracle, catalog: Catalog, tuning_dir: str,
                  out_dir: Optional[str] = None, log=print) -> List[StateResult]:
    folded = oracle.fold_yarn(rec.route, continuous=True)
    lines = route_lines(folded)
    states = visual_states(lines)
    log(f"경로 {len(rec.route)}개 파일 · 줄 {len(lines)} · 보이는 상태 {len(states)}")

    times, frames, src_fps, size = sample_video(rec.video)
    if not frames:
        raise IOError(f"프레임을 읽지 못했습니다: {rec.video}")
    log(f"영상 {os.path.basename(rec.video)} · {size[0]}x{size[1]} · {src_fps:.2f}fps · 표본 {len(frames)}")
    box = box_top_ratio(tuning_dir)
    small_mask = stage_mask(frames[0].shape[1], frames[0].shape[0], box)
    scale_small = frames[0].shape[1] / oracle.width
    preds = [_prep(render(s.snapshot, catalog, oracle.width, oracle.height, scale=0.5), ANALYSIS_WIDTH)
             for s in states]
    states, preds = merge_identical(states, preds, small_mask)
    log(f"화면으로 구별되는 상태 {len(states)}")
    weights = state_weights(preds, small_mask)
    sim = emission(frames, preds, weights)
    outside = emission(frames, [np.zeros_like(preds[0])], weights)[:, 0]
    path = align(sim, outside)

    cap = cv2.VideoCapture(rec.video)
    full_mask = stage_mask(size[0], size[1], box)
    results: List[StateResult] = []
    for si, state in enumerate(states):
        frames_in = [fi for fi, s in enumerate(path) if s == si]
        text = (state.lines[0].get("text") or "")
        if not frames_in:
            results.append(StateResult(si, (state.first, state.last), text, None, None, None, 0.0))
            continue
        # 정착 프레임: 구간 뒤쪽 절반에서 가장 잘 맞는 것(전이는 구간 앞쪽에 있다). 합쳐진 상태(화면이 같은 줄들)는
        # 마지막 줄 쪽에서 재게 된다.
        tail = frames_in[len(frames_in) // 2:]
        best = max(tail, key=lambda fi: sim[fi, si])
        res = StateResult(si, (state.first, state.last), text, times[frames_in[0]], times[frames_in[-1]],
                          times[best], float(sim[best, si]))
        cap.set(cv2.CAP_PROP_POS_MSEC, times[best] * 1000.0)
        ok, actual = cap.read()
        if ok:
            if actual.shape[1] != int(oracle.width):
                actual = cv2.resize(actual, (int(oracle.width), int(oracle.height)), interpolation=cv2.INTER_AREA)
            layers = render_layers(state.snapshot, catalog, oracle.width, oracle.height)
            predicted = composite(layers, int(oracle.width), int(oracle.height))
            for key, cover in coverages(layers).items():
                off = measure_slot(predicted, actual, cover, full_mask, key)
                if off:
                    res.offsets.append(off)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                overlay = cv2.addWeighted(predicted, 0.5, actual, 0.5, 0)
                from .imageio import write_image
                write_image(os.path.join(out_dir, f"state_{si:03d}.jpg"),
                            np.hstack([cv2.resize(predicted, (640, 360)), cv2.resize(actual, (640, 360)),
                                       cv2.resize(overlay, (640, 360))]))
        results.append(res)
    cap.release()
    return results


def summarize(results: Sequence[StateResult], tolerance_px: float = 1.0) -> Dict[str, object]:
    offsets = [o for r in results for o in r.offsets if np.isfinite(o.distance)]
    found = [r for r in results if r.start is not None]
    worst = max((o.distance for o in offsets), default=float("nan"))
    return {
        "states": len(results),
        "statesFound": len(found),
        "slotsMeasured": len(offsets),
        "withinTolerance": sum(1 for o in offsets if o.distance <= tolerance_px),
        "worstPx": worst,
        "medianPx": float(np.median([o.distance for o in offsets])) if offsets else float("nan"),
        "pass": bool(offsets) and len(found) == len(results) and worst <= tolerance_px,
    }
