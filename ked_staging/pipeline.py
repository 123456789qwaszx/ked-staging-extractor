"""추출 한 번 — 영상(+ 게임텍스트추출기 엑셀) → 줄 · 대사 · 무대 관측 · 커맨드 → staging.json · .yarn.

  1. 줄 경계(lines)        대사창 본문 잉크로 줄 머리 · 다 찍힘 · 끝
  2. 대사 글자(gte)         엑셀 줄을 읽은 순간으로 줄에 붙인다(없으면 자리만)
  3. 관측(observe)          줄마다 끝 직전 정착 프레임 몇 장의 평균에서 배경 · 인물 · 표정 · 자리
  4. 역해석(inverse)        앞 줄과의 차이 → 정본 커맨드
  5. 출력(emit)
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from .catalog import Catalog
from .dialogue import BoxLayout
from .emit import staging_document, write_json, write_yarn
from .gte import LineText, attach, read_xlsx
from .inverse import build_script
from .lines import LineSpan, detect_lines, track_ink
from .observe import ObservedStage, Observer
from .oracle import Oracle

SETTLE_FRAMES = 3          # 정착 프레임 평균 장수
SETTLE_LEAD = 0.05         # 다음 줄 머리보다 이만큼 앞에서 끝낸다(초)


def settled_times(span: LineSpan, fps: float, frames: int = SETTLE_FRAMES) -> List[float]:
    """줄 끝 직전 프레임들 — 다 찍힌 뒤 · 끝 직전. 줄이 너무 짧으면 끝 직전 한 장."""
    end = span.end - SETTLE_LEAD
    start = max(span.head, span.typed)
    ts = [end - k / fps for k in range(frames)]
    ts = [t for t in ts if t >= start] or [max(span.head, end)]
    return sorted(ts)


def grab_frames(video: str, wanted: Sequence[Sequence[float]]) -> List[np.ndarray]:
    """줄마다 원하는 시각들의 프레임을 한 번 훑어 모아 평균한다."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"영상을 열지 못했습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    flat = sorted((t, i) for i, ts in enumerate(wanted) for t in ts)
    sums: List[Optional[np.ndarray]] = [None] * len(wanted)
    counts = [0] * len(wanted)
    k = 0
    index = 0
    last = None
    while k < len(flat):
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and index > 0:
            t = index / fps
        index += 1
        last = frame
        while k < len(flat) and flat[k][0] <= t + 0.5 / fps:
            i = flat[k][1]
            sums[i] = frame.astype(np.float32) if sums[i] is None else sums[i] + frame
            counts[i] += 1
            k += 1
    cap.release()
    out = []
    for s, c in zip(sums, counts):
        if s is None:
            out.append(last.astype(np.float32) if last is not None else None)
        else:
            out.append(s / c)
    return out


def extract(video: str, oracle: Oracle, catalog: Catalog, tuning_dir: str, out_dir: str,
            text_xlsx: Optional[str] = None, node: str = "Extracted", preset: str = "bottom",
            log: Callable[[str], None] = print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    layout = BoxLayout.from_tuning(tuning_dir, int(oracle.width), int(oracle.height), preset)

    track = track_ink(video, layout)
    spans = detect_lines(track)
    log(f"줄 {len(spans)}개 (지움 {sum(s.kind == 'clear' for s in spans)} · 등장 {sum(s.kind == 'onset' for s in spans)})")

    if text_xlsx:
        rows = read_xlsx(text_xlsx)
        texts = attach([s.head for s in spans], [s.end for s in spans], rows)
        log(f"대사 엑셀 {len(rows)}줄 → 붙은 줄 {sum(bool(t.rows) for t in texts)}/{len(spans)}")
    else:
        texts = [LineText(review=["대사 글자 없음(게임텍스트추출기 엑셀을 --text로 주면 붙는다)"]) for _ in spans]

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    wanted = [settled_times(s, fps) for s in spans]
    frames = grab_frames(video, wanted)

    observer = Observer(oracle, catalog, tuning_dir, layout.stage_bottom())
    stages: List[ObservedStage] = []
    previous = None
    for i, (span, frame) in enumerate(zip(spans, frames)):
        st = observer.observe(frame, float(np.mean(wanted[i])), previous)
        stages.append(st)
        previous = st
        cast = ", ".join(f"{s.character}/{s.variant}/{s.emotion}@{s.place}-{s.depth}" for s in st.slots)
        log(f"  {i:3d} {span.head:7.2f}s  배경 {st.background or '없음'} · {cast or '인물 없음'}")

    script = build_script(stages)
    lines = []
    for i, (span, text, st, cmds, notes) in enumerate(zip(spans, texts, stages, script.lines, script.notes)):
        lines.append({
            "index": i,
            "head": round(span.head, 4), "typed": round(span.typed, 4), "end": round(span.end, 4),
            "boundary": span.kind,
            "speaker": text.speaker, "text": text.text, "textRows": text.rows, "textNotes": text.notes,
            "observed": st.to_json(),
            "commands": cmds,
            "review": text.review + notes,
        })
    settings = {"boxPreset": preset, "catalog": catalog.source, "analysisScale": observer.scale,
                "oracleSize": [oracle.width, oracle.height]}
    doc = staging_document(video, node, script, lines, settings, text_xlsx)
    json_path = os.path.abspath(os.path.join(out_dir, f"{stem}.staging.json"))
    yarn_path = write_yarn(os.path.abspath(os.path.join(out_dir, f"{stem}.yarn")), node, script, lines, video)
    doc["files"] = {"staging": json_path, "yarn": yarn_path}
    write_json(json_path, doc)
    log(f"→ {json_path}\n→ {yarn_path}")
    return doc
