"""합성 녹화 — 실제 녹화가 없을 때 시험장 도구가 제대로 도는지 확인한다.

예측 화면을 그대로 이어 붙이므로 오라클 자체의 정확도는 재지 못한다(그건 실제 런타임 녹화의 몫이다).
재는 것: 정렬이 상태 순서를 되찾는가, 일부러 옮긴 슬롯의 어긋남(px)을 되읽는가.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .catalog import Catalog
from .oracle import Snapshot
from .render import render


def shifted(snapshot: Snapshot, key: str, dx: float, dy: float) -> Snapshot:
    snap = copy.deepcopy(snapshot)
    for s in snap.slots:
        if s["key"] == key and s.get("quad"):
            s["quad"] = [[p[0] + dx, p[1] + dy] for p in s["quad"]]
    return snap


def write_recording(path: str, snapshots: Sequence[Snapshot], catalog: Catalog, width: float, height: float,
                    hold: float = 1.2, fade: float = 0.3, fps: float = 30.0, noise: float = 2.0,
                    lead_in: float = 0.5, box_ratio: float = 0.34, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    w, h = int(width), int(height)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"영상을 쓰지 못했습니다: {path}")
    frames = [render(s, catalog, width, height).astype(np.float32) for s in snapshots]
    black = np.zeros_like(frames[0])
    box_top = int(round(h * (1.0 - box_ratio)))

    def emit(img: np.ndarray, line_seed: int) -> None:
        out = img.copy()
        out[box_top:, 40:w - 40] = out[box_top:, 40:w - 40] * 0.25 + 20  # 대사창 흉내(대조에서 가려지는 띠)
        r = np.random.default_rng(line_seed)
        for _ in range(12):  # 글자 흉내
            x, y = int(r.integers(80, w - 300)), int(r.integers(box_top + 40, h - 40))
            cv2.rectangle(out, (x, y), (x + int(r.integers(20, 200)), y + 24), (230, 230, 230), -1)
        out += rng.normal(0, noise, out.shape).astype(np.float32)
        writer.write(np.clip(out, 0, 255).astype(np.uint8))

    for _ in range(int(lead_in * fps)):
        emit(black, 0)
    previous = black
    for i, frame in enumerate(frames):
        n_fade, n_hold = int(fade * fps), int(hold * fps)
        for k in range(n_fade):
            t = (k + 1) / (n_fade + 1)
            emit(previous * (1 - t) + frame * t, i + 1)
        for _ in range(n_hold):
            emit(frame, i + 1)
        previous = frame
    writer.release()
