"""합성 녹화 — 실제 녹화가 없을 때 시험장 도구가 제대로 도는지 확인한다.

예측 화면을 그대로 이어 붙이므로 오라클 자체의 정확도는 재지 못한다(그건 실제 런타임 녹화의 몫이다).
재는 것: 정렬이 상태 순서를 되찾는가, 일부러 옮긴 슬롯의 어긋남(px)을 되읽는가.
"""
from __future__ import annotations

import copy
import os
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


# ── P1 시험장: 대사창 · 타자기 · 전이가 있는 합성 녹화 ─────────────────

SCENARIO_SCHEMA = "ked-staging-scenario/1"


def transition_seconds(commands: Sequence[dict], cap: float = 1.0) -> float:
    """줄 커맨드 중 가장 긴 시간 인자(초). 무대가 정착하는 데 걸리는 시간의 근사 — 합성 녹화의 전이 길이다."""
    from .tokens import parse_duration
    longest = 0.0
    for cmd in commands:
        for arg in cmd.get("args", []):
            a = str(arg).strip().lower()
            if a.endswith("fr") or (a.endswith("s") and a[:-1].replace(".", "", 1).isdigit()):
                try:
                    longest = max(longest, parse_duration(a))
                except ValueError:
                    pass
    return min(cap, longest)


def _out_cubic(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1.0 - (1.0 - t) ** 3


def write_scenario(path: str, lines: Sequence[dict], catalog: Catalog, oracle_size: Tuple[float, float],
                   layout, size: Tuple[int, int] = (1920, 1080), fps: float = 30.0, cps: float = 24.0,
                   hold: float = 0.9, hold_jitter: float = 0.4, lead_in: float = 0.6, tail: float = 0.8,
                   noise: float = 2.0, seed: int = 7, font_path: Optional[str] = None,
                   route: Sequence[str] = (), oracle=None, effect_presets=None) -> dict:
    """오라클이 접은 줄들(route_lines)로 대사창 · 타자기 · 무대 전이가 있는 녹화를 쓰고 정답을 돌려준다.

    줄 머리(head)에 앞 줄 본문이 지워지고 이름이 바뀌며 무대가 움직이기 시작한다.
    - oracle을 주면 무대는 오라클 시간 축(timeline)대로 움직인다 — 커맨드마다 제 시작 · 길이 · 이징, 대기(pause),
      페이드 투명도까지 런타임 규칙 그대로다(P2). 정답에 줄마다 그 시간표(schedule)가 붙는다.
    - 없으면 커맨드의 가장 긴 시간 동안 앞 화면에서 새 화면으로 OutCubic 크로스페이드한다(P1).
    본문은 cps 글자/초로 찍히고, 다 찍히고 무대도 정착한 뒤 hold(+흔들림)만큼 머문다(다음 줄 머리의 트윈 가속을 피한다).
    """
    from .dialogue import BoxPainter, split_speaker
    from .effects import EffectState, render_with_effects
    from .render import render_layers
    rng = np.random.default_rng(seed)
    w, h = int(size[0]), int(size[1])
    box = layout.scaled(w, h)
    painter = BoxPainter(box, font_path)
    scale = w / float(oracle_size[0])
    effects = EffectState()
    fx_presets = effect_presets
    fx_rng = np.random.default_rng(seed + 1)

    def draw(snapshot) -> np.ndarray:
        return render(snapshot, catalog, oracle_size[0], oracle_size[1], scale=scale)

    def draw_fx(snapshot, t: float) -> np.ndarray:
        """효과(근사)가 걸려 있으면 층마다 그려 얹는다."""
        if fx_presets is None or not effects.active(t):
            return draw(snapshot).astype(np.float32)
        layer_of = {sl["key"]: sl.get("layer", "mid") for sl in snapshot.slots}
        layers = [(k, layer_of.get(k, "mid"), c, a)
                  for k, c, a in render_layers(snapshot, catalog, oracle_size[0], oracle_size[1], scale)]
        return render_with_effects(layers, effects, fx_presets, t, (w, h), scale / 0.25, fx_rng)

    stages = [draw(l["snapshot"]) for l in lines]          # uint8 — 긴 경로에서도 메모리를 아낀다
    black = np.zeros((h, w, 3), np.uint8)
    commands = [l.get("commands", []) for l in lines]

    # 시간표
    plan = []
    t = lead_in
    for i, line in enumerate(lines):
        speaker, body = split_speaker(line.get("text") or "")
        same = i > 0 and np.array_equal(stages[i], stages[i - 1])
        schedule = None
        if oracle is not None:
            tl = oracle.timeline(commands[:i + 1], [])
            schedule = tl["schedule"]
            trans = float(tl["end"])   # 정착 화면이 같아도 줄 안에서 지나가는 움직임(제자리 모션)이 있을 수 있다
        else:
            trans = 0.0 if same else transition_seconds(line.get("commands", []))
        typed = len(body) / cps
        end = t + max(trans, typed) + hold + float(rng.uniform(0, hold_jitter))
        entry = {"index": i, "node": line.get("node"), "lineId": line.get("lineId"), "text": line.get("text"),
                 "speaker": speaker, "body": body, "commands": line.get("commands", []),
                 "head": t, "settle": t + trans, "typed": t + typed, "end": end, "transition": trans}
        if schedule is not None:
            entry["schedule"] = schedule
        plan.append(entry)
        t = end
    total = t + tail

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"영상을 쓰지 못했습니다: {path}")
    bank = [rng.normal(0, noise, (h, w, 3)).astype(np.float32) for _ in range(6)] if noise > 0 else None
    n_frames = int(np.ceil(total * fps))
    k = -1
    moving: Dict[int, np.ndarray] = {}      # 지금 줄의 움직이는 프레임(프레임 번호 → 그림)
    for n in range(n_frames):
        t = n / fps
        while k + 1 < len(plan) and t >= plan[k + 1]["head"] - 1e-9:
            k += 1
            moving = {}
            p = plan[k]
            effects.flashes = []
            effects.apply_commands(p["commands"], p["head"])
            if oracle is not None and p["transition"] > 0:
                ns = [m for m in range(n, n_frames) if m / fps < p["settle"]]
                tl = oracle.timeline(commands[:k + 1], [m / fps - p["head"] for m in ns])
                moving = {m: snap for m, snap in zip(ns, tl["snapshots"])}
        boxed = not (k < 0 or t >= plan[-1]["end"])
        if k < 0:
            frame = black
        elif not boxed:
            frame = stages[-1]
        else:
            p = plan[k]
            prev = stages[k - 1] if k > 0 else black
            if n in moving:
                frame = draw_fx(moving[n], t)
            elif oracle is None and p["transition"] > 0 and t < p["settle"]:
                e = _out_cubic((t - p["head"]) / p["transition"])
                frame = prev.astype(np.float32) * (1 - e) + stages[k].astype(np.float32) * e
            elif fx_presets is not None and effects.active(t):
                frame = draw_fx(lines[k]["snapshot"], t)
            else:
                frame = stages[k]
        frame = frame.astype(np.float32)
        if bank is not None:
            frame = frame + bank[n % len(bank)]
        img = np.clip(frame + 0.5, 0, 255).astype(np.uint8)
        if boxed:
            p = plan[k]
            shown = int(np.floor((t - p["head"]) * cps + 1e-6))
            img = painter.paint(img, p["speaker"], p["body"], shown)
        writer.write(img)
    writer.release()
    return {"schema": SCENARIO_SCHEMA, "video": os.path.abspath(path), "fps": fps, "size": [w, h],
            "cps": cps, "duration": total, "route": [os.path.abspath(r) for r in route],
            "catalog": catalog.source, "boxPreset": layout.preset, "timed": oracle is not None, "lines": plan}
