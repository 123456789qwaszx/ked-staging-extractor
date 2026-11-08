"""시간과 이징(P2) — 줄 안에서 커맨드마다 시작 · 길이 · 이징을 잰다.

관측(P1)이 정한 커맨드를 그대로 두고, 시간을 가진 커맨드를 '흐름(channel)'으로 묶는다.
  - 움직임(move): 한 슬롯의 place · size · move_by · slide … — 보이는 채 자리가 바뀔 때만 잰다
  - 투명도(alpha): fade_in · fade_out · bg_fade_in · bg_fade_out

움직임은 오라클로 '진행도 u'를 화면으로 바꿔 놓고(그 흐름만 Linear · 1초로, 나머지는 즉시로 바꾼 줄의 시간 축)
영상 프레임마다 가장 닮은 u를 찾는다. 투명도는 그 슬롯이 다 보일 때 · 안 보일 때 두 화면 사이의 섞임 비율을
최소 제곱으로 푼다. 그다음 u(t)에 (시작, 길이, 이징)을 맞춘다 — 이징은 런타임 DOTween 표(ease-golden.json)
35종, 페이드는 런타임 FadeCommandCharR의 OutCubic 고정. 시작이 늦은 흐름은 앞에 pause를 끼운다(런타임에서
줄 안의 커맨드를 늦추는 것은 전경 대기뿐이다).

보간 · 투영 수식은 여기 없다 — 화면은 오라클 timeline이, 이징 값은 golden 표가 낸다.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .catalog import Catalog
from .ease import EaseTable
from .observe import Painter, robust_error
from .oracle import Oracle, Snapshot

FPS = 24.0
MOVE = {"place_left", "place_center", "place_right", "place", "size", "move_by", "slide_in", "slide_out",
        "left", "right", "up", "down", "scale_by", "rotate_by", "char_scale_to", "char_rotate_to"}
ALPHA = {"fade_in", "fade_out", "bg_fade_in", "bg_fade_out"}
SHOT = {"shot_to", "shot_zoom", "shot_track", "shot_reset", "shot_focus_to"}
SHOT_TARGET = "__shot__"
DEFAULT_EASE = "OutCubic"
U_STEPS = 24
MAX_START_FR = 48
MAX_DURATION_FR = 120
START_SLACK_FR = 1.5      # 줄 머리를 영상 프레임으로 잡아 생기는 늦음(1프레임 안)을 시작이 음수로 흡수한다


@dataclass
class Channel:
    target: str
    kind: str                     # "move" | "alpha"
    indices: List[int]            # 줄 커맨드 안의 자리
    start: float = 0.0            # 초(줄 머리부터)
    duration: float = 0.0
    ease: str = DEFAULT_EASE
    rmse: float = float("nan")
    samples: int = 0
    note: str = ""
    series: Optional[tuple] = None     # (t, v, w) — 진단용(staging.json에는 싣지 않는다)

    def to_json(self) -> dict:
        return {"target": self.target, "kind": self.kind, "commands": self.indices,
                "startFrames": int(round(self.start * FPS)), "durationFrames": int(round(self.duration * FPS)),
                "ease": self.ease, "rmse": None if np.isnan(self.rmse) else round(self.rmse, 4),
                "samples": self.samples, "note": self.note}


# ── 맞추기 ───────────────────────────────────────────────────

AMPLITUDE = (0.97, 1.03)    # 진행도 끝이 1에서 조금 벗어나도(관측 자리 반올림 · 압축) 모양으로 맞춘다


def _sse(e: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """후보 곡선들 e(D, N)과 표본 v(N)의 가중 제곱 오차 — 곡선마다 진폭 A(0.9~1.1)를 닫힌 식으로 맞춘 뒤."""
    num = (e * v[None, :] * w[None, :]).sum(1)
    den = (e * e * w[None, :]).sum(1)
    a = np.clip(np.where(den > 1e-12, num / np.maximum(den, 1e-12), 1.0), *AMPLITUDE)
    return ((a[:, None] * e - v[None, :]) ** 2 * w[None, :]).sum(1)


def fit_progress(t: np.ndarray, v: np.ndarray, w: np.ndarray, table: EaseTable,
                 eases: Optional[Sequence[str]] = None, max_start_fr: int = MAX_START_FR,
                 max_duration_fr: int = MAX_DURATION_FR, prefer: str = DEFAULT_EASE
                 ) -> Tuple[float, float, str, float]:
    """진행도 표본 v(t) ≈ ease((t − 시작) / 길이)에 (시작, 길이, 이징)을 맞춘다(24fps 격자). → (시작 s, 길이 s, 이징, RMSE)"""
    t = np.asarray(t, np.float64)
    v = np.asarray(v, np.float64)
    w = np.asarray(w, np.float64)
    if w.sum() <= 0:
        return 0.0, 0.0, prefer, float("nan")
    names = list(eases) if eases else list(table.names)
    durations = np.arange(1, max_duration_fr + 1, dtype=np.float64) / FPS
    xs = np.linspace(0.0, 1.0, 257)
    best = (np.inf, 0.0, 0.0, prefer)
    by_ease: Dict[str, float] = {}
    # 시작 격자: 줄 머리 근처는 0.1프레임(머리를 영상 프레임으로 잡아 생기는 어긋남을 흡수), 그 뒤는 반 프레임.
    starts = np.concatenate([np.arange(-START_SLACK_FR, 3.0, 0.1),
                             np.arange(3.0, max_start_fr + 0.25, 0.5)]) / FPS
    for s in starts:
        p = np.clip((t[None, :] - s) / durations[:, None], 0.0, 1.0)           # (D, N)
        for name in names:
            curve = table.samples[name]
            e = np.interp(p, xs, curve)
            sse = _sse(e, v, w)
            i = int(np.argmin(sse))
            if sse[i] < by_ease.get(name, np.inf):
                by_ease[name] = float(sse[i])
            if sse[i] < best[0]:
                best = (float(sse[i]), float(s), float(durations[i]), name)
    # 거의 같으면 생략값 이징을 고른다(같은 화면이면 대본을 짧게).
    if prefer in by_ease and by_ease[prefer] <= best[0] * 1.02 + 1e-9 and best[3] != prefer:
        # 생략값 이징으로 다시 맞춘 시작 · 길이
        s_best, d_best, sse_best = 0.0, 0.0, np.inf
        for s in starts:
            p = np.clip((t[None, :] - s) / durations[:, None], 0.0, 1.0)
            e = np.interp(p, xs, table.samples[prefer])
            sse = _sse(e, v, w)
            i = int(np.argmin(sse))
            if sse[i] < sse_best:
                s_best, d_best, sse_best = float(s), float(durations[i]), float(sse[i])
        best = (sse_best, s_best, d_best, prefer)
    rmse = float(np.sqrt(best[0] / w.sum()))
    return max(0.0, best[1]), best[2], best[3], rmse


# ── 화면 ─────────────────────────────────────────────────────

def compose(snapshot: Snapshot, catalog: Catalog, painter: Painter, override: Optional[Dict[str, float]] = None,
            size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """스냅샷 → 분석 배율 합성 화면과 슬롯별 덮임(α). override: 슬롯 키 → 강제 투명도."""
    w, h = size or painter.size
    canvas = np.zeros((h, w, 3), np.float32)
    covers: Dict[str, np.ndarray] = {}
    for slot in snapshot.slots:
        alpha_value = float(slot.get("alpha", 0.0))
        if override and slot["key"] in override:
            alpha_value = override[slot["key"]]
        if not slot.get("quad") or alpha_value <= 0.003:
            continue
        if slot.get("background"):
            entry = catalog.background(slot.get("sprite") or "")
        else:
            entry = catalog.portrait(slot.get("character") or "", slot.get("variant") or "a", slot.get("emotion") or "")
        if entry is None:
            continue
        color, alpha = painter.layer(entry, np.asarray(slot["quad"], np.float64), bool(slot.get("toolMirror")))
        a = alpha * alpha_value
        canvas = color * a[:, :, None] + canvas * (1 - a[:, :, None])
        covers[slot["key"]] = alpha
    return canvas, covers


# ── 한 줄 ────────────────────────────────────────────────────

def channels_of(commands: Sequence[dict], visible_before: set, visible_after: set,
                shot_changed: bool = True) -> List[Channel]:
    groups: Dict[Tuple[str, str], Channel] = {}
    for j, c in enumerate(commands):
        name = c["name"]
        if name in SHOT:
            if not shot_changed:
                continue
            groups.setdefault((SHOT_TARGET, "move"), Channel(SHOT_TARGET, "move", [])).indices.append(j)
            continue
        if not c.get("args"):
            continue
        target = c["args"][0]
        if name in ALPHA:
            kind = "alpha"
        elif name in MOVE:
            # 보이는 채 자리가 바뀔 때만 잰다 — 숨은 채 놓는 것은 시간이 보이지 않는다.
            if target not in visible_before or target not in visible_after:
                continue
            kind = "move"
        else:
            continue
        groups.setdefault((target, kind), Channel(target, kind, [])).indices.append(j)
    return list(groups.values())


def _set_arg(cmd: dict, index: Optional[int], value: str) -> None:
    if index is None:
        return
    args = list(cmd.get("args", []))
    while len(args) <= index:
        args.append("")
    args[index] = value
    while args and args[-1] == "":
        args.pop()
    cmd["args"] = args


class TimingMeter:
    def __init__(self, oracle: Oracle, catalog: Catalog, painter: Painter, mask: np.ndarray, table: EaseTable,
                 effects=None, px_scale: float = 1.0):
        self.oracle = oracle
        self.catalog = catalog
        self.painter = painter
        self.mask = mask
        self.table = table
        self.effects = effects          # EffectPresets — 캐릭터 비주얼이 바뀌는 슬롯을 그릴 때(근사, P3)
        self.px_scale = px_scale
        self._recolor: Dict[str, tuple] = {}

    def _probe(self, history: Sequence[Sequence[dict]], line: Sequence[dict], live: Sequence[int],
               times: Sequence[float]) -> List[Snapshot]:
        """line의 live 커맨드만 Linear · 1초로, 나머지 시간 커맨드는 즉시로, 대기는 빼고 times에서 뜬다."""
        probe = []
        for j, c in enumerate(line):
            if c["name"] == "pause" or c["name"].endswith("fr") and c["name"][:-2].isdigit():
                continue
            c2 = copy.deepcopy(c)
            di = self.oracle.param_index(c2["name"], "duration")
            if di is not None:
                if j in live:
                    _set_arg(c2, di, "24fr")
                    _set_arg(c2, self.oracle.param_index(c2["name"], "ease"), "Linear")
                else:
                    _set_arg(c2, di, "0fr")
            probe.append(c2)
        return self.oracle.timeline(list(history) + [probe], list(times))["snapshots"]

    def _layers(self, snapshot: Snapshot) -> List[Tuple[str, float, np.ndarray, np.ndarray]]:
        """스냅샷의 슬롯마다 (키, 투명도, 색, 덮임) — 그리기 순서. 투명도만 바꿔 다시 합성할 때 쓴다."""
        out = []
        for slot in snapshot.slots:
            if not slot.get("quad"):
                continue
            if slot.get("background"):
                entry = self.catalog.background(slot.get("sprite") or "")
            else:
                entry = self.catalog.portrait(slot.get("character") or "", slot.get("variant") or "a",
                                              slot.get("emotion") or "")
            if entry is None:
                continue
            color, alpha = self.painter.layer(entry, np.asarray(slot["quad"], np.float64), bool(slot.get("toolMirror")))
            out.append((slot["key"], float(slot.get("alpha", 0.0)), color, alpha))
        return out

    @staticmethod
    def _mix(layers, alphas: Dict[str, float], shape) -> np.ndarray:
        canvas = np.zeros(shape, np.float32)
        for key, a0, color, alpha in layers:
            a = alpha * alphas.get(key, a0)
            canvas = color * a[:, :, None] + canvas * (1 - a[:, :, None])
        return canvas

    def measure_line(self, history: Sequence[Sequence[dict]], line: Sequence[dict], channels: Sequence[Channel],
                     frames: Sequence[Tuple[float, np.ndarray]], recolor: Optional[Dict[str, tuple]] = None) -> None:
        """한 줄의 흐름을 모두 잰다. 움직임은 하나씩, 투명도는 함께(동시에 사라지는 페이드가 서로 섞인다).
        recolor: 슬롯 → (앞 비주얼, 뒤 비주얼, 바뀌는 초) — 캐릭터 비주얼이 줄 머리에서 바뀌는 슬롯(근사로 섞어 그린다)."""
        self._recolor = recolor or {}
        if not frames:
            for ch in channels:
                ch.note = "프레임 없음"
            return
        for ch in channels:
            if ch.kind == "move":
                others = {o.target for o in channels if o is not ch and o.target != ch.target}
                self._measure_move(history, line, ch, frames, others)
        alphas = [ch for ch in channels if ch.kind == "alpha"]
        if alphas:
            moving = [j for ch in channels if ch.kind == "move" for j in ch.indices]
            self._measure_alpha(history, line, alphas, frames,
                                {ch.target for ch in channels if ch.kind == "move"}, moving)

    def _recolored(self, layers):
        """줄 머리에서 비주얼이 바뀌는 슬롯은 (앞 · 뒤) 두 모양을 미리 그려 둔다 → t초의 layers를 내는 함수."""
        from .effects import visual_layer
        if not self._recolor or self.effects is None:
            return lambda t: layers
        pairs = {}
        for key, a0, color, alpha in layers:
            spec = self._recolor.get(key)
            if not spec:
                continue
            looks = []
            for v in spec[:2]:
                if v and v[0] in self.effects.visual:
                    looks.append(visual_layer(color, alpha, self.effects.visual[v[0]], v[1], self.px_scale))
                else:
                    looks.append((color, alpha))
            pairs[key] = (looks[0], looks[1], max(1e-6, float(spec[2])))
        if not pairs:
            return lambda t: layers

        def at(t: float):
            out = []
            for key, a0, color, alpha in layers:
                if key in pairs:
                    (c0, al0), (c1, al1), dur = pairs[key]
                    r = float(np.clip(t / dur, 0.0, 1.0))
                    color, alpha = c0 * (1 - r) + c1 * r, al0 * (1 - r) + al1 * r
                out.append((key, a0, color, alpha))
            return out
        return at

    def _measure_move(self, history, line, channel: Channel, frames, others=frozenset()) -> None:
        ts = np.array([t for t, _ in frames])
        us = list(np.linspace(0.0, 1.0, U_STEPS + 1))
        snaps = self._probe(history, line, channel.indices, us)
        # 넘침(OutBack 같은 이징)을 잴 수 있게 그 슬롯의 사각형을 진행도 밖으로 늘린다 — 오라클 표본에 이차식을 맞춰 연장.
        quads = [next((sl["quad"] for sl in sn.slots if sl["key"] == channel.target and sl.get("quad")), None)
                 for sn in snaps]
        frames_u = list(zip(us, snaps))
        if all(q is not None for q in quads):
            q = np.asarray(quads, np.float64).reshape(len(us), -1)
            coef = np.polyfit(np.asarray(us), q, 2)
            for u in list(np.arange(-0.3, 0.0, 0.05)) + list(np.arange(1.05, 1.41, 0.05)):
                base = copy.deepcopy(snaps[0] if u < 0 else snaps[-1])
                qu = np.polyval(coef, u) if coef.ndim == 1 else np.array([np.polyval(coef[:, i], u) for i in range(q.shape[1])])
                for sl in base.slots:
                    if sl["key"] == channel.target:
                        sl["quad"] = qu.reshape(4, 2).tolist()
                frames_u.append((float(u), base))
            frames_u.sort(key=lambda x: x[0])
        grid = np.array([u for u, _ in frames_u])
        renders, region = [], np.zeros(self.mask.shape, bool)
        for u, sn in frames_u:
            img, covers = compose(sn, self.catalog, self.painter)
            renders.append(img)
            if channel.target in covers and 0.0 <= u <= 1.0:
                region |= covers[channel.target] > 0.5
        if channel.target == SHOT_TARGET:
            region = np.ones(self.mask.shape, bool)   # 카메라는 무대 전체가 함께 움직인다
        region &= self.mask > 0
        # 같은 줄에서 함께 변하는 다른 슬롯(페이드 · 움직임)이 덮는 자리는 뺀다 — 탐침에서는 그 슬롯이 이미 끝 상태다.
        for sn in (snaps[0], snaps[-1]):
            for key, _a, _c, cover in self._layers(sn):
                if key in others:
                    region &= cover < 0.05
        if region.sum() < 100:
            channel.note = "보이는 영역이 너무 작다"
            return
        ys, xs = np.nonzero(region)
        bb = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        reg = region[bb]
        crops = [r[bb] for r in renders]
        vs, ws = [], []
        for _, f in frames:
            fc = f[bb]
            errs = np.array([robust_error(fc, r)[reg].mean() for r in crops])
            i = int(np.argmin(errs))
            u = grid[i]
            if 0 < i < len(grid) - 1:  # 포물선으로 다듬기
                ea, eb, ec = errs[i - 1], errs[i], errs[i + 1]
                den = ea - 2 * eb + ec
                if den > 1e-9:
                    step = 0.5 * (grid[i + 1] - grid[i - 1])
                    u = grid[i] + 0.5 * (ea - ec) / den * step
            spread = float(np.median(errs) - errs[i])
            vs.append(float(u))
            ws.append(float(np.clip(spread / 3.0, 0.0, 1.0)))
        self._fit(channel, ts, np.array(vs), np.array(ws), None)

    def _measure_alpha(self, history, line, channels: Sequence[Channel], frames, movers=frozenset(),
                       moving: Sequence[int] = ()) -> None:
        ts = np.array([t for t, _ in frames])
        snap = self._probe(history, line, [], [0.0])[0]
        layers = self._layers(snap)
        shape = frames[0][1].shape
        keys = [ch.target for ch in channels]
        fading_in = {ch.target: line[ch.indices[-1]]["name"] in ("fade_in", "bg_fade_in") for ch in channels}
        # 같은 줄에서 움직이는 슬롯이 지나는 자리는 뺀다(탐침에서는 이미 끝 자리에 있다).
        busy = np.zeros(self.mask.shape, bool)
        if movers:
            for sn in self._probe(history, line, list(moving), [0.0, 1.0]):
                for key, _a, _c, cover in self._layers(sn):
                    if key in movers:
                        busy |= cover > 0.05
        regions = {}
        for k in keys:
            full = self._mix(layers, {k: 1.0}, shape)
            empty = self._mix(layers, {k: 0.0}, shape)
            regions[k] = (np.abs(full - empty).mean(2) > 12) & (self.mask > 0) & ~busy
        series = {k: [] for k in keys}
        layers_at = self._recolored(layers)
        for t_frame, f in frames:
            layers = layers_at(t_frame)
            est = {k: 0.5 for k in keys}
            for _ in range(4 if len(keys) > 1 else 1):
                for k in keys:
                    if regions[k].sum() < 100:
                        continue
                    full = self._mix(layers, {**est, k: 1.0}, shape)
                    empty = self._mix(layers, {**est, k: 0.0}, shape)
                    d = (full - empty)[regions[k]]
                    num = (f[regions[k]] - empty[regions[k]])
                    strong = np.abs(d) > 24
                    if strong.sum() < 50:
                        continue
                    # 픽셀마다 섞임 비율의 중앙값 — 다른 변화(움직이는 슬롯 · 압축)가 섞인 픽셀에 끌려가지 않는다.
                    est[k] = float(np.clip(np.median(num[strong] / d[strong]), 0.0, 1.0))
            for k in keys:
                series[k].append(est[k] if fading_in[k] else 1.0 - est[k])
        for ch in channels:
            if regions[ch.target].sum() < 100:
                ch.note = "보이는 영역이 너무 작다"
                continue
            self._fit(ch, ts, np.array(series[ch.target]), np.ones(len(ts)), [DEFAULT_EASE])

    def _fit(self, channel: Channel, ts, v, w, eases) -> None:
        channel.series = (list(map(float, ts)), list(map(float, v)), list(map(float, w)))
        s, dur, ease, rmse = fit_progress(ts, v, w, self.table, eases)
        channel.start, channel.duration, channel.ease, channel.rmse, channel.samples = s, dur, ease, rmse, len(v)
        if rmse > 0.12:
            channel.note = f"맞춤 오차가 큽니다(RMSE {rmse:.2f})"


def apply_timing(line: Sequence[dict], channels: Sequence[Channel], oracle: Oracle,
                 min_start_fr: int = 2) -> List[dict]:
    """잰 시간 · 이징을 커맨드 인자에 써 넣고, 늦게 시작하는 흐름 앞에 pause를 끼운다."""
    out = [copy.deepcopy(c) for c in line]
    start_of: Dict[int, int] = {}
    for ch in channels:
        if np.isnan(ch.rmse):
            continue
        dur = f"{int(round(ch.duration * FPS))}fr"
        for j in ch.indices:
            _set_arg(out[j], oracle.param_index(out[j]["name"], "duration"), dur)
            ei = oracle.param_index(out[j]["name"], "ease")
            if ei is not None:
                _set_arg(out[j], ei, "" if ch.ease == DEFAULT_EASE else ch.ease)
            sf = int(round(ch.start * FPS))
            if sf >= min_start_fr:
                start_of[j] = sf
    if not start_of:
        return out
    # 시작 순서대로 다시 늘어놓는다 — 즉시 커맨드와 시작이 0인 흐름이 먼저, 그다음 대기 · 늦은 흐름.
    order = sorted(range(len(out)), key=lambda j: (start_of.get(j, 0), j))
    result: List[dict] = []
    clock = 0
    for j in order:
        sf = start_of.get(j, 0)
        if sf > clock:
            result.append({"name": "pause", "args": [f"{sf - clock}fr"]})
            clock = sf
        result.append(out[j])
    return result
