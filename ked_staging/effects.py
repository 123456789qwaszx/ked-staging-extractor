"""화면 효과(P3) — 캐릭터 비주얼(char_visual_*) · 비네트 · 노이즈 · 플래시 · 블러.

⚠ 근사다. 런타임은 이 효과들을 셰이더로 그리고, 코어 · 오라클은 접지 않는다(Unhandled). 여기서는 런타임 프리셋
(ExportedTuning/presets의 visual-focus · screen-vignette · screen-flash · screen-noise)의 수치로 '비슷한 모양'을
정해 두고, 그 모양으로 합성 녹화를 그리고(시험장) 같은 모양에 맞춰 효과를 알아본다. 실제 런타임 화면과는 세기 ·
모양이 다를 수 있다 — 추출 결과에는 언제나 `// 확인: 효과는 근사`가 붙는다.

근사 모양
  char_visual  색 = 색 × (1 − dim·s) + 색 × 틴트 × dim·s, 흐림 = blur·s × 3px(분석 배율 1/4 기준). 림(rim)은 그리지 않는다.
  vignette     화면 중심 거리 d(가로는 aspect로 나눔)에서 m = clip((d − radius) / softness) → 색 쪽으로 amount·s·m 섞기
  flash        색 쪽으로 amount·s·봉우리(attack 직선 · hold · release 직선) 섞기 — 줄 머리에서
  noise        화소마다 N(0, 40·amount·s) × 색
  blur         그 층(layer)의 슬롯을 amount × 6px(1/4 배율)로 흐림
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .tokens import parse_duration

VISUAL_PRESETS_SKIP = {"clear", "test"}
STRENGTHS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def _load(path: str) -> Dict[str, dict]:
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except OSError:
        return {}
    return {e["key"]: e for e in data.get("MonoBehaviour", data).get("entries", [])}


def _bgr(c: dict) -> np.ndarray:
    return np.array([c.get("b", 1.0), c.get("g", 1.0), c.get("r", 1.0)], np.float32)


@dataclass
class EffectPresets:
    visual: Dict[str, dict]
    vignette: Dict[str, dict]
    flash: Dict[str, dict]
    noise: Dict[str, dict]

    @classmethod
    def load(cls, tuning_dir: str) -> "EffectPresets":
        p = os.path.join(tuning_dir, "presets")
        return cls(_load(os.path.join(p, "visual-focus.json")), _load(os.path.join(p, "screen-vignette.json")),
                   _load(os.path.join(p, "screen-flash.json")), _load(os.path.join(p, "screen-noise.json")))


# ── 근사 모양 ────────────────────────────────────────────────

def visual_layer(color: np.ndarray, alpha: np.ndarray, entry: dict, strength: float, px_scale: float = 1.0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    dim = float(np.clip(entry.get("dim", 0.0) * strength, 0.0, 1.0))
    tint = _bgr(entry.get("dimTintColor", {}))
    out = color * (1 - dim) + color * tint[None, None, :] * dim
    blur = entry.get("blur", 0.0) * strength * 3.0 * px_scale
    if blur > 0.3:
        k = int(2 * round(blur * 2) + 1)
        out = cv2.GaussianBlur(out, (k, k), blur)
        alpha = cv2.GaussianBlur(alpha, (k, k), blur)
    return out, alpha


def vignette_mask(w: int, h: int, entry: dict) -> np.ndarray:
    aspect = float(entry.get("aspect", 1.0)) or 1.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xs / max(1, w - 1) * 2 - 1) / aspect
    dy = ys / max(1, h - 1) * 2 - 1
    d = np.sqrt(dx * dx + dy * dy)
    soft = max(1e-3, float(entry.get("softness", 0.1)))
    return np.clip((d - float(entry.get("radius", 1.0))) / soft, 0.0, 1.0)


def _vignette_weight(img: np.ndarray, entry: dict, strength: float, mask: Optional[np.ndarray]) -> np.ndarray:
    base = vignette_mask(img.shape[1], img.shape[0], entry) if mask is None else mask
    return np.clip(base * float(entry.get("amount", 0.0)) * strength, 0, 1)[:, :, None]


def apply_vignette(img: np.ndarray, entry: dict, strength: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
    m = _vignette_weight(img, entry, strength, mask)
    color = _bgr(entry.get("color", {})) * 255
    return img * (1 - m) + color[None, None, :] * m


UNDO_FLOOR = 0.2   # 비네트를 걷어 낼 때 (1 − m)의 하한 — 거의 다 덮인 가장자리는 잡음만 키운다


def undo_vignette(img: np.ndarray, entry: dict, strength: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """apply_vignette의 역 — 관측 · 시간 재기 전에 화면에서 비네트를 걷어 낸다."""
    m = _vignette_weight(img, entry, strength, mask)
    color = _bgr(entry.get("color", {})) * 255
    return (img - color[None, None, :] * m) / np.maximum(1 - m, UNDO_FLOOR)


class VignetteMasks:
    """프리셋 · 화면 크기마다 비네트 모양(m)을 한 번만 계산한다."""

    def __init__(self, presets: "EffectPresets"):
        self.p = presets
        self._cache: Dict[Tuple[str, int, int], np.ndarray] = {}

    def get(self, key: str, w: int, h: int) -> np.ndarray:
        k = (key, w, h)
        if k not in self._cache:
            self._cache[k] = vignette_mask(w, h, self.p.vignette[key])
        return self._cache[k]

    def apply(self, img: np.ndarray, hyp: Optional[Tuple[str, float]]) -> np.ndarray:
        if hyp is None or hyp[0] not in self.p.vignette:
            return img
        return apply_vignette(img, self.p.vignette[hyp[0]], hyp[1], self.get(hyp[0], img.shape[1], img.shape[0]))

    def undo(self, img: np.ndarray, hyp: Optional[Tuple[str, float]]) -> np.ndarray:
        if hyp is None or hyp[0] not in self.p.vignette:
            return img
        return undo_vignette(img, self.p.vignette[hyp[0]], hyp[1], self.get(hyp[0], img.shape[1], img.shape[0]))

    def hypotheses(self) -> List[Tuple[str, float]]:
        return [(n, s) for n, e in self.p.vignette.items() if n != "clear" and e.get("amount", 0) > 0
                for s in STRENGTHS]


def visual_hypotheses(presets: "EffectPresets") -> List[Tuple[str, float]]:
    """캐릭터 비주얼 후보(프리셋 × 세기) — 색이 바뀌는(dim) · 흐려지는(blur) 것만. 림은 근사하지 않는다."""
    return [(n, s) for n, e in presets.visual.items()
            if n not in VISUAL_PRESETS_SKIP and (e.get("dim", 0) > 0 or e.get("blur", 0) > 0) for s in STRENGTHS]


def flash_envelope(entry: dict, t: float) -> float:
    a, h, r = (float(entry.get(k, 0.0)) for k in ("attackDuration", "holdDuration", "releaseDuration"))
    if t < 0:
        return 0.0
    if t < a:
        return t / a if a > 0 else 1.0
    if t < a + h:
        return 1.0
    if t < a + h + r:
        return 1.0 - (t - a - h) / r if r > 0 else 0.0
    return 0.0


def apply_flash(img: np.ndarray, entry: dict, strength: float, t: float) -> np.ndarray:
    k = float(np.clip(entry.get("amount", 0.0) * strength * flash_envelope(entry, t), 0, 1))
    return img * (1 - k) + _bgr(entry.get("color", {}))[None, None, :] * 255 * k


def noise_sigma(entry: dict, strength: float) -> float:
    return 40.0 * float(entry.get("amount", 0.0)) * strength


def apply_noise(img: np.ndarray, entry: dict, strength: float, rng: np.random.Generator) -> np.ndarray:
    s = noise_sigma(entry, strength)
    if s <= 0:
        return img
    n = rng.normal(0, s, img.shape[:2]).astype(np.float32)[:, :, None]
    return img + n * _bgr(entry.get("color", {}))[None, None, :]


# ── 상태(합성 녹화가 줄마다 따라간다) ──────────────────────────

@dataclass
class _Ramp:
    key: str
    target: float
    start_value: float
    start_time: float
    duration: float

    def value(self, t: float) -> float:
        if self.duration <= 0 or t >= self.start_time + self.duration:
            return self.target
        p = max(0.0, (t - self.start_time) / self.duration)
        return self.start_value + (self.target - self.start_value) * p


@dataclass
class EffectState:
    """커맨드를 따라가며 어느 순간 어떤 효과가 얼마나 걸려 있나(근사 — 세기만 직선으로 바뀐다)."""
    visual: Dict[str, _Ramp] = field(default_factory=dict)       # 슬롯 → 프리셋 · 세기
    vignette: Optional[_Ramp] = None
    noise: Optional[_Ramp] = None
    blur: Dict[str, float] = field(default_factory=dict)          # 층 → 양
    flashes: List[Tuple[str, float, float]] = field(default_factory=list)   # (프리셋, 세기, 시작)

    def apply_commands(self, commands: Sequence[dict], head: float) -> None:
        def dur(tok: Optional[str], default: str) -> float:
            try:
                return parse_duration(tok or default)
            except ValueError:
                return parse_duration(default)

        for c in commands:
            n, a = c["name"], list(c.get("args", []))
            if n.startswith("char_visual_") and n != "char_visual_clear" and a:
                key = n[len("char_visual_"):]
                s = float(a[1]) if len(a) > 1 else 1.0
                cur = self.visual.get(a[0])
                start = cur.value(head) if cur and cur.key == key else 0.0
                self.visual[a[0]] = _Ramp(key, s, start, head, dur(a[2] if len(a) > 2 else None, "6fr"))
            elif n == "char_visual_clear" and a:
                cur = self.visual.get(a[0])
                if cur:
                    self.visual[a[0]] = _Ramp(cur.key, 0.0, cur.value(head), head, dur(a[1] if len(a) > 1 else None, "6fr"))
            elif n == "screen_vignette" and a:
                s = float(a[1]) if len(a) > 1 else 1.0
                self.vignette = _Ramp(a[0], s, 0.0 if not self.vignette or self.vignette.key != a[0] else self.vignette.value(head),
                                      head, dur(a[2] if len(a) > 2 else None, "0.35s"))
            elif n == "screen_vignette_clear" and self.vignette:
                self.vignette = _Ramp(self.vignette.key, 0.0, self.vignette.value(head), head, dur(a[0] if a else None, "0.35s"))
            elif n == "screen_noise":
                key = a[0] if a else "default"
                s = float(a[1]) if len(a) > 1 else 1.0
                self.noise = _Ramp(key, s, 0.0 if not self.noise or self.noise.key != key else self.noise.value(head),
                                   head, dur(a[2] if len(a) > 2 else None, "0.35s"))
            elif n == "screen_noise_clear" and self.noise:
                self.noise = _Ramp(self.noise.key, 0.0, self.noise.value(head), head, dur(a[0] if a else None, "0.35s"))
            elif n == "screen_blur" and a:
                self.blur[a[0]] = float(a[1]) if len(a) > 1 else 1.0
            elif n == "screen_blur_reset":
                self.blur = {}
            elif n == "screen_flash" and a:
                self.flashes.append((a[0], float(a[1]) if len(a) > 1 else 1.0, head))

    def active(self, t: float) -> bool:
        return bool(any(r.value(t) > 0 for r in self.visual.values()) or (self.vignette and self.vignette.value(t) > 0)
                    or (self.noise and self.noise.value(t) > 0) or any(v > 0 for v in self.blur.values())
                    or self.flashes)


def render_with_effects(layers: Sequence[Tuple[str, str, np.ndarray, np.ndarray]], state: EffectState,
                        presets: EffectPresets, t: float, size: Tuple[int, int], px_scale: float,
                        rng: np.random.Generator) -> np.ndarray:
    """layers: (슬롯 키, 층, 색, α) 그리기 순서 → 효과를 얹은 화면(BGR float32)."""
    w, h = size
    canvas = np.zeros((h, w, 3), np.float32)
    for key, layer, color, alpha in layers:
        ramp = state.visual.get(key)
        if ramp and ramp.value(t) > 0 and ramp.key in presets.visual:
            color, alpha = visual_layer(color, alpha, presets.visual[ramp.key], ramp.value(t), px_scale)
        amount = state.blur.get(layer, 0.0)
        if amount > 0:
            sigma = amount * 6.0 * px_scale
            k = int(2 * round(sigma * 2) + 1)
            color = cv2.GaussianBlur(color, (k, k), sigma)
            alpha = cv2.GaussianBlur(alpha, (k, k), sigma)
        a = alpha[:, :, None]
        canvas = color * a + canvas * (1 - a)
    if state.vignette and state.vignette.value(t) > 0 and state.vignette.key in presets.vignette:
        canvas = apply_vignette(canvas, presets.vignette[state.vignette.key], state.vignette.value(t))
    if state.noise and state.noise.value(t) > 0 and state.noise.key in presets.noise:
        canvas = apply_noise(canvas, presets.noise[state.noise.key], state.noise.value(t), rng)
    for key, s, start in state.flashes:
        if key in presets.flash:
            canvas = apply_flash(canvas, presets.flash[key], s, t - start)
    return canvas


# ── 알아보기 ─────────────────────────────────────────────────

FLASH_SETTLE = 0.45   # 이 뒤로는 플래시가 다 사그라들었다고 본다(프리셋 중 가장 긴 soft가 0.38초)
FLASH_PEAK_BY = 0.3   # 봉우리는 줄 머리에서 이 안에 있어야 한다
FLASH_MIN_LEVEL = 8.0  # 두 바닥을 넘은 밝기(칸 중앙값의 중앙값) — 이보다 작으면 플래시가 아니다


class EffectDetector:
    """정착 화면과 효과를 입힌 예측을 견줘 남은 효과(노이즈 · 층 블러 · 플래시)를 고른다(같은 근사 모양으로 다시
    그려 오차가 가장 줄어드는 것). 캐릭터 비주얼 · 비네트는 관측(Observer)이 무대와 함께 고른다."""

    def __init__(self, presets: EffectPresets, mask: np.ndarray, px_scale: float):
        self.p = presets
        self.mask = mask > 0
        self.px_scale = px_scale

    @staticmethod
    def _err(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
        return float(np.minimum(np.abs(a - b).mean(2), 60)[m].mean()) if m.any() else 0.0

    def residual_sigma(self, frame: np.ndarray, model: np.ndarray) -> float:
        """잔차(화면 − 예측)의 고주파 세기 — 중앙값 기반(MAD)이라 한 곳의 큰 어긋남(가장자리 · 남은 오차)에 덜 흔들린다."""
        r = (frame - model).mean(2)
        hp = (r - cv2.GaussianBlur(r, (0, 0), 2.0))[self.mask]
        if hp.size == 0:
            return 0.0
        return float(1.4826 * np.median(np.abs(hp - np.median(hp))))

    def noise(self, frame: np.ndarray, model: np.ndarray, floor_sigma: float, upscale: float = 1.0
              ) -> Optional[Tuple[str, float]]:
        """잔차의 고주파 세기가 바닥(효과 없는 줄)보다 크면 노이즈 — 'default' 프리셋 세기로 낸다(색은 못 가린다).
        upscale: 영상 화소 / 분석 화소 — 화소마다 따로인 잡음은 줄여 볼 때 그만큼 약해진다."""
        s = self.residual_sigma(frame, model)
        extra = np.sqrt(max(0.0, s * s - floor_sigma * floor_sigma)) / 0.75   # 고주파만 본 만큼 키운다
        if os.environ.get("KED_DEBUG"):
            print(f"      noise sigma={s:.2f} floor={floor_sigma:.2f} extra={extra:.2f} up={upscale:.2f}")
        if extra < 6 / max(1.0, upscale):
            return None
        extra *= upscale
        if extra < 6:
            return None
        base = self.p.noise.get("default", {"amount": 1.0})
        strength = round(extra / noise_sigma(base, 1.0) * 4) / 4
        return ("default", max(0.25, strength)) if strength > 0 else None

    def flash(self, frames: Sequence[Tuple[float, np.ndarray]], models: Sequence[np.ndarray],
              before: Optional[np.ndarray]) -> Optional[Tuple[str, float, float]]:
        """줄 앞머리에서 화면 전체가 확 밝아졌다 사그라들면 플래시 — 봉우리 높이 · 색 · 길이로 프리셋을 고른다.

        frames: (줄 머리부터 초, 화면) — FLASH_SETTLE 뒤 프레임이 있어야 한다. models: 같은 순간의 오라클 예측(효과 없음).
        before: 앞 줄 정착 화면. 바닥은 '앞 줄 화면'과 '사그라든 뒤 화면'(둘 다 예측의 무대 변화만큼 옮김) 중 밝은 쪽 —
        효과가 바뀌는 줄(비네트 · 캐릭터 비주얼이 서서히 바뀜)은 두 바닥 사이를 오갈 뿐 둘 다를 넘지 않는다. 플래시만
        둘 다를 넘고, 그것도 화면 대부분에서 고르게 넘는다."""
        if not frames or before is None:
            return None
        times = np.array([t for t, _ in frames])
        tail = np.nonzero(times >= FLASH_SETTLE)[0]
        if tail.size == 0:
            return None
        after = np.median(np.stack([frames[i][1] for i in tail]), axis=0)
        after_model = np.mean(np.stack([models[i] for i in tail]), axis=0)
        m0 = models[0]
        h, w = self.mask.shape
        cells = [(slice(y * h // 4, (y + 1) * h // 4), slice(x * w // 4, (x + 1) * w // 4)) for y in range(4) for x in range(4)]
        cells = [c for c in cells if self.mask[c].mean() > 0.5]
        obs, bases, peak = [], [], (0.0, -1, 0.0)
        for i, ((t, f), m) in enumerate(zip(frames, models)):
            if t >= FLASH_SETTLE:
                break
            base = np.maximum(before + (m - m0), after + (m - after_model))
            excess = (f - base).max(2)
            per_cell = np.array([float(np.median(excess[c][self.mask[c]])) for c in cells]) if cells else np.zeros(1)
            level = float(np.median(per_cell))
            even = float((per_cell >= 0.4 * level).mean()) if level > 0 else 0.0
            obs.append((f - base)[self.mask].mean(0))
            bases.append(base)
            if os.environ.get("KED_DEBUG"):
                print(f"      flash t={t:.3f} level={level:.1f} even={even:.2f}")
            if level > peak[0] and even >= 0.75 and t <= FLASH_PEAK_BY:
                peak = (level, i, even)
        if peak[1] < 0 or peak[0] < FLASH_MIN_LEVEL:
            return None
        obs_a = np.array(obs)
        best = None
        for name, entry in self.p.flash.items():
            if name == "clear" or entry.get("amount", 0) <= 0:
                continue
            for s in STRENGTHS:
                pred = np.array([(apply_flash(b, entry, s, frames[i][0]) - b)[self.mask].mean(0)
                                 for i, b in enumerate(bases)])
                e = float(np.abs(pred - obs_a).mean())
                if best is None or e < best[0]:
                    best = (e, name, s)
        if best is None:
            return None
        return best[1], best[2], 1 - best[0] / max(1e-6, float(np.abs(obs_a).mean()))

    def compose_blur(self, layers, blur: Dict[str, float]) -> np.ndarray:
        """layers(슬롯 키, 층, 색, α)를 층마다 블러 양만큼 흐려 합성한다."""
        h, w = self.mask.shape
        canvas = np.zeros((h, w, 3), np.float32)
        for _, lay, col, al in layers:
            amount = blur.get(lay, 0.0)
            if amount > 0:
                sigma = amount * 6.0 * self.px_scale
                k = int(2 * round(sigma * 2) + 1)
                col, al = cv2.GaussianBlur(col, (k, k), sigma), cv2.GaussianBlur(al, (k, k), sigma)
            canvas = col * al[:, :, None] + canvas * (1 - al[:, :, None])
        return canvas

    def blur(self, frame: np.ndarray, layers, layer_name: str) -> Optional[Tuple[float, float]]:
        """한 층의 슬롯들이 예측보다 흐리면 그 층 블러 양(0.25 간격)."""
        region = np.zeros(self.mask.shape, bool)
        for i, (_, lay, _, alpha) in enumerate(layers):
            if lay == layer_name:
                vis = alpha > 0.95
                for _, _, _, a2 in layers[i + 1:]:
                    vis &= a2 < 0.05
                region |= vis
        region &= self.mask
        if region.sum() < 500:
            return None
        e0 = self._err(frame, self.compose_blur(layers, {}), region)
        best = min(((self._err(frame, self.compose_blur(layers, {layer_name: a}), region), a)
                    for a in (0.25, 0.5, 0.75, 1.0, 1.5)), key=lambda x: x[0])
        gain = 1 - best[0] / max(e0, 1e-6)
        return (best[1], gain) if gain > 0.3 and e0 > 4 else None


# ── 줄마다 효과 상태 → 커맨드 ─────────────────────────────────

@dataclass
class ObservedEffects:
    visual: Dict[str, Tuple[str, float]] = field(default_factory=dict)   # 슬롯 → (프리셋, 세기)
    vignette: Optional[Tuple[str, float]] = None
    noise: Optional[Tuple[str, float]] = None
    blur: Dict[str, float] = field(default_factory=dict)                # 층 → 양
    flash: Optional[Tuple[str, float]] = None                           # 그 줄에서만


def effect_commands(prev: ObservedEffects, cur: ObservedEffects) -> List[dict]:
    """앞 줄 효과 상태 → 이 줄 상태로 가는 커맨드(시간 인자는 런타임 생략값)."""
    out: List[dict] = []
    for slot in sorted(set(prev.visual) | set(cur.visual)):
        a, b = prev.visual.get(slot), cur.visual.get(slot)
        if a == b:
            continue
        if b is None:
            out.append({"name": "char_visual_clear", "args": [slot]})
        else:
            out.append({"name": f"char_visual_{b[0]}", "args": [slot, f"{b[1]:g}"]})
    if prev.vignette != cur.vignette:
        out.append({"name": "screen_vignette_clear", "args": []} if cur.vignette is None else
                   {"name": "screen_vignette", "args": [cur.vignette[0], f"{cur.vignette[1]:g}"]})
    if prev.noise != cur.noise:
        out.append({"name": "screen_noise_clear", "args": []} if cur.noise is None else
                   {"name": "screen_noise", "args": [cur.noise[0], f"{cur.noise[1]:g}"]})
    if prev.blur != cur.blur:
        if not cur.blur:
            out.append({"name": "screen_blur_reset", "args": []})
        else:
            for layer, amount in sorted(cur.blur.items()):
                if prev.blur.get(layer) != amount:
                    out.append({"name": "screen_blur", "args": [layer, f"{amount:g}"]})
    if cur.flash is not None:
        out.append({"name": "screen_flash", "args": [cur.flash[0], f"{cur.flash[1]:g}"]})
    return out
