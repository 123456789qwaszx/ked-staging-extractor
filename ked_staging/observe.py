"""관측 — 줄마다 정착 화면에서 배경 · 인물(정체 · 변형 · 표정) · 자리 · 반전을 찾는다.

자리는 자유 좌표로 찾지 않는다. 런타임 어휘로 설 수 있는 자리(place_left · center · right × focus × size 프리셋)를
오라클로 미리 계산해 두고(후보), 그림을 그 자리에 그려 영상과 가장 잘 맞는 것을 고른다(합성으로 해석하기).
어휘로 설명되지 않는 나머지 어긋남은 평행 이동으로 재서 move_by(u)로 남긴다 — 단위 환산도 오라클이 한다.

- 배경: 카탈로그 배경마다 같은 자리에 그려 대조 영역의 일치 비율이 가장 큰 것(없으면 '없음').
- 인물: 앞 줄 상태가 그대로 맞으면 그대로 둔다. 아니면 맞지 않는 인물을 빼고, 남은 인물에 하나씩 더해 보며
  오차를 가장 많이 줄이는 (인물 · 변형 · 자리 · 반전)을 고른다(탐욕). 한 인물은 한 슬롯에만 선다(런타임 cast 규칙).
- 표정: 같은 인물 · 변형의 표정 그림을 같은 자리에 바꿔 그려 가장 잘 맞는 것. 그림이 모두 같으면 앞 줄 표정을 잇는다.

대조는 분석 배율(기본 1/4)의 무대 영역(대사창 판 위)에서 하고, 오차는 채널 평균 절대차를 상한으로 자른 값이다.
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .catalog import Catalog, Portrait
from .effects import EffectPresets, VignetteMasks, visual_hypotheses, visual_layer
from .oracle import Oracle

ANALYSIS_SCALE = 0.25
ERROR_CAP = 60.0
PLACES = ("center", "left", "right")
FOCI = ("bust", "face")
DEPTH_ORDER = ("mid", "far", "back", "front", "close", "exp1", "exp2")
MIN_GAIN_PER_PIXEL = 6.0      # 인물을 더할 때 덮은 픽셀당 줄어야 하는 오차
KEEP_ERROR = 7.0              # 앞 줄 인물이 그대로라고 볼 영역 평균 오차
TOP_K = 6
SHOT_DROP = 0.08             # 앞 줄보다 배경 일치가 이만큼 떨어져도 다시 잰다
SHOT_ACCEPT = 0.02            # 다시 잰 카메라가 배경 일치를 이만큼은 올려야 받는다
SHOT_TRIGGER_ERROR = 6.0      # 앞 줄 무대가 지금 카메라로 이만큼 안 맞으면(평균 오차) 카메라를 다시 잰다
SHOT_IMPROVE = 0.75           # 다시 잰 카메라가 오차를 이 비율 아래로 줄여야 받는다
SHOT_TRIGGER = 0.6           # 앞 줄 배경의 일치 비율이 이보다 낮으면 카메라를 다시 잰다
UNEXPLAINED_SKIP = 0.01       # 설명 못 한 화면이 이 비율보다 적으면 인물을 더 찾지 않는다
CANONICAL_SNAP_PX = 8.0       # 이동 없는 어휘 자리로 바꿔 부를 허용(캔버스 px)
TRACK_REACH = 128            # 앞 줄 인물을 따라갈 때 평행 이동 범위(분석 px)
SHIFT_REACH = 48             # 평행 이동 탐색 범위(분석 px, 1/4 배율이면 화면 192px)
VISUAL_TRY = 0.5             # 앞 줄 인물 영역 오차가 KEEP_ERROR의 이 배를 넘으면 캐릭터 비주얼을 바꿔 본다
VISUAL_GAIN = 0.7            # 비주얼을 바꾸려면 오차가 이 비율 아래로 줄어야 한다
VISUAL_MIN_DROP = 2.0        # …그리고 이만큼은 줄어야 한다(평균 오차)
VIGNETTE_GAIN = 0.8          # 비네트를 바꾸려면 무대 평균 오차가 이 비율 아래로 줄어야 한다
VIGNETTE_EXPLAINED = 0.6     # …그리고 무대의 이만큼은 잘 맞아야 한다(오차 < 18)
VIGNETTE_QUIET = 3.0         # 지금 가정으로 무대 평균 오차가 이보다 작으면 비네트를 다시 고르지 않는다


# ── 후보 자리(오라클) ──────────────────────────────────────────

@dataclass
class Placement:
    place: str
    focus: str
    depth: str
    quad: np.ndarray            # (4,2) BL, BR, TL, TR — 캔버스 px
    depth_scale: float
    unit_x: np.ndarray          # move_by 1u 0u 일 때 화면에서 옮겨지는 px
    unit_y: np.ndarray          # move_by 0u 1u
    extras: Tuple[str, ...] = ()   # 자리 위에 남은 commit 모션(예: bow) — 이름 순서대로 접힌다

    def moved(self, mx: float, my: float) -> np.ndarray:
        return self.quad + self.unit_x * mx + self.unit_y * my

    def units_for(self, dx: float, dy: float) -> Tuple[float, float]:
        """화면 px 이동 → move_by (u, u)."""
        m = np.stack([self.unit_x, self.unit_y], 1)
        sol, *_ = np.linalg.lstsq(m, np.array([dx, dy]), rcond=None)
        return float(sol[0]), float(sol[1])


def emotion_arg(code: str) -> str:
    return str(int(code)) if (code or "").isdigit() else (code or "1")


def depth_presets(tuning_dir: str) -> List[str]:
    path = os.path.join(tuning_dir, "presets", "depth.json")
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        names = list(data.get("MonoBehaviour", data).get("presets", {}).keys())
    except (OSError, ValueError):
        names = []
    ordered = [d for d in DEPTH_ORDER if d in names] + [d for d in names if d not in DEPTH_ORDER]
    return ordered or ["mid"]


def placement_commands(slot: str, character: str, variant: str, emotion: str, place: str, focus: str,
                       depth: str, move: Tuple[float, float] = (0.0, 0.0), extras: Sequence[str] = ()) -> List[dict]:
    cmds = [{"name": "slot", "args": [slot, "stage00", "mid"]},
            {"name": "cast", "args": [slot, character, variant, emotion_arg(emotion)]},
            {"name": f"place_{place}", "args": [slot, focus, "0fr"]},
            {"name": "size", "args": [slot, depth, focus, "0fr"]}]
    if move != (0.0, 0.0):
        cmds.append({"name": "move_by", "args": [slot, f"{move[0]:g}u", f"{move[1]:g}u", "0fr"]})
    for m in extras:
        cmds.append({"name": "motion", "args": [slot, m]})
    cmds.append({"name": "fade_in", "args": [slot, "0fr"]})
    return cmds


def shot_commands(shot: Optional[Tuple[float, float, float]]) -> List[dict]:
    """관측한 카메라 → 접을 커맨드. None이면 샷 없음(shot_reset 상태)."""
    if shot is None:
        return []
    zoom, x, y = shot
    return [{"name": "shot_to", "args": [f"{zoom:g}", f"{x:g}u", f"{y:g}u", "0fr"]}]


def load_motions(path: Optional[str]) -> List[dict]:
    """motions.json(schema motions/2)의 이름 · 끝(restore | commit) · 길이."""
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    return [{"name": m["name"], "end": m.get("end", "restore"), "duration": float(m.get("duration", 0))}
            for m in data.get("motions", []) if m.get("rig", "character") == "character"]


class Candidates:
    """인물 · 변형마다 설 수 있는 자리 목록과 배경 자리 — 오라클로 한 번 계산해 둔다."""

    def __init__(self, oracle: Oracle, catalog: Catalog, depths: Sequence[str],
                 shot: Optional[Tuple[float, float, float]] = None, commits: Sequence[str] = ()):
        self.oracle = oracle
        self.catalog = catalog
        self.depths = list(depths)
        self.shot = shot
        self.prefix = shot_commands(shot)
        self.commits = list(commits)
        self.identities: List[Tuple[str, str]] = sorted({(p.character, p.variant) for p in catalog.portraits})
        self._placements: Dict[Tuple[str, str], List[Placement]] = {}
        self._bg: Dict[str, Optional[np.ndarray]] = {}
        self._build()

    def emotions(self, character: str, variant: str) -> List[Portrait]:
        return sorted([p for p in self.catalog.portraits if p.character == character and p.variant == variant],
                      key=lambda p: p.emotion)

    def _build(self) -> None:
        by_variant: Dict[str, List[str]] = {}
        for c, v in self.identities:
            by_variant.setdefault(v, []).append(c)
        extras_list = [()] + [(m,) for m in self.commits]
        for (place, focus, depth), variant, extras in itertools.product(
                itertools.product(PLACES, FOCI, self.depths), sorted(by_variant), extras_list):
            chars = by_variant[variant]
            quads = {}
            for move in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
                cmds = list(self.prefix)
                for i, c in enumerate(chars):
                    e = self.emotions(c, variant)[0].emotion
                    cmds += placement_commands(f"q{i}", c, variant, e, place, focus, depth, move, extras)
                snap = self.oracle.fold([cmds])[0]
                for s in snap.slots:
                    if s.get("quad"):
                        quads[(s["character"], move)] = (np.asarray(s["quad"], np.float64), float(s["depthScale"]))
            for c in chars:
                if (c, (0.0, 0.0)) not in quads:
                    continue
                q0, ds = quads[(c, (0.0, 0.0))]
                ux = quads[(c, (1.0, 0.0))][0][0] - q0[0]
                uy = quads[(c, (0.0, 1.0))][0][0] - q0[0]
                lst = self._placements.setdefault((c, variant), [])
                if any(np.abs(p.quad - q0).max() < 0.5 for p in lst):
                    continue  # 같은 자리를 다른 이름으로 — 앞(정본에 가까운) 이름을 둔다
                lst.append(Placement(place, focus, depth, q0, ds, ux, uy, tuple(extras)))

    def placements(self, character: str, variant: str) -> List[Placement]:
        return self._placements.get((character, variant), [])

    def find(self, character: str, variant: str, place: str, focus: str, depth: str,
             extras: Sequence[str] = ()) -> Optional[Placement]:
        for p in self.placements(character, variant):
            if (p.place, p.focus, p.depth, p.extras) == (place, focus, depth, tuple(extras)):
                return p
        return None

    def background_quad(self, sprite: str) -> Optional[np.ndarray]:
        if sprite not in self._bg:
            snap = self.oracle.fold([self.prefix + [{"name": "bg_spawn", "args": ["bg_main", sprite]},
                                                    {"name": "bg_fade_in", "args": ["bg_main", "0fr"]}]])[0]
            quad = next((s["quad"] for s in snap.slots if s.get("background") and s.get("quad")), None)
            self._bg[sprite] = None if quad is None else np.asarray(quad, np.float64)
        return self._bg[sprite]


# ── 빠른 그리기 ───────────────────────────────────────────────

class Painter:
    """분석 배율로 그림을 quad 자리에 편다. 줄인 그림은 배율별로 캐시한다."""

    def __init__(self, catalog: Catalog, size: Tuple[int, int], scale: float):
        self.catalog = catalog
        self.size = size            # (w, h) 분석 배율
        self.scale = scale          # 캔버스 px → 분석 px
        self._cache: Dict[Tuple[str, bool, int], np.ndarray] = {}

    def _sprite(self, entry, mirror: bool, width: int) -> np.ndarray:
        key = (entry.path, mirror, width)
        img = self._cache.get(key)
        if img is None:
            src = self.catalog.image(entry)
            h = max(1, int(round(src.shape[0] * width / src.shape[1])))
            img = cv2.resize(src, (width, h), interpolation=cv2.INTER_AREA)
            if mirror:
                img = img[:, ::-1].copy()
            self._cache[key] = img
        return img

    def layer(self, entry, quad: np.ndarray, mirror: bool = False, margin: int = 0
              ) -> Tuple[np.ndarray, np.ndarray]:
        """margin을 주면 화면 둘레를 margin px씩 넓힌 캔버스에 그린다(평행 이동 탐색용)."""
        bl, br, tl, tr = (np.asarray(p, np.float64) * self.scale + margin for p in quad)
        target_w = float(np.linalg.norm(tr - tl))
        width = max(8, int(np.ceil(target_w / 4.0)) * 4)
        img = self._sprite(entry, mirror, min(width, self.catalog.image(entry).shape[1]))
        sh, sw = img.shape[:2]
        m = cv2.getAffineTransform(np.float32([[0, 0], [sw, 0], [0, sh]]), np.float32([tl, tr, bl]))
        size = (self.size[0] + 2 * margin, self.size[1] + 2 * margin)
        warped = cv2.warpAffine(img, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0)).astype(np.float32)
        return warped[:, :, :3], warped[:, :, 3] / 255.0


def robust_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(np.abs(a - b).mean(2), ERROR_CAP)


# ── 관측 결과 ────────────────────────────────────────────────

@dataclass
class ObservedSlot:
    character: str
    variant: str
    emotion: str
    place: str
    focus: str
    depth: str
    mirror: bool
    move: Tuple[float, float]        # u
    error: float                     # 보이는 영역 평균 오차
    visible: int                     # 보이는 픽셀(분석 배율)
    emotion_sure: bool = True
    quad: Optional[List[List[float]]] = None
    extras: Tuple[str, ...] = ()     # 자리 위에 남은 commit 모션

    def key(self) -> Tuple:
        return (self.character, self.variant, self.emotion, self.place, self.focus, self.depth, self.mirror,
                self.move, self.extras)

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["move"] = [round(self.move[0], 3), round(self.move[1], 3)]
        d["error"] = round(self.error, 2)
        d["extras"] = list(self.extras)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "ObservedSlot":
        d = dict(d)
        d["move"] = tuple(d["move"])
        d["extras"] = tuple(d.get("extras", ()))
        return cls(**d)


@dataclass
class ObservedStage:
    time: float
    background: Optional[str]
    background_match: float
    slots: List[ObservedSlot]        # 먼 것 먼저(그리기 순서)
    error: float                     # 무대 영역 평균 오차
    residual: float                  # 오차가 큰(>25) 픽셀 비율
    notes: List[str] = field(default_factory=list)
    shot: Optional[Tuple[float, float, float]] = None   # 카메라(shot_to zoom x y) — None은 샷 없음
    # 화면 효과(근사, P3): {"vignette": [프리셋, 세기] | None, "visual": {인물: [프리셋, 세기]}}
    effects: Dict = field(default_factory=dict)

    @property
    def vignette(self) -> Optional[Tuple[str, float]]:
        v = (self.effects or {}).get("vignette")
        return (str(v[0]), float(v[1])) if v else None

    @property
    def visual(self) -> Dict[str, Tuple[str, float]]:
        return {c: (str(v[0]), float(v[1])) for c, v in ((self.effects or {}).get("visual") or {}).items()}

    def to_json(self) -> dict:
        return {"time": round(self.time, 3), "background": self.background,
                "backgroundMatch": round(self.background_match, 3), "slots": [s.to_json() for s in self.slots],
                "error": round(self.error, 2), "residual": round(self.residual, 4), "notes": self.notes,
                "shot": list(self.shot) if self.shot else None, "effects": self.effects}

    @classmethod
    def from_json(cls, d: dict) -> "ObservedStage":
        return cls(d["time"], d["background"], d["backgroundMatch"], [ObservedSlot.from_json(x) for x in d["slots"]],
                   d["error"], d["residual"], list(d.get("notes", [])),
                   tuple(d["shot"]) if d.get("shot") else None, dict(d.get("effects") or {}))


@dataclass
class _Fit:
    character: str
    variant: str
    entry: Portrait
    placement: Placement
    mirror: bool
    move: Tuple[float, float]
    color: np.ndarray
    alpha: np.ndarray
    visual: Optional[Tuple[str, float]] = None     # 입혀 그린 캐릭터 비주얼(프리셋, 세기)

    @property
    def quad(self) -> np.ndarray:
        return self.placement.moved(*self.move)


# ── 관측기 ───────────────────────────────────────────────────

class Observer:
    def __init__(self, oracle: Oracle, catalog: Catalog, tuning_dir: str, stage_bottom: float,
                 scale: float = ANALYSIS_SCALE, unit_step: float = 0.25, motions_path: Optional[str] = None,
                 effects: Optional[EffectPresets] = None):
        self.oracle = oracle
        # 화면 효과(P3, 근사) — 주면 비네트를 걷어 내고 보고, 앞 줄 인물의 색 바뀜(dim · 실루엣 …)을 비주얼로 설명한다.
        self.fx = effects
        self.vmasks = VignetteMasks(effects) if effects is not None else None
        self.visual_hyps = visual_hypotheses(effects) if effects is not None else []
        self.px_scale = scale / 0.25
        self.catalog = catalog
        self.scale = scale
        self.size = (int(round(oracle.width * scale)), int(round(oracle.height * scale)))
        self.painter = Painter(catalog, self.size, scale)
        self.depths = depth_presets(tuning_dir)
        self.motions = load_motions(motions_path)
        self.commits = [m["name"] for m in self.motions if m["end"] == "commit"]
        self._cands: Dict[Optional[Tuple[float, float, float]], Candidates] = {}
        self.cands = self.candidates(None)
        self.mask = np.zeros((self.size[1], self.size[0]), np.float32)
        self.mask[:int(np.floor(stage_bottom * scale)), :] = 1.0
        self.unit_step = unit_step
        self._bg_layers: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._coarse: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
        self._zoom_tables: Dict[str, List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]] = {}

    def candidates(self, shot: Optional[Tuple[float, float, float]]) -> Candidates:
        """카메라마다 자리 후보 — 샷이 바뀌면 같은 place도 화면 자리가 다르다(층마다 시차도 있다)."""
        if shot not in self._cands:
            self._cands[shot] = Candidates(self.oracle, self.catalog, self.depths, shot, self.commits)
        return self._cands[shot]

    # ── 배경 ──
    def _bg_layer(self, sprite: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        key = (sprite, self.cands.shot)
        if key not in self._bg_layers:
            entry = self.catalog.background(sprite)
            quad = self.cands.background_quad(sprite) if entry else None
            if entry is None or quad is None:
                return None
            self._bg_layers[key] = self.painter.layer(entry, quad)
        return self._bg_layers[key]

    # ── 카메라 ──
    def _zooms(self, sprite: str) -> List[Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]:
        """샷 배율표(배경 그림마다) — zoom마다 (zoom, 배경 폭 비율, 배경 사각형, x 1u당 이동, y 1u당 이동). 오라클이 계산한다."""
        if sprite not in self._zoom_tables:
            table = []
            base_w = None
            for zoom in np.arange(-10.0, 20.01, 0.5):
                quads = []
                for x, y in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)):
                    snap = self.oracle.fold([shot_commands((float(zoom), x, y)) + [
                        {"name": "bg_spawn", "args": ["bg_main", sprite]},
                        {"name": "bg_fade_in", "args": ["bg_main", "0fr"]}]])[0]
                    q = next((np.asarray(sl["quad"], np.float64) for sl in snap.slots if sl.get("quad")), None)
                    quads.append(q)
                if any(q is None for q in quads):
                    continue
                w = float(np.linalg.norm(quads[0][1] - quads[0][0]))
                if abs(zoom) < 1e-9:
                    base_w = w
                table.append((float(zoom), w, quads[0], quads[1] - quads[0], quads[2] - quads[0]))
            base_w = base_w or (table[0][1] if table else 1.0)
            self._zoom_tables[sprite] = [(z, w / base_w, q, ux, uy) for z, w, q, ux, uy in table]
        return self._zoom_tables[sprite]

    def _estimate_shot(self, frame: np.ndarray, sprite: str) -> Optional[Tuple[float, float, float]]:
        """배경 특징점을 맞춰 카메라 배율 · 이동을 재고 shot_to(zoom, x, y)로 바꾼다(0.5 · 0.25u 격자)."""
        entry = self.catalog.background(sprite)
        if entry is None or not self.catalog.backgrounds:
            return None
        ref_cands = self.candidates(None)
        quad0 = ref_cands.background_quad(sprite)
        if quad0 is None:
            return None
        color, alpha = self.painter.layer(entry, quad0)
        ref = cv2.cvtColor(np.clip(color * alpha[:, :, None], 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        cur = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        m8 = (self.mask > 0).astype(np.uint8) * 255
        orb = cv2.ORB_create(3000, fastThreshold=5)
        k1, d1 = orb.detectAndCompute(ref, m8)
        k2, d2 = orb.detectAndCompute(cur, m8)
        if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
            return None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            return None
        src = np.float32([k1[m.queryIdx].pt for m in good])
        dst = np.float32([k2[m.trainIdx].pt for m in good])
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is None or inliers is None or int(inliers.sum()) < 8:
            return None
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        # 특징점 추정(배율 · 이동)은 거칠다 — 그 근처 배율마다 배경을 그려 밀어 보며 가장 잘 맞는 (zoom, x, y)를 고른다.
        q0 = quad0 * self.scale
        seen = (q0 @ M[:, :2].T + M[:, 2]) / self.scale
        w = self.mask > 0
        best = None
        for zoom, ratio, quad, ux, uy in self._zooms(sprite):
            if abs(ratio - scale) > 0.06:
                continue
            a = np.stack([ux.reshape(-1), uy.reshape(-1)], 1)
            sol, *_ = np.linalg.lstsq(a, (seen - quad).reshape(-1), rcond=None)
            m = 24
            col, alp = self.painter.layer(entry, quad + ux * sol[0] + uy * sol[1], margin=m)
            img = col * alp[:, :, None]
            h, wd = frame.shape[:2]
            fr, mk = frame[::2, ::2], w[::2, ::2]
            best_local = None
            for dy in range(-m, m + 1, 2):
                for dx in range(-m, m + 1, 2):
                    win = img[m - dy:m - dy + h:2, m - dx:m - dx + wd:2]
                    sc = float((robust_error(fr, win)[mk] < 18).mean())
                    if best_local is None or sc > best_local[0]:
                        best_local = (sc, dx, dy)
            sc, dx, dy = best_local
            du, dv = np.linalg.lstsq(np.stack([ux[0], uy[0]], 1), np.array([dx, dy]) / self.scale, rcond=None)[0]
            x = float(np.round((sol[0] + du) / 0.25) * 0.25)
            y = float(np.round((sol[1] + dv) / 0.25) * 0.25)
            if best is None or sc > best[0]:
                best = (sc, zoom, x, y)
        if best is None:
            return None
        _, zoom, x, y = best
        if abs(zoom) < 1e-9 and x == 0 and y == 0:
            return None
        return (zoom, x, y)

    def _shot_error(self, frame: np.ndarray, stage: "ObservedStage", shot) -> float:
        """앞 줄 무대(배경 + 인물)를 shot 카메라로 그렸을 때 무대 영역 평균 오차 — 오라클로 한 번 접어 그린다."""
        img = self._stage_image(stage, shot)
        return float(robust_error(frame, img)[self.mask > 0].mean())

    def _stage_image(self, stage: "ObservedStage", shot) -> np.ndarray:
        """관측한 무대를 shot 카메라로 다시 그린다(캐릭터 비주얼은 입히고, 비네트는 뺀 화면)."""
        cmds = shot_commands(shot) + [{"name": "bg_spawn", "args": ["bg_main", stage.background]},
                                      {"name": "bg_fade_in", "args": ["bg_main", "0fr"]}]
        for i, sl in enumerate(stage.slots):
            key = f"q{i}"
            cmds += placement_commands(key, sl.character, sl.variant, sl.emotion, sl.place, sl.focus, sl.depth,
                                       sl.move, sl.extras)
            if sl.mirror:
                cmds.append({"name": "mirror", "args": [key, "left"]})
        snap = self.oracle.fold([cmds])[0]
        visual = stage.visual if self.fx is not None else {}
        canvas = np.zeros((self.size[1], self.size[0], 3), np.float32)
        for sl in snap.slots:
            value = float(sl.get("alpha", 0.0))
            if not sl.get("quad") or value <= 0.003:
                continue
            if sl.get("background"):
                entry = self.catalog.background(sl.get("sprite") or "")
            else:
                entry = self.catalog.portrait(sl.get("character") or "", sl.get("variant") or "a",
                                              sl.get("emotion") or "")
            if entry is None:
                continue
            color, alpha = self.painter.layer(entry, np.asarray(sl["quad"], np.float64), bool(sl.get("toolMirror")))
            v = visual.get(sl.get("character") or "") if not sl.get("background") else None
            if v and v[0] in self.fx.visual:
                color, alpha = visual_layer(color, alpha, self.fx.visual[v[0]], v[1], self.px_scale)
            a = (alpha * value)[:, :, None]
            canvas = color * a + canvas * (1 - a)
        return canvas

    # ── 화면 효과(근사) ──
    def _choose_vignette(self, frame: np.ndarray, img: np.ndarray, current: Optional[Tuple[str, float]]
                         ) -> Optional[Tuple[str, float]]:
        """img(비네트 없는 예측)에 비네트 후보를 씌워 frame과 견준다. 지금 가정보다 확실히 나을 때만 바꾼다."""
        w = self.mask > 0

        def score(h):
            e = robust_error(frame, self.vmasks.apply(img, h))[w]
            return float(e.mean()), float((e < 18).mean())

        e_cur, _ = score(current)
        if e_cur < VIGNETTE_QUIET:
            return current
        best = (e_cur, 0.0, current)
        for h in [None] + self.vmasks.hypotheses():
            if h == current:
                continue
            e, ok = score(h)
            if e < best[0]:
                best = (e, ok, h)
        if best[2] is not None:
            # 비네트가 거의 다 덮는 화면은 '아무것도 없는(검은) 무대'로도 설명된다 — 그보다 확실히 나아야 비네트다.
            e_dark = float(robust_error(frame, np.zeros_like(img))[w].mean())
            if best[0] >= VIGNETTE_GAIN * e_dark:
                return None if current is not None and e_dark < e_cur else current
        if best[2] != current and best[0] < VIGNETTE_GAIN * e_cur and best[1] > VIGNETTE_EXPLAINED:
            return best[2]
        return current

    def _with_visual(self, fit: _Fit, v: Optional[Tuple[str, float]]) -> _Fit:
        color, alpha = self.painter.layer(fit.entry, fit.quad, fit.mirror)
        if v and self.fx is not None and v[0] in self.fx.visual:
            color, alpha = visual_layer(color, alpha, self.fx.visual[v[0]], v[1], self.px_scale)
        else:
            v = None
        return _Fit(fit.character, fit.variant, fit.entry, fit.placement, fit.mirror, fit.move, color, alpha, v)

    def _best_visual(self, frame: np.ndarray, base: np.ndarray, fits: List[_Fit], i: int
                     ) -> Optional[Tuple[float, float, _Fit]]:
        """fits[i]에 캐릭터 비주얼 후보(없음 포함)를 입혀 본다 → (지금 오차, 가장 나은 오차, 그 맞춤)."""
        f = fits[i]
        raw_c, raw_a = self.painter.layer(f.entry, f.quad, f.mirror)
        sl = _bbox_slice(raw_a)
        if sl is None:
            return None
        region = (raw_a[sl] > 0.95) & (self.mask[sl] > 0)
        for g in fits[i + 1:]:
            region &= g.alpha[sl] < 0.05
        if region.sum() < 50:
            return None
        below = self._compose(base[sl], [_crop_fit(g, sl) for g in fits[:i]])
        above = fits[i + 1:]

        def err_of(c, a):
            a3 = a[sl][:, :, None]
            out = c[sl] * a3 + below * (1 - a3)
            for g in above:
                ga = g.alpha[sl][:, :, None]
                out = g.color[sl] * ga + out * (1 - ga)
            return float(robust_error(frame[sl], out)[region].mean())

        cur = err_of(f.color, f.alpha)
        best = (err_of(raw_c, raw_a), None, raw_c, raw_a)
        for v in self.visual_hyps:
            c, a = visual_layer(raw_c, raw_a, self.fx.visual[v[0]], v[1], self.px_scale)
            e = err_of(c, a)
            if e < best[0]:
                best = (e, v, c, a)
        e, v, c, a = best
        if v is not None:
            # 어둡게 입힌 인물(실루엣 · dim)은 '인물이 없는' 자리로도 설명될 수 있다 — 그보다 확실히 나아야 비주얼이다.
            gone = err_of(np.zeros_like(raw_c), np.zeros_like(raw_a))
            if e >= VISUAL_GAIN * gone:
                return None
        return cur, e, _Fit(f.character, f.variant, f.entry, f.placement, f.mirror, f.move, c, a, v)

    def _background(self, frame: np.ndarray) -> Tuple[Optional[str], float, np.ndarray]:
        best, best_score, best_img = None, 0.0, np.zeros_like(frame)
        w = self.mask > 0
        for b in self.catalog.backgrounds:
            layer = self._bg_layer(b.key)
            if layer is None:
                continue
            color, alpha = layer
            img = color * alpha[:, :, None]
            score = float((robust_error(frame, img)[w] < 18).mean())
            if score > best_score:
                best, best_score, best_img = b.key, score, img
        if best_score < 0.2:
            return None, best_score, np.zeros_like(frame)
        return best, best_score, best_img

    # ── 합성 ──
    @staticmethod
    def _compose(base: np.ndarray, fits: Sequence[_Fit]) -> np.ndarray:
        out = base.copy()
        for f in fits:
            a = f.alpha[:, :, None]
            out = f.color * a + out * (1 - a)
        return out

    def _order_index(self, fits: Sequence[_Fit], cand: _Fit) -> List[int]:
        """오라클 그리기 순서(깊이 배율이 작을수록 먼저)에 맞는 끼울 자리. 같은 배율이면 앞뒤 둘 다."""
        ds = cand.placement.depth_scale
        lo = sum(1 for f in fits if f.placement.depth_scale < ds - 1e-6)
        hi = sum(1 for f in fits if f.placement.depth_scale <= ds + 1e-6)
        return list(range(lo, hi + 1))

    def _gain(self, frame: np.ndarray, base: np.ndarray, fits: List[_Fit], cand: _Fit, pos: int,
              current_err: np.ndarray) -> Tuple[float, int]:
        ys, xs = np.nonzero(cand.alpha > 0.02)
        if len(xs) == 0:
            return -1e9, 0
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sl = (slice(y0, y1), slice(x0, x1))
        below = self._compose(base[sl], [_crop_fit(f, sl) for f in fits[:pos]])
        a = cand.alpha[sl][:, :, None]
        out = cand.color[sl] * a + below * (1 - a)
        for f in fits[pos:]:
            fa = f.alpha[sl][:, :, None]
            out = f.color[sl] * fa + out * (1 - fa)
        new_err = robust_error(frame[sl], out)
        m = self.mask[sl]
        gain = float(((current_err[sl] - new_err) * m).sum())
        covered = int(((cand.alpha[sl] > 0.5) * m).sum())
        return gain, covered

    def _make_fit(self, character: str, variant: str, entry: Portrait, pl: Placement, mirror: bool,
                  move: Tuple[float, float] = (0.0, 0.0)) -> _Fit:
        color, alpha = self.painter.layer(entry, pl.moved(*move), mirror)
        return _Fit(character, variant, entry, pl, mirror, move, color, alpha)

    def _scan(self, frame: np.ndarray, base: np.ndarray, fits: List[_Fit], fit: _Fit, pos: int,
              xs: Sequence[int], ys: Sequence[int], step: int, margin: int = SHIFT_REACH,
              layer=None) -> Tuple[float, int, int, float]:
        """fit을 (dx, dy)만큼 옮겨 pos 자리에 끼웠을 때의 오차 합을 xs × ys에서 재서 가장 작은 것을 돌려준다.
        넓힌 캔버스에 한 번만 그리고 창을 밀어 가며 잰다(step 간격으로 성기게). → (오차, dx, dy, 끼우기 전 오차)"""
        m = margin
        color, alpha = layer if layer is not None else self.painter.layer(fit.entry, fit.quad, fit.mirror, margin=m)
        h, w = frame.shape[:2]
        below = self._compose(base, fits[:pos])[::step, ::step]
        fr = frame[::step, ::step]
        mk = self.mask[::step, ::step]
        ac = aa = None
        for f in fits[pos:]:
            fa = f.alpha[::step, ::step]
            if ac is None:
                ac = np.zeros_like(fr)
                aa = np.zeros(fr.shape[:2], np.float32)
            ac = f.color[::step, ::step] * fa[:, :, None] + ac * (1 - fa[:, :, None])
            aa = fa + aa * (1 - fa)

        def compose(out):
            return out if ac is None else ac + out * (1 - aa[:, :, None])

        before = float((robust_error(fr, compose(below)) * mk).sum())
        best = (float("inf"), 0, 0)
        for dy in ys:
            for dx in xs:
                if abs(dx) > m or abs(dy) > m:
                    continue
                win = (slice(m - dy, m - dy + h, step), slice(m - dx, m - dx + w, step))
                a = alpha[win][:, :, None]
                out = compose(color[win] * a + below * (1 - a))
                c = float((robust_error(fr, out) * mk).sum())
                if c < best[0]:
                    best = (c, dx, dy)
        return best[0], best[1], best[2], before

    def _rank(self, frame: np.ndarray, base: np.ndarray, fits: List[_Fit], cand: _Fit, pos: int,
              xs: Sequence[int] = range(-40, 41, 8), step: int = 4) -> float:
        """후보 순위용 이득(성기게) — 넓힌 캔버스 그림을 step 간격으로 줄여 캐시해 두고 가로로만 밀어 본다."""
        m = SHIFT_REACH
        key = (cand.entry.path, cand.placement.place, cand.placement.focus, cand.placement.depth,
               cand.placement.extras, cand.mirror, self.cands.shot)
        layer = self._coarse.get(key)
        if layer is None:
            color, alpha = self.painter.layer(cand.entry, cand.quad, cand.mirror, margin=m)
            layer = (np.ascontiguousarray(color[::step, ::step]), np.ascontiguousarray(alpha[::step, ::step]))
            self._coarse[key] = layer
        color, alpha = layer
        h, w = frame.shape[0] // step + (frame.shape[0] % step > 0), frame.shape[1] // step + (frame.shape[1] % step > 0)
        below = self._compose(base, fits[:pos])[::step, ::step]
        fr = frame[::step, ::step]
        mk = self.mask[::step, ::step]
        ac = aa = None
        for f in fits[pos:]:
            fa = f.alpha[::step, ::step]
            if ac is None:
                ac = np.zeros_like(fr)
                aa = np.zeros(fr.shape[:2], np.float32)
            ac = f.color[::step, ::step] * fa[:, :, None] + ac * (1 - fa[:, :, None])
            aa = fa + aa * (1 - fa)

        def compose(out):
            return out if ac is None else ac + out * (1 - aa[:, :, None])

        before = float((robust_error(fr, compose(below)) * mk).sum())
        best = float("inf")
        o = m // step
        for dx in xs:
            j = o - dx // step
            win = (slice(o, o + h), slice(j, j + w))
            a = alpha[win][:, :, None]
            out = compose(color[win] * a + below * (1 - a))
            best = min(best, float((robust_error(fr, out) * mk).sum()))
        return before - best

    def _shift_search(self, frame: np.ndarray, base: np.ndarray, fits: List[_Fit], fit: _Fit, pos: int,
                      reach: int = SHIFT_REACH) -> Tuple[float, float]:
        """어휘 자리에서 얼마나 옮겨져 있나(분석 px). 가로를 넓게 → 세로 → 가로를 촘촘히(move_by는 대개 가로다)."""
        layer = self.painter.layer(fit.entry, fit.quad, fit.mirror, margin=reach)
        _, bx, _, _ = self._scan(frame, base, fits, fit, pos, range(-reach, reach + 1, 2), [0], 2, reach, layer)
        _, _, by, _ = self._scan(frame, base, fits, fit, pos, [bx], range(-reach // 2, reach // 2 + 1, 2), 2,
                                 reach, layer)
        _, bx, by, _ = self._scan(frame, base, fits, fit, pos, range(bx - 2, bx + 3), range(by - 1, by + 2), 2,
                                  reach, layer)
        return float(bx), float(by)

    def _refine(self, frame: np.ndarray, base: np.ndarray, fits: List[_Fit], fit: _Fit, pos: int,
                reach: int = SHIFT_REACH) -> _Fit:
        """평행 이동을 재서 move_by(u)를 정한다 — 피라미드 ECC, 결과는 unit_step 단위로 반올림."""
        move = np.array(fit.move, np.float64)
        sx, sy = self._shift_search(frame, base, fits, fit, pos, reach)
        if sx or sy:
            du, dv = fit.placement.units_for(sx / self.scale, sy / self.scale)
            move = move + [du, dv]
            q = np.round(move / self.unit_step) * self.unit_step
            fit = self._make_fit(fit.character, fit.variant, fit.entry, fit.placement, fit.mirror,
                                 (float(q[0]), float(q[1])))
        for _ in range(3):
            model = self._compose(base, fits[:pos] + [fit] + fits[pos:])
            vis = fit.alpha > 0.95
            for f in fits[pos:]:
                vis &= f.alpha < 0.05
            vis &= self.mask > 0
            vis = cv2.erode(vis.astype(np.uint8), np.ones((5, 5), np.uint8))
            if vis.sum() < 400:
                break
            shift = _ecc_shift(model, frame, vis)
            if shift is None:
                break
            dx, dy = shift[0] / self.scale, shift[1] / self.scale
            if abs(dx) < 1.0 and abs(dy) < 1.0:
                break
            du, dv = fit.placement.units_for(dx, dy)
            move = move + [du, dv]
            q = np.round(move / self.unit_step) * self.unit_step
            fit = self._make_fit(fit.character, fit.variant, fit.entry, fit.placement, fit.mirror,
                                 (float(q[0]), float(q[1])))
        return fit

    def _track(self, frame: np.ndarray, base: np.ndarray, fits: List[_Fit], lost: _Fit
               ) -> Optional[Tuple[_Fit, int]]:
        """앞 줄 인물을 같은 자리 어휘 · 반전 그대로 넓게 밀어 찾는다. 잘 맞으면 (맞춘 것, 끼울 자리)."""
        cand = self._make_fit(lost.character, lost.variant, lost.entry, lost.placement, lost.mirror, lost.move)
        pos = self._order_index(fits, cand)[-1]
        fit = self._refine(frame, base, fits, cand, pos, TRACK_REACH)
        model = self._compose(base, fits[:pos] + [fit] + fits[pos:])
        vis = (fit.alpha > 0.95) & (self.mask > 0)
        for g in fits[pos:]:
            vis &= g.alpha < 0.05
        if vis.sum() < 200 or float(robust_error(frame, model)[vis].mean()) > KEEP_ERROR * 1.6:
            return None
        if fit.move == lost.move:
            return None   # 제자리에서 안 맞던 것 — 다른 변화(표정 · 크기 …)다
        return self._canonical(fit), pos

    def _canonical(self, fit: _Fit) -> _Fit:
        """옮겨 찾은 자리가 이동 없는 어휘 자리(place × focus × size)와 같으면 그 이름으로 — 대본은 move_by보다 place가 읽기 쉽다."""
        quad = fit.quad
        tol = CANONICAL_SNAP_PX
        for pl in self.cands.placements(fit.character, fit.variant):
            if np.abs(pl.quad - quad).max() <= tol:
                if pl is fit.placement and fit.move == (0.0, 0.0):
                    return fit
                return self._make_fit(fit.character, fit.variant, fit.entry, pl, fit.mirror, (0.0, 0.0))
        return fit

    def _choose_emotion(self, frame: np.ndarray, base: np.ndarray, fits: List[_Fit], i: int,
                        previous: Optional[str]) -> Tuple[_Fit, bool]:
        f = fits[i]
        options = self.cands.emotions(f.character, f.variant)
        if len(options) <= 1:
            return f, True
        sl = _bbox_slice(f.alpha)
        if sl is None:
            return f, False
        below = self._compose(base[sl], [_crop_fit(g, sl) for g in fits[:i]])
        scores = []
        for entry in options:
            color, alpha = self.painter.layer(entry, f.quad, f.mirror)
            if f.visual and self.fx is not None:
                color, alpha = visual_layer(color, alpha, self.fx.visual[f.visual[0]], f.visual[1], self.px_scale)
            a = alpha[sl][:, :, None]
            out = color[sl] * a + below * (1 - a)
            for g in fits[i + 1:]:
                ga = g.alpha[sl][:, :, None]
                out = g.color[sl] * ga + out * (1 - ga)
            scores.append((float((robust_error(frame[sl], out) * self.mask[sl]).sum()), entry, color, alpha))
        scores.sort(key=lambda s: s[0])
        best = scores[0]
        spread = scores[-1][0] - best[0]
        sure = len(scores) > 1 and (scores[1][0] - best[0]) > max(30.0, 0.02 * best[0])
        if spread < 1.0 and previous:
            # 표정 그림이 모두 같다 — 화면으로는 모른다. 앞 줄 표정을 잇는다.
            keep = next((s for s in scores if s[1].emotion == previous), best)
            best = keep
            sure = False
        _, entry, color, alpha = best
        return _Fit(f.character, f.variant, entry, f.placement, f.mirror, f.move, color, alpha, f.visual), sure

    # ── 한 줄 ──
    def observe(self, frame_full: np.ndarray, time: float = 0.0, previous: Optional[ObservedStage] = None
                ) -> ObservedStage:
        frame = cv2.resize(frame_full, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.fx is None:
            return self._observe(frame, time, previous, None)
        # 비네트: 앞 줄 무대를 다시 그려 견주고(무대는 그대로 · 효과만 바뀐 줄), 다 본 뒤 맞춘 무대로 한 번 더 견준다
        # (장면이 바뀌며 비네트도 함께 바뀐 줄). 뒤 점검에서 바뀌면 그 가정으로 다시 본다.
        vig = previous.vignette if previous is not None else None
        if previous is not None and previous.background:
            vig = self._choose_vignette(frame, self._stage_image(previous, previous.shot), vig)
        st = self._observe(frame, time, previous, vig)
        again = self._choose_vignette(frame, self._last_model, vig)
        if again != vig:
            st = self._observe(frame, time, previous, again)
        return st

    def _observe(self, frame: np.ndarray, time: float, previous: Optional[ObservedStage],
                 vignette: Optional[Tuple[str, float]]) -> ObservedStage:
        if vignette is not None:
            frame = self.vmasks.undo(frame, vignette)
        notes: List[str] = []
        shot = previous.shot if previous is not None else None
        self.cands = self.candidates(shot)
        bg, bg_score, base = self._background(frame)
        # 카메라: 앞 줄 무대 전체(배경 + 인물)를 지금 샷으로 그려 잘 안 맞으면 샷을 다시 잰다.
        if previous is not None and previous.background:
            cur = self._shot_error(frame, previous, shot)
            if cur > SHOT_TRIGGER_ERROR:
                trials = {None}
                est = self._estimate_shot(frame, previous.background)
                if est is not None:
                    z0, x0, y0 = est
                    for dz in (-0.5, 0.0, 0.5):
                        for dx in np.arange(-0.5, 0.51, 0.25):
                            for dy in np.arange(-0.5, 0.51, 0.25):
                                trials.add((float(z0 + dz), float(x0 + dx), float(y0 + dy)))
                if shot is not None:
                    # 같은 배율에서 옮겨 비추기(pan) — 거칠게(1u) 훑고 가장 나은 곳 둘레를 0.25u로.
                    z0, x0, y0 = shot
                    coarse = [(z0, float(x0 + dx), float(y0 + dy)) for dx in range(-6, 7) for dy in (-1, 0, 1)]
                    top = min(coarse, key=lambda t: self._shot_error(frame, previous, t))
                    for dx in np.arange(-0.75, 0.76, 0.25):
                        for dy in np.arange(-0.75, 0.76, 0.25):
                            trials.add((z0, float(top[1] + dx), float(top[2] + dy)))
                trials.discard(shot)
                scored = sorted(((self._shot_error(frame, previous, t), t) for t in trials),
                                key=lambda x: (x[0], x[1] is not None))
                if scored and scored[0][0] < cur * SHOT_IMPROVE:
                    shot = scored[0][1]
                    if shot is not None and abs(shot[0]) < 1e-9 and shot[1] == 0 and shot[2] == 0:
                        shot = None
                    self.cands = self.candidates(shot)
                    bg, bg_score, base = self._background(frame)
        fits: List[_Fit] = []

        # 앞 줄 인물 중 그대로 맞는 것은 둔다.
        prev_emotion: Dict[str, str] = {}
        if previous is not None:
            prev_visual = previous.visual if self.fx is not None else {}
            for s in previous.slots:
                prev_emotion[s.character] = s.emotion
                pl = self.cands.find(s.character, s.variant, s.place, s.focus, s.depth, s.extras)
                entry = self.catalog.portrait(s.character, s.variant, s.emotion)
                if pl is None or entry is None:
                    continue
                fit = self._make_fit(s.character, s.variant, entry, pl, s.mirror, s.move)
                if s.character in prev_visual:
                    fit = self._with_visual(fit, prev_visual[s.character])
                fits.append(fit)
            tried: set = set()
            changed = True
            while changed and fits:
                changed = False
                model = self._compose(base, fits)
                err = robust_error(frame, model)
                for i, f in enumerate(fits):
                    vis = (f.alpha > 0.95) & (self.mask > 0)
                    for g in fits[i + 1:]:
                        vis &= g.alpha < 0.05
                    if vis.sum() < 50:
                        continue
                    e_i = float(err[vis].mean())
                    if self.fx is not None and f.character not in tried and e_i > KEEP_ERROR * VISUAL_TRY:
                        # 자리는 그대로인데 색만 바뀌었나(캐릭터 비주얼) — 비주얼을 바꿔 입혀 본다.
                        tried.add(f.character)
                        hit = self._best_visual(frame, base, fits, i)
                        if hit is not None and hit[2].visual != f.visual and hit[1] < hit[0] * VISUAL_GAIN \
                                and hit[0] - hit[1] > VISUAL_MIN_DROP:
                            fits[i] = hit[2]
                            changed = True
                            break
                    if e_i > KEEP_ERROR * 1.6:
                        lost = fits.pop(i)
                        changed = True
                        # 같은 자리 어휘에서 옮겨졌을 뿐인가(move_by) — 넓게 따라가 본다.
                        tracked = self._track(frame, base, fits, lost)
                        if tracked is not None:
                            fits.insert(tracked[1], tracked[0])
                        break

        # 남은 오차를 인물로 설명한다(탐욕). 설명 못 한 화면이 거의 없으면 찾지 않는다.
        while True:
            model = self._compose(base, fits)
            err = robust_error(frame, model)
            if float((_unexplained(err) * self.mask).sum()) < UNEXPLAINED_SKIP * float(self.mask.sum()):
                break
            used = {f.character for f in fits}
            scored = []
            for (c, v) in self.cands.identities:
                if c in used:
                    continue
                entry = self.cands.emotions(c, v)[0]
                for pl in self.cands.placements(c, v):
                    for mirror in (False, True):
                        cand = _Fit(c, v, entry, pl, mirror, (0.0, 0.0), None, None)  # type: ignore[arg-type]
                        pos = self._order_index(fits, cand)[-1]
                        # 순위는 어긋남을 조금 허용해 매긴다 — move_by로 밀린 인물도 후보에서 빠지지 않게.
                        scored.append((self._rank(frame, base, fits, cand, pos), cand))
            if not scored:
                break
            scored.sort(key=lambda s: -s[0])
            best = None
            for gain0, cand in scored[:TOP_K]:
                cand = self._make_fit(cand.character, cand.variant, cand.entry, cand.placement, cand.mirror)
                for pos in self._order_index(fits, cand):
                    fit = self._refine(frame, base, fits, cand, pos)
                    gain, covered = self._gain(frame, base, fits, fit, pos, err)
                    if best is None or gain > best[0]:
                        best = (gain, covered, fit, pos)
            gain, covered, fit, pos = best
            if covered == 0 or gain < MIN_GAIN_PER_PIXEL * covered:
                break
            fits.insert(pos, fit)

        # 표정
        sure_by: Dict[int, bool] = {}
        for i in range(len(fits)):
            fits[i], sure_by[i] = self._choose_emotion(frame, base, fits, i, prev_emotion.get(fits[i].character))

        model = self._compose(base, fits)
        err = robust_error(frame, model)
        w = self.mask > 0
        slots = []
        for i, f in enumerate(fits):
            vis = (f.alpha > 0.95) & w
            for g in fits[i + 1:]:
                vis &= g.alpha < 0.05
            pl = f.placement
            slots.append(ObservedSlot(
                f.character, f.variant, f.entry.emotion, pl.place, pl.focus, pl.depth, f.mirror, f.move,
                float(err[vis].mean()) if vis.any() else float("nan"), int(vis.sum()), sure_by[i],
                [[round(float(x), 2) for x in p] for p in f.quad], tuple(pl.extras)))
            if vis.sum() < 200:
                notes.append(f"{f.character}: 거의 가려져 있습니다")
        residual = float(_unexplained(err)[w].mean())
        if residual > 0.01:
            notes.append(f"설명 못 한 화면 {residual * 100:.1f}%")
        self._last_model = model
        effects: Dict = {}
        if self.fx is not None:
            effects = {"vignette": [vignette[0], vignette[1]] if vignette else None,
                       "visual": {f.character: [f.visual[0], f.visual[1]] for f in fits if f.visual}}
        return ObservedStage(time, bg, bg_score, slots, float(err[w].mean()), residual, notes, shot, effects)


def _unexplained(err: np.ndarray, threshold: float = 25.0) -> np.ndarray:
    """설명 못 한 화면 — 오차가 큰 픽셀에서 가는 선(압축 · 경계 번짐)을 열림 연산으로 걷어 낸 덩어리."""
    m = (err > threshold).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(np.float32)


def _bbox_slice(alpha: np.ndarray):
    ys, xs = np.nonzero(alpha > 0.02)
    if len(xs) == 0:
        return None
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)


def _crop_fit(f: _Fit, sl) -> _Fit:
    return _Fit(f.character, f.variant, f.entry, f.placement, f.mirror, f.move, f.color[sl], f.alpha[sl])


def _ecc_shift(template: np.ndarray, image: np.ndarray, mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """template을 얼마나 옮기면 image와 맞나(분석 px). 거친 배율부터 세 단계."""
    tg = cv2.cvtColor(np.clip(template, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    ig = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
    for level in (4, 2, 1):
        t = cv2.resize(tg, None, fx=1 / level, fy=1 / level, interpolation=cv2.INTER_AREA) if level > 1 else tg
        im = cv2.resize(ig, None, fx=1 / level, fy=1 / level, interpolation=cv2.INTER_AREA) if level > 1 else ig
        m = cv2.resize(mask, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_NEAREST) if level > 1 else mask
        if m.sum() < 30:
            continue
        w = warp.copy()
        w[:, 2] /= level
        try:
            _, w = cv2.findTransformECC(t, im, w, cv2.MOTION_TRANSLATION, criteria, m.astype(np.uint8), 3)
        except cv2.error:
            return None
        warp = w.copy()
        warp[:, 2] *= level
    return float(warp[0, 2]), float(warp[1, 2])
