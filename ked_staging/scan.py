"""다른 게임(P4) — 영상 → 카탈로그 초안(draft.json, 스키마 ked-staging-draft/1).

자료(초상 · 배경 원본, 치수표)가 없는 게임 영상에서 "무엇이 화면에 섰나"를 화면으로 배운다. 이름은 임시
ID(c01 · 01 · bg01)이고, 사람이 카탈로그 툴(컴포지터 Vn.CatalogTool) 또는 `draft` 명령으로 고친 뒤
`catalog-build`가 카탈로그 + 튜닝 겹본을 만든다(catalogdraft).

  1. 줄 경계      그 게임의 게임텍스트추출기 프로필(대사창 자리)로 P1과 같이 잉크를 따라 잡는다.
  2. 정착 화면    줄마다 끝 직전 몇 장의 평균. 대사창 위(stageBottom 위)만 무대로 본다.
  3. 배경 판      무대가 대부분 같은 줄끼리 묶는다. 장면(이어진 줄)의 첫 · 끝 줄을 씨앗으로 중앙값을 내고,
                  판과 다른 자리(인물)를 가린 중앙값으로 두 번 다시 깨끗이 한다.
  4. 인물 조각    판과 다른 덩어리를 RGBA로 자른다. 두 인물이 겹쳐 한 덩어리가 되면, 다른 줄의 온전한 단독
                  조각이 그 안에 통째로 들어맞는지 보고 앞 인물과 나머지(가린 뒤 인물)로 나눈다.
  5. 정체         조각끼리 겹친 자리의 차로 맞댄 뒤(마스크 상관), 몸 전체가 거의 같으면 같은 인물이다.
                  겹친 자리만 견주므로 화면 끝에 잘린 조각도 맞는다. 같은 줄의 두 조각은 같은 인물이 아니다.
  6. 표정         같은 인물의 얼굴 쪽(보이는 몸 위 60%, 윤곽 안쪽)을 영상 화소 그대로 견줘 크게 다른 화소가
                  덩어리로 있으면 다른 표정이다.

조각 맞대기(align · refine)는 catalog-build도 쓴다 — 사람이 고친 뒤 무리를 다시 맞대어 몸을 채운다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .imageio import read_image, write_image

SCHEMA = "ked-staging-draft/1"

WORK_SCALE = 0.5              # 정체 견주기 배율
COARSE_SCALE = 0.25           # 맞대기 첫 어림 배율
HEAD_PART = 0.6               # 보이는 몸 위쪽 이만큼이 얼굴(표정이 바뀌는 자리 — 입까지 든다)
STRONG = 40.0                 # 조각 대조에서 '크게 다른' 화소의 차(채널 최대)
IDENTITY_FRACTION = 0.08      # 겹친 자리에서 크게 다른 화소가 이 비율 아래면 같은 인물
EXPR_PIXELS = 45.0            # 얼굴 쪽(윤곽 안쪽)에서 크게 다른 화소가 이만큼(720p 화소 수) 넘으면 다른 표정
EXPR_INSET = 8                # 표정은 윤곽에서 이만큼(720p 화소) 안쪽만 견준다
MIN_OVERLAP = 0.5             # 맞댈 때 작은 조각의 이만큼은 겹쳐야 한다
ALIGN_REFINE = 2              # 영상 화소에서 다듬는 범위(±px)
DIFF_THRESHOLD = 24.0         # 판과 이만큼(채널 최대) 다르면 인물 화소
MIN_PIECE = 0.003             # 무대 넓이 대비 조각 최소 넓이
SAME_BACKGROUND = 0.15        # 무대에서 어둡지 않은 화소의 이만큼이 같으면 같은 배경(인물이 무대 대부분을 덮어도)
PIXEL_SAME = 14.0             # 배경 묶기에서 '같은 화소'의 차
CONTAIN = 0.9                 # 단독 조각의 이만큼이 덩어리에 들어맞으면 그 덩어리 안의 앞 인물
SPLIT_REST = 0.15             # 들어맞은 나머지가 덩어리의 이만큼은 돼야 나눈다
SHAPE_PENALTY = 50.0 ** 2     # 맞대기에서 모양이 어긋난 화소 하나의 값(회색조 차의 제곱)
WHOLE_FRAME = 0.6             # 판과 다른 덩어리가 무대의 이만큼을 넘으면 화면 전체가 바뀐 것(카메라 · 전환)
FRONT_MIN = 0.25              # 덩어리 안 앞 인물로 볼 단독 조각은 덩어리 넓이의 이만큼은 돼야 한다
FRONT_HEIGHT = 0.6            # …그리고 덩어리 높이의 이만큼은 돼야 한다(손 · 무기 · 작은 조각이 앞 인물이 되지 않게)


# ── 조각 ──────────────────────────────────────────────────────

@dataclass
class Piece:
    """잘린 인물 조각 하나 — 색(BGR float32)과 마스크(bool). 배율마다 줄인 판을 캐시한다."""
    color: np.ndarray
    mask: np.ndarray
    _levels: Dict[float, "Piece"] = field(default_factory=dict, repr=False)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def gray(self) -> np.ndarray:
        c = self.color
        return (0.114 * c[:, :, 0] + 0.587 * c[:, :, 1] + 0.299 * c[:, :, 2]).astype(np.float32)

    def at(self, scale: float) -> "Piece":
        if scale == 1.0:
            return self
        if scale not in self._levels:
            h, w = self.mask.shape
            size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            m = cv2.resize(self.mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
            c = cv2.resize(self.color * self.mask[:, :, None], size, interpolation=cv2.INTER_AREA)
            keep = m > 0.75
            c = c / np.maximum(m, 1e-6)[:, :, None]
            self._levels[scale] = Piece(c.astype(np.float32), keep)
        return self._levels[scale]

    @classmethod
    def from_rgba(cls, rgba: np.ndarray) -> "Piece":
        return cls(rgba[:, :, :3].astype(np.float32), rgba[:, :, 3] > 127)

    def rgba(self) -> np.ndarray:
        out = np.zeros(self.mask.shape + (4,), np.uint8)
        out[:, :, :3] = np.clip(self.color + 0.5, 0, 255).astype(np.uint8)
        out[:, :, 3] = self.mask.astype(np.uint8) * 255
        return out


def load_piece(path: str) -> Piece:
    img = read_image(path, alpha=True)
    if img is None:
        raise FileNotFoundError(path)
    return Piece.from_rgba(img)


def _overlap(a: Piece, b: Piece, ox: int, oy: int):
    """b를 a 좌표 (ox, oy)에 놓았을 때 둘 다 불투명한 자리 → (a 색, b 색, 마스크, a 쪽 자르기). 없으면 None."""
    ha, wa = a.mask.shape
    hb, wb = b.mask.shape
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(wa, ox + wb), min(ha, oy + hb)
    if x1 <= x0 or y1 <= y0:
        return None
    ca = a.color[y0:y1, x0:x1]
    cb = b.color[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    m = a.mask[y0:y1, x0:x1] & b.mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return ca, cb, m, (slice(y0, y1), slice(x0, x1))


def _xcorr(a: np.ndarray, b: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """out[dy + hb - 1, dx + wb - 1] = Σ a[y, x] · b[y - dy, x - dx] (b를 a 좌표 (dx, dy)에 놓은 상관)."""
    fa = np.fft.rfft2(a, shape)
    fb = np.fft.rfft2(b[::-1, ::-1], shape)
    return np.fft.irfft2(fa * fb, shape)


def _coarse_align(a: Piece, b: Piece, min_overlap: float) -> Optional[Tuple[int, int, float]]:
    """겹친 자리의 평균 제곱차가 가장 작은 자리 — 마스크 상관 넷(FFT)으로 모든 자리를 한 번에 잰다."""
    ga, gb = a.gray, b.gray
    ma, mb = a.mask.astype(np.float32), b.mask.astype(np.float32)
    ha, wa = ma.shape
    hb, wb = mb.shape
    shape = (ha + hb - 1, wa + wb - 1)
    n = _xcorr(ma, mb, shape)
    saa = _xcorr(ma * ga * ga, mb, shape)
    sbb = _xcorr(ma, mb * gb * gb, shape)
    sab = _xcorr(ma * ga, mb * gb, shape)
    # 모양이 어긋난 화소(한쪽만 불투명 — 두 조각이 다 보이는 자리 안에서)는 크게 다른 화소로 센다.
    # 무늬가 되풀이되는 옷 · 화면 끝에 잘린 조각이 엉뚱한 자리에 미끄러지지 않게 한다.
    miss = _xcorr(1.0 - ma, mb, shape) + _xcorr(ma, 1.0 - mb, shape)
    miss = np.maximum(miss, 0.0)
    need = max(20.0, min_overlap * min(ma.sum(), mb.sum()))
    ssd = np.where(n >= need, (saa + sbb - 2 * sab + SHAPE_PENALTY * miss) / np.maximum(n + miss, 1.0), np.inf)
    k = int(np.argmin(ssd))
    if not np.isfinite(ssd.flat[k]):
        return None
    ky, kx = np.unravel_index(k, ssd.shape)
    return int(kx - (wb - 1)), int(ky - (hb - 1)), float(ssd.flat[k])


def refine(a: Piece, b: Piece, ox: int, oy: int, reach: int = ALIGN_REFINE + 1) -> Tuple[int, int]:
    """어림한 맞대기 자리 둘레(±reach)에서 겹친 화소 차가 가장 작은 곳."""
    best = None
    need = max(50, int(0.3 * min(a.area, b.area)))
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            ov = _overlap(a, b, ox + dx, oy + dy)
            if ov is None or ov[2].sum() < need:
                continue
            e = float(np.abs(ov[0] - ov[1]).max(2)[ov[2]].mean())
            if best is None or e < best[0]:
                best = (e, ox + dx, oy + dy)
    return (best[1], best[2]) if best else (ox, oy)


def align(a: Piece, b: Piece, scale: float = 1.0, min_overlap: float = MIN_OVERLAP) -> Optional[Tuple[int, int]]:
    """b의 왼쪽 위가 a 좌표 어디에 오면 가장 잘 겹치나 → scale 배율 화소의 (x, y). 맞댈 수 없으면 None.

    COARSE_SCALE에서 모든 자리를 한 번에 잰 뒤, 배율을 올리며 ±2px씩 다듬는다."""
    coarse = min(COARSE_SCALE, scale)
    found = _coarse_align(a.at(coarse), b.at(coarse), min_overlap)
    if found is None:
        return None
    ox, oy = found[0], found[1]
    s = coarse
    for nxt in (WORK_SCALE, 1.0):
        if nxt <= s or nxt > scale:
            continue
        f = nxt / s
        ox, oy = refine(a.at(nxt), b.at(nxt), int(round(ox * f)), int(round(oy * f)), int(np.ceil(f)) + 1)
        s = nxt
    if s != scale:
        f = scale / s
        ox, oy = refine(a.at(scale), b.at(scale), int(round(ox * f)), int(round(oy * f)), int(np.ceil(f)) + 1)
    return ox, oy


def _eroded(p: Piece) -> np.ndarray:
    """테두리 한 겹을 걷은 마스크 — 테두리 화소는 그 줄 배경이 섞여 있다(배경이 다르면 크게 다르다)."""
    return cv2.erode(p.mask.astype(np.uint8), np.ones((3, 3), np.uint8), borderValue=0).astype(bool)


def _slack_diff(a: Piece, b: Piece, ox: int, oy: int, slack: int = 1):
    """a의 각 화소를 b의 둘레(±slack) 화소 가운데 가장 가까운 것과 견준 차(채널 최대) — 같은 그림이 다른 반 화소
    자리에 다시 표본 내져(place 간격 460.8px) 윤곽이 갈라지는 것을 크게 다르다고 보지 않는다.
    → (차, 견줄 마스크, a 쪽 자르기). 겹치지 않으면 None."""
    ov = _overlap(a, b, ox, oy)
    if ov is None:
        return None
    ca, _cb, _m, (sy, sx) = ov
    ea, eb = _eroded(a), _eroded(b)
    hb, wb = b.mask.shape
    best = np.full(ca.shape[:2], np.inf, np.float32)
    ys = np.arange(sy.start, sy.stop)
    xs = np.arange(sx.start, sx.stop)
    for dy in range(-slack, slack + 1):
        for dx in range(-slack, slack + 1):
            by, bx = ys - oy + dy, xs - ox + dx
            vy, vx = (by >= 0) & (by < hb), (bx >= 0) & (bx < wb)
            if not vy.any() or not vx.any():
                continue
            yy, xx = np.ix_(np.clip(by, 0, hb - 1), np.clip(bx, 0, wb - 1))
            ok = vy[:, None] & vx[None, :] & b.mask[yy, xx]
            d = np.abs(ca - b.color[yy, xx]).max(2)
            best = np.where(ok, np.minimum(best, d), best)
    yy, xx = np.ix_(np.clip(ys - oy, 0, hb - 1), np.clip(xs - ox, 0, wb - 1))
    inside = ((ys - oy >= 0) & (ys - oy < hb))[:, None] & ((xs - ox >= 0) & (xs - ox < wb))[None, :]
    m = ea[sy, sx] & inside & eb[yy, xx] & np.isfinite(best)
    return best, m, (sy, sx)


def strong_fraction(a: Piece, b: Piece, ox: int, oy: int, rows: Optional[Tuple[int, int]] = None,
                    min_overlap: float = MIN_OVERLAP) -> Optional[float]:
    """겹친 자리(rows를 주면 a 좌표 그 줄 범위만)에서 크게 다른 화소의 비율(±1px 너그럽게, 테두리 빼고).
    겹침이 모자라면 None."""
    ov = _slack_diff(a, b, ox, oy)
    if ov is None:
        return None
    diff, m, (sy, _sx) = ov
    if rows is not None:
        ys = np.arange(sy.start, sy.stop)
        m = m & ((ys >= rows[0]) & (ys < rows[1]))[:, None]
        need = 30
    else:
        need = max(30, int(0.8 * min_overlap * min(a.area, b.area)))
    if m.sum() < need:
        return None
    return float((diff[m] > STRONG).mean())


def identity_distance(a: Piece, b: Piece, offset: Tuple[int, int]) -> float:
    """몸 전체(겹친 자리)에서 크게 다른 화소의 비율 — 맞댈 수 없으면 1."""
    d = strong_fraction(a, b, offset[0], offset[1])
    return 1.0 if d is None else d


def _interior(p: Piece, inset: int) -> np.ndarray:
    dt = cv2.distanceTransform(np.pad(p.mask.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    return dt >= inset


def expression_distance(a: Piece, b: Piece, offset: Tuple[int, int], head_rows: Tuple[int, int],
                        inset: int = EXPR_INSET) -> float:
    """얼굴 쪽(a 좌표 head_rows)에서 크게 다른 화소 수 — 겹치지 않으면 0(가리지 못한다).

    비율이 아니라 수로 센다: 표정 표시(눈썹 · 입)의 넓이는 얼굴 크기로 정해지고, 얼굴 쪽 넓이로 나누면 가린
    조각 · 큰 조각에서 묽어진다.

    윤곽에서 inset px 안쪽만 본다: 머리카락 끝처럼 반투명한 윤곽은 그 줄 배경이 비쳐 배경이 바뀌면 크게
    달라진다. 표정(눈썹 · 입 · 볼)은 얼굴 안쪽의 덩어리진 표시이므로 3px 열림으로 가는 선을 거른다 — 얼굴을
    가로지르는 머리카락 · 옷 윤곽선은 다른 반 화소 자리에서 굵기가 달라져 크게 다르다고 나온다."""
    ov = _slack_diff(a, b, offset[0], offset[1])
    if ov is None:
        return 0.0
    diff, m, (sy, sx) = ov
    hb, wb = b.mask.shape
    ys = np.arange(sy.start, sy.stop)
    xs = np.arange(sx.start, sx.stop)
    inner_b = _interior(b, inset)[np.ix_(np.clip(ys - offset[1], 0, hb - 1), np.clip(xs - offset[0], 0, wb - 1))]
    m = m & _interior(a, inset)[sy, sx] & inner_b & ((ys >= head_rows[0]) & (ys < head_rows[1]))[:, None]
    if m.sum() < 30:
        return 0.0
    hit = cv2.morphologyEx(((diff > STRONG) & m).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(hit.sum())


def cluster(items: Sequence[int], dist: Callable[[int, int], float], threshold: float,
            complete: bool = False, apart: Optional[Callable[[int, int], bool]] = None) -> List[List[int]]:
    """가까운 것부터 묶는다. complete면 무리 안 모든 쌍이 threshold 아래여야 한다(표정), 아니면 한 쌍이면 된다(정체).
    apart(i, j)가 참인 둘은 같은 무리에 두지 않는다(같은 줄의 두 조각)."""
    items = list(items)
    d = {}
    for x in range(len(items)):
        for y in range(x + 1, len(items)):
            d[(items[x], items[y])] = d[(items[y], items[x])] = dist(items[x], items[y])
    groups = [[i] for i in items]

    def gap(g: List[int], h: List[int]) -> float:
        vals = [d[(i, j)] for i in g for j in h]
        return max(vals) if complete else min(vals)

    while len(groups) > 1:
        best = None
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                if apart and any(apart(i, j) for i in groups[x] for j in groups[y]):
                    continue
                g = gap(groups[x], groups[y])
                if g < threshold and (best is None or g < best[0]):
                    best = (g, x, y)
        if best is None:
            break
        _, x, y = best
        groups[x] = groups[x] + groups[y]
        del groups[y]
    return groups


# ── 초안 ──────────────────────────────────────────────────────

@dataclass
class Observation:
    id: str
    line: int
    time: float
    bbox: Tuple[int, int, int, int]      # 영상 화소 (x, y, w, h)
    png: str
    area: int
    edge: bool                           # 화면 끝에 잘렸다
    background: Optional[str]
    character: str = "c01"
    variant: str = "a"
    emotion: str = "01"
    excluded: bool = False
    representative: bool = False
    split: bool = False                  # 겹친 덩어리에서 나눈 뒤 인물(가린 자리가 있다)

    def to_json(self) -> dict:
        return {"id": self.id, "line": self.line, "time": round(self.time, 3), "bbox": list(self.bbox),
                "png": self.png, "area": self.area, "edge": self.edge, "background": self.background,
                "character": self.character, "variant": self.variant, "emotion": self.emotion,
                "excluded": self.excluded, "representative": self.representative, "split": self.split}


@dataclass
class Draft:
    video: str
    video_size: Tuple[int, int]
    stage_bottom: int
    box_profile: Optional[str]
    lines: List[dict] = field(default_factory=list)
    backgrounds: List[dict] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "video": self.video, "videoSize": list(self.video_size),
                "stageBottom": self.stage_bottom, "boxProfile": self.box_profile,
                "lines": self.lines, "backgrounds": self.backgrounds,
                "observations": [o.to_json() for o in self.observations]}

    def save(self, folder: str) -> str:
        path = os.path.join(folder, "draft.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=1)
            f.write("\n")
        os.replace(tmp, path)
        return path


# ── 화면 ──────────────────────────────────────────────────────

def _small(frame: np.ndarray, bottom: int, width: int = 160) -> np.ndarray:
    stage = frame[:bottom]
    h = max(1, int(round(width * stage.shape[0] / stage.shape[1])))
    return cv2.resize(stage, (width, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def _same_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """같은 화소의 비율 — 둘 다 거의 검은 화소는 세지 않는다(어두운 전환 · 메뉴 · 어두운 장면이 다리가 되지 않게)."""
    informative = np.maximum(a.mean(2), b.mean(2)) > 28.0
    n = int(informative.sum())
    if n == 0:
        return 0.0
    return float(((np.abs(a - b).max(2) < PIXEL_SAME) & informative).sum() / n)


def _is_blank(small: np.ndarray) -> bool:
    return float(small.mean()) < 8.0 and float(small.std()) < 6.0


def group_backgrounds(smalls: Sequence[np.ndarray]) -> List[Optional[int]]:
    """줄마다 배경 무리 번호(빈 화면은 None) — 무대 화소 절반 이상이 같은 줄끼리 잇는다."""
    n = len(smalls)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    blank = [_is_blank(s) for s in smalls]
    for i in range(n):
        for j in range(i + 1, n):
            if blank[i] or blank[j]:
                continue
            if _same_fraction(smalls[i], smalls[j]) >= SAME_BACKGROUND:
                parent[find(j)] = find(i)
    ids: Dict[int, int] = {}
    out: List[Optional[int]] = []
    for i in range(n):
        if blank[i]:
            out.append(None)
            continue
        r = find(i)
        ids.setdefault(r, len(ids))
        out.append(ids[r])
    return out


def _diff(frame: np.ndarray, plate: np.ndarray) -> np.ndarray:
    a = cv2.GaussianBlur(frame, (3, 3), 0)
    b = cv2.GaussianBlur(plate, (3, 3), 0)
    return np.abs(a - b).max(2)


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def build_plate(frames: Sequence[np.ndarray], seeds: Sequence[int], bottom: int, unit: float,
                strip: int = 96) -> np.ndarray:
    """씨앗 줄의 중앙값 → 판과 다른 자리를 가린 중앙값(두 번). 가린 자리만 남은 화소는 앞 판 값을 둔다.
    중앙값은 가로 띠마다 낸다 — 1080p 줄 수십 장을 한꺼번에 쌓지 않는다."""
    import warnings
    h = frames[0].shape[0]
    plate = np.empty_like(frames[0])
    for y0 in range(0, h, strip):
        plate[y0:y0 + strip] = np.median(np.stack([frames[i][y0:y0 + strip] for i in seeds]), axis=0)
    grow = _kernel(9 * unit)
    for _ in range(2):
        masks = []
        for f in frames:
            m = np.zeros(f.shape[:2], np.uint8)
            m[:bottom] = _diff(f[:bottom], plate[:bottom]) > DIFF_THRESHOLD
            masks.append(cv2.dilate(m, grow).astype(bool))
        new = plate.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for y0 in range(0, h, strip):
                stack = np.stack([np.where(m[y0:y0 + strip, :, None], np.nan, f[y0:y0 + strip])
                                  for f, m in zip(frames, masks)])
                med = np.nanmedian(stack, axis=0)
                new[y0:y0 + strip] = np.where(np.isnan(med), plate[y0:y0 + strip], med)
        plate = new
    return plate


def find_pieces(frame: np.ndarray, plate: np.ndarray, bottom: int, unit: float
                ) -> List[Tuple[Tuple[int, int, int, int], np.ndarray, bool]]:
    """판과 다른 덩어리 → [(bbox, 마스크(bbox 크기), 화면 끝에 닿았나)]."""
    h, w = frame.shape[:2]
    m = np.zeros((h, w), np.uint8)
    m[:bottom] = (_diff(frame[:bottom], plate[:bottom]) > DIFF_THRESHOLD).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _kernel(3 * unit))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _kernel(7 * unit))
    m[bottom:] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = MIN_PIECE * w * bottom
    out = []
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area < min_area:
            continue
        if area > WHOLE_FRAME * w * bottom or (bw > 0.9 * w and bh < 0.3 * bottom):
            continue    # 화면 전체가 바뀜(카메라 이동 · 전환 · 레터박스 띠) — 인물 조각이 아니다
        comp = (labels[y:y + bh, x:x + bw] == k).astype(np.uint8)
        # 덩어리 안의 작은 구멍(배경과 색이 비슷한 자리)을 메운다 — 큰 구멍(팔과 몸 사이)은 둔다.
        filled = comp.copy()
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        holes = filled & (1 - comp)
        hn, hl, hs, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
        for j in range(1, hn):
            if hs[j, 4] < 0.02 * area:
                comp[hl == j] = 1
        edge = x <= 0 or y <= 0 or x + bw >= w
        out.append(((int(x), int(y), int(bw), int(bh)), comp.astype(bool), bool(edge)))
    return out


# ── 겹친 덩어리 나누기 ─────────────────────────────────────────

def _contained(c: Piece, r: Piece) -> Optional[Tuple[int, int, int]]:
    """단독 조각 r이 덩어리 c 안에 통째로 들어맞나 → (x, y, 맞은 화소 수). 아니면 None."""
    if r.area >= 0.9 * c.area or r.area < FRONT_MIN * c.area or r.mask.shape[0] < FRONT_HEIGHT * c.mask.shape[0]:
        return None
    found = align(c, r, scale=WORK_SCALE, min_overlap=0.8)
    if found is None:
        return None
    rc, cw = r.at(WORK_SCALE), c.at(WORK_SCALE)
    ov = _overlap(cw, rc, found[0], found[1])
    if ov is None:
        return None
    ca, cb, m, _ = ov
    matched = int((m & (np.abs(ca - cb).max(2) < STRONG)).sum())
    if matched < CONTAIN * rc.area:
        return None
    ox, oy = refine(c, r, int(round(found[0] / WORK_SCALE)), int(round(found[1] / WORK_SCALE)), 3)
    return ox, oy, matched


def split_blob(c: Piece, r: Piece, at: Tuple[int, int], unit: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """c를 r이 덮는 앞 인물과 나머지로 → (앞 마스크, 뒤 마스크) — c 크기. 나머지가 작으면 None."""
    h, w = c.mask.shape
    front = np.zeros((h, w), bool)
    ox, oy = at
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(w, ox + r.mask.shape[1]), min(h, oy + r.mask.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    front[y0:y1, x0:x1] = r.mask[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    front = cv2.dilate(front.astype(np.uint8), _kernel(3 * unit)).astype(bool) & c.mask
    rest = (c.mask & ~front).astype(np.uint8)
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, _kernel(5 * unit))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, 4]))
    back = labels == k
    if back.sum() < SPLIT_REST * c.area:
        return None
    return front, back


# ── 훑기 ──────────────────────────────────────────────────────

def scan(video: str, layout, out_dir: str, box_profile: Optional[str] = None,
         log: Callable[[str], None] = print, pieces: bool = True) -> Draft:
    """영상 → 초안 폴더(draft.json · frames/ · bg/ · obs/). layout: 영상 크기의 BoxLayout(대사창 자리).
    pieces=False면 줄 · 정착 화면 · 배경 판까지만 낸다 — 원본 스프라이트가 있을 때(catalog-originals)는 인물을 배우지 않는다."""
    from .lines import detect_lines, track_ink
    from .pipeline import grab_frames, settled_times

    os.makedirs(out_dir, exist_ok=True)
    for sub in ("frames", "bg", "obs"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"영상을 열지 못했습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if (layout.width, layout.height) != (vw, vh):
        layout = layout.scaled(vw, vh)
    bottom = int(layout.stage_bottom())
    unit = vh / 720.0

    spans = detect_lines(track_ink(video, layout))
    if not spans:
        raise ValueError("줄을 하나도 찾지 못했습니다 — 대사창 프로필(--box-profile)을 확인하세요")
    wanted = [settled_times(s, fps) for s in spans]
    frames = [f.astype(np.float32) for f in grab_frames(video, wanted)]
    log(f"줄 {len(spans)}개 · 영상 {vw}×{vh} · 무대 아래 끝 {bottom}px")

    draft = Draft(os.path.abspath(video), (vw, vh), bottom, os.path.abspath(box_profile) if box_profile else None)
    for i, (s, f) in enumerate(zip(spans, frames)):
        rel = os.path.join("frames", f"l{i:04d}.jpg")
        write_image(os.path.join(out_dir, rel), np.clip(f + 0.5, 0, 255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        draft.lines.append({"index": i, "head": round(s.head, 3), "end": round(s.end, 3),
                            "time": round(float(np.mean(wanted[i])), 3), "background": None, "frame": rel})

    # 배경
    groups = group_backgrounds([_small(f, bottom) for f in frames])
    plates: Dict[int, np.ndarray] = {}
    keys: Dict[int, str] = {}
    for g in sorted({x for x in groups if x is not None}, key=lambda x: groups.index(x)):
        members = [i for i, x in enumerate(groups) if x == g]
        seeds = sorted({m for k, m in enumerate(members)
                        if k == 0 or k == len(members) - 1 or members[k - 1] != m - 1 or
                        (k + 1 < len(members) and members[k + 1] != m + 1)})
        plate = build_plate([frames[i] for i in members], [members.index(s) for s in seeds], bottom, unit)
        key = f"bg{len(keys) + 1:02d}"
        keys[g], plates[g] = key, plate
        rel = os.path.join("bg", f"{key}.png")
        write_image(os.path.join(out_dir, rel), np.clip(plate + 0.5, 0, 255).astype(np.uint8))
        draft.backgrounds.append({"key": key, "plate": rel, "lines": members, "excluded": False})
        for i in members:
            draft.lines[i]["background"] = key
    log(f"배경 판 {len(keys)}개")

    if not pieces:
        path = draft.save(out_dir)
        log(f"줄 · 배경 판만 냈습니다(인물 조각 없음) → {path}")
        return draft

    # 인물 조각
    pieces: List[Piece] = []
    records: List[dict] = []
    for i, f in enumerate(frames):
        g = groups[i]
        if g is None:
            continue
        for (x, y, w, h), m, edge in find_pieces(f, plates[g], bottom, unit):
            pieces.append(Piece(f[y:y + h, x:x + w].copy(), m))
            records.append({"line": i, "bbox": (x, y, w, h), "edge": edge, "split": False, "front": False})

    # 겹친 덩어리 나누기 — 다른 줄의 온전한(끝에 안 닿은) 단독 조각이 통째로 들어맞으면 앞 인물이다.
    k = 0
    while k < len(pieces):
        if records[k]["front"]:
            k += 1          # 나눠 낸 앞 인물은 다시 나누지 않는다(통째로 들어맞은 조각이다)
            continue
        best = None
        for j, r in enumerate(pieces):
            if j == k or records[j]["line"] == records[k]["line"] or records[j]["edge"] or records[j]["split"]:
                continue
            hit = _contained(pieces[k], r)
            if hit and (best is None or hit[2] > best[1][2]):
                best = (j, hit)
        if best is not None:
            parts = split_blob(pieces[k], pieces[best[0]], best[1][:2], unit)
            if parts is not None:
                c, rec = pieces[k], records[k]
                x, y = rec["bbox"][:2]
                new = []
                for part, is_back in ((parts[0], False), (parts[1], True)):
                    ys, xs = np.nonzero(part)
                    px0, py0, px1, py1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                    sub = part[py0:py1, px0:px1]
                    bbox = (int(x + px0), int(y + py0), int(px1 - px0), int(py1 - py0))
                    edge = bbox[0] <= 0 or bbox[1] <= 0 or bbox[0] + bbox[2] >= vw
                    new.append((Piece(c.color[py0:py1, px0:px1].copy(), sub),
                                {"line": rec["line"], "bbox": bbox, "edge": bool(edge), "split": is_back,
                                 "front": not is_back}))
                pieces[k:k + 1] = [p for p, _ in new]
                records[k:k + 1] = [r for _, r in new]
                log(f"  줄 {rec['line']}: 겹친 덩어리를 둘로 나눔")
                continue    # 앞 인물 조각부터 다시 본다(셋이 겹쳤을 수 있다)
        k += 1
    log(f"인물 조각 {len(pieces)}개")

    # 정체
    n = len(pieces)
    offsets: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}

    def offset(a: int, b: int) -> Optional[Tuple[int, int]]:
        if (a, b) not in offsets:
            offsets[(a, b)] = align(pieces[a], pieces[b], scale=WORK_SCALE)
            o = offsets[(a, b)]
            offsets[(b, a)] = None if o is None else (-o[0], -o[1])
        return offsets[(a, b)]

    def id_dist(a: int, b: int) -> float:
        o = offset(a, b)
        return 1.0 if o is None else identity_distance(pieces[a].at(WORK_SCALE), pieces[b].at(WORK_SCALE), o)

    ids = cluster(range(n), id_dist, IDENTITY_FRACTION,
                  apart=lambda a, b: records[a]["line"] == records[b]["line"])
    ids.sort(key=lambda g: min((records[i]["line"], records[i]["bbox"][0]) for i in g))

    # 표정 — 인물 무리마다, 가장 넓은 온전한 조각을 기준으로 맞대고 머리 쪽을 영상 화소로 견준다.
    labels: Dict[int, Tuple[str, str]] = {}
    for ci, members in enumerate(ids):
        name = f"c{ci + 1:02d}"
        whole = [i for i in members if not records[i]["edge"] and not records[i]["split"]] or members
        ref = max(whole, key=lambda i: pieces[i].area)
        at: Dict[int, Tuple[int, int]] = {ref: (0, 0)}
        for i in members:
            if i == ref:
                continue
            o = offset(ref, i)
            if o is None:
                o = (0, 0)
            at[i] = refine(pieces[ref], pieces[i], int(round(o[0] / WORK_SCALE)), int(round(o[1] / WORK_SCALE)),
                           ALIGN_REFINE + 1)
        rows = np.nonzero(pieces[ref].mask.any(1))[0]
        top, low = int(rows.min()), int(rows.max()) + 1
        head = (top, int(np.ceil(top + HEAD_PART * (low - top))))
        pair: Dict[Tuple[int, int], Tuple[int, int]] = {}

        def face_offset(a: int, b: int) -> Tuple[int, int]:
            if (a, b) not in pair:
                ox, oy = at[b][0] - at[a][0], at[b][1] - at[a][1]
                pair[(a, b)] = refine(pieces[a], pieces[b], ox, oy, ALIGN_REFINE)
            return pair[(a, b)]

        def ex_dist(a: int, b: int) -> float:
            ha = (head[0] - at[a][1], head[1] - at[a][1])
            return expression_distance(pieces[a], pieces[b], face_offset(a, b), ha, max(3, int(round(EXPR_INSET * unit))))

        exprs = cluster(members, ex_dist, EXPR_PIXELS * unit * unit, complete=True,
                        apart=lambda a, b: records[a]["line"] == records[b]["line"])
        exprs.sort(key=lambda g: min(records[i]["line"] for i in g))
        for ei, g in enumerate(exprs):
            for i in g:
                labels[i] = (name, f"{ei + 1:02d}")
        log(f"  {name}: 조각 {len(members)} · 표정 {len(exprs)}")

    order = sorted(range(n), key=lambda i: (records[i]["line"], records[i]["bbox"][0]))
    for k, i in enumerate(order):
        rec = records[i]
        oid = f"o{k + 1:04d}"
        rel = os.path.join("obs", f"{oid}.png")
        write_image(os.path.join(out_dir, rel), pieces[i].rgba())
        line = rec["line"]
        draft.observations.append(Observation(
            oid, line, float(draft.lines[line]["time"]), rec["bbox"], rel, pieces[i].area, rec["edge"],
            draft.lines[line]["background"], labels[i][0], "a", labels[i][1], split=rec["split"]))
    path = draft.save(out_dir)
    log(f"인물 {len(ids)}명 · 표정 무리 {len(set(labels.values()))}개 → {path}")
    return draft
