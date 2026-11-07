"""예측 화면 그리기 — 오라클 스냅샷 + 카탈로그 그림 → 정착 상태의 무대 한 장.

좌표는 전부 오라클이 준다(quad = 실제로 그려지는 그림의 네 모서리, 캔버스 좌상 원점 · y 아래).
여기는 그림을 그 자리에 붙이기만 한다: 아핀 변환 + 알파 합성, 순서는 오라클의 그리기 순서(먼 것 먼저).
대사창 · 화면 효과 · 마스크는 그리지 않는다 — 대조할 때 그 영역을 가린다.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .catalog import Catalog
from .oracle import Snapshot


class MissingSprite(LookupError):
    pass


def sprite_for(slot: dict, catalog: Catalog) -> np.ndarray:
    """슬롯이 그리는 그림(BGRA). mirror(코어 밖 축)면 좌우로 뒤집는다."""
    if slot.get("background"):
        entry = catalog.background(slot.get("sprite") or "")
        what = f"배경 '{slot.get('sprite')}'"
    else:
        entry = catalog.portrait(slot.get("character") or "", slot.get("variant") or "a", slot.get("emotion") or "")
        what = f"초상 {slot.get('character')}/{slot.get('variant')}/{slot.get('emotion')}"
    if entry is None:
        raise MissingSprite(f"카탈로그에 없는 {what} (슬롯 {slot.get('key')})")
    img = catalog.image(entry)
    if slot.get("toolMirror"):
        img = img[:, ::-1]
    return img


def _affine(slot: dict, src_w: int, src_h: int, scale: float) -> np.ndarray:
    """그림 좌상 · 우상 · 좌하 → quad의 TopLeft · TopRight · BottomLeft (quad 순서: BL, BR, TL, TR)."""
    bl, _br, tl, tr = (np.asarray(p, dtype=np.float32) * scale for p in slot["quad"])
    src = np.float32([[0, 0], [src_w, 0], [0, src_h]])
    dst = np.float32([tl, tr, bl])
    return cv2.getAffineTransform(src, dst)


def render_slot(slot: dict, catalog: Catalog, size: Tuple[int, int], scale: float = 1.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """슬롯 하나를 캔버스 크기로 편다 → (BGR float32, α float32 0..1). α에는 슬롯의 투명도가 곱해져 있다."""
    img = sprite_for(slot, catalog)
    h, w = img.shape[:2]
    m = _affine(slot, w, h, scale)
    # 많이 줄일 때는 미리 줄여 계단 현상을 막는다.
    shrink = min(1.0, np.sqrt(abs(np.linalg.det(m[:, :2]))))
    if shrink < 0.5:
        img = cv2.resize(img, (max(1, int(w * shrink)), max(1, int(h * shrink))), interpolation=cv2.INTER_AREA)
        m = _affine(slot, img.shape[1], img.shape[0], scale)
    warped = cv2.warpAffine(img, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0)).astype(np.float32)
    alpha = warped[:, :, 3] / 255.0 * float(slot.get("alpha", 1.0))
    return warped[:, :, :3], alpha


def render_layers(snapshot: Snapshot, catalog: Catalog, width: float, height: float, scale: float = 1.0,
                  only: Optional[Iterable[str]] = None) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """보이는 슬롯을 그리기 순서대로 편다 → [(키, BGR float32, α float32)]."""
    size = (int(round(width * scale)), int(round(height * scale)))
    keys = set(only) if only is not None else None
    layers = []
    for slot in snapshot.visible_slots():
        if keys is not None and slot["key"] not in keys:
            continue
        color, alpha = render_slot(slot, catalog, size, scale)
        layers.append((slot["key"], color, alpha))
    return layers


def composite(layers: Sequence[Tuple[str, np.ndarray, np.ndarray]], width: int, height: int,
              background_color: Sequence[int] = (0, 0, 0)) -> np.ndarray:
    canvas = np.empty((height, width, 3), np.float32)
    canvas[:] = np.asarray(background_color, np.float32)
    for _key, color, alpha in layers:
        a = alpha[:, :, None]
        canvas = color * a + canvas * (1.0 - a)
    return np.clip(canvas + 0.5, 0, 255).astype(np.uint8)


def coverages(layers: Sequence[Tuple[str, np.ndarray, np.ndarray]]) -> dict:
    """슬롯마다 화면에서 <b>보이는</b> 정도(앞에 선 슬롯에 가린 만큼 뺀다) — 키 → float32 0..1."""
    out = {}
    for i, (key, _color, alpha) in enumerate(layers):
        visible = alpha.copy()
        for _k, _c, front in layers[i + 1:]:
            visible *= (1.0 - front)
        out[key] = visible
    return out


def render(snapshot: Snapshot, catalog: Catalog, width: float, height: float, scale: float = 1.0,
           background_color: Sequence[int] = (0, 0, 0), only: Optional[Iterable[str]] = None
           ) -> np.ndarray:
    """정착 화면(BGR uint8). only를 주면 그 슬롯 키만 그린다."""
    layers = render_layers(snapshot, catalog, width, height, scale, only)
    return composite(layers, int(round(width * scale)), int(round(height * scale)), background_color)
