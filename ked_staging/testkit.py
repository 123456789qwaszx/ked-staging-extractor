"""시험용 초상 — 표정 · 변형마다 화면에서 구별되는 그림을 만든다.

런타임 견본 초상은 표정 · 변형 그림이 모두 같은 파일이라 face · pose 교체를 영상으로 구별할 수 없다.
P1 시험장은 그래서 견본 초상 위에 표정마다 다른 얼굴 표시(눈썹 · 입 · 볼)를 얹고, 변형 b는 색조를 돌린
그림을 쓴다. 실제 게임 그림처럼 표정 차이는 얼굴 근처에만 있고, 변형 차이는 몸 전체에 있다.

이 그림은 시험장 전용이다 — 추출기는 어떤 그림이 오든 "표정끼리 다른 자리"를 스스로 찾는다(observe).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .catalog import Catalog, Portrait
from .imageio import write_image

KIT_VERSION = "1"
# 견본 초상(1745×2170)에서 눈으로 잰 얼굴 자리 — (왼눈, 오른눈, 입). 모르는 인물은 불투명 상자에서 어림한다.
FACE_POINTS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = {
    "harin": ((818, 258), (932, 258), (878, 330)),
    "sigure": ((889, 232), (991, 252), (919, 324)),
    "kasmi": ((820, 318), (904, 322), (850, 378)),
}
_REF_SIZE = (1745, 2170)
_INK = (40, 35, 60, 255)
_BLUSH = (150, 140, 245, 255)
_MOUTH = (60, 50, 170, 255)


def face_points(portrait: Portrait) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, h = portrait.size
    if portrait.character in FACE_POINTS:
        sx, sy = w / _REF_SIZE[0], h / _REF_SIZE[1]
        return tuple(np.array([x * sx, y * sy]) for x, y in FACE_POINTS[portrait.character])  # type: ignore
    x, y, bw, bh = portrait.opaque_box
    cx, top = x + bw / 2, y
    d = 0.06 * bw
    return (np.array([cx - d, top + 0.1 * bh]), np.array([cx + d, top + 0.1 * bh]),
            np.array([cx, top + 0.13 * bh]))


def _pt(p) -> Tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def draw_expression(img: np.ndarray, portrait: Portrait, emotion: int) -> np.ndarray:
    """표정 번호(1부터)마다 눈썹 기울기 · 입 모양 · 볼 표시가 다른 얼굴을 얹는다."""
    out = img.copy()
    left, right, mouth = face_points(portrait)
    d = float(np.linalg.norm(right - left))
    t = max(3, int(round(0.07 * d)))
    k = (emotion - 1) % 5
    # 눈썹: 바깥쪽 끝의 높낮이로 표정을 낸다(0 평평 · 1 올라감 · 2 찡그림 · 3 크게 올라감 · 4 처짐).
    tilt = [0.0, -0.12, 0.18, -0.25, 0.22][k] * d
    for eye, side in ((left, -1), (right, 1)):
        inner = eye + np.array([-side * 0.18 * d, -0.42 * d])
        outer = eye + np.array([side * 0.26 * d, -0.42 * d + tilt])
        cv2.line(out, _pt(inner), _pt(outer), _INK, t, cv2.LINE_AA)
    # 입
    w = 0.42 * d
    if k == 0:
        cv2.line(out, _pt(mouth - [w / 2, 0]), _pt(mouth + [w / 2, 0]), _INK, t, cv2.LINE_AA)
    elif k == 1:
        cv2.ellipse(out, _pt(mouth - [0, 0.08 * d]), (int(w / 2), int(0.22 * d)), 0, 10, 170, _INK, t, cv2.LINE_AA)
    elif k == 2:
        cv2.ellipse(out, _pt(mouth + [0, 0.18 * d]), (int(w / 2), int(0.2 * d)), 0, 190, 350, _INK, t, cv2.LINE_AA)
    elif k == 3:
        cv2.ellipse(out, _pt(mouth), (int(0.16 * d), int(0.22 * d)), 0, 0, 360, _MOUTH, -1, cv2.LINE_AA)
        cv2.ellipse(out, _pt(mouth), (int(0.16 * d), int(0.22 * d)), 0, 0, 360, _INK, max(2, t // 2), cv2.LINE_AA)
    else:
        pts = [mouth + [-w / 2 + i * w / 4, (-1) ** i * 0.06 * d] for i in range(5)]
        cv2.polylines(out, [np.int32([_pt(p) for p in pts])], False, _INK, t, cv2.LINE_AA)
    # 볼: 웃는 표정(1)과 놀람(3)에만
    if k in (1, 3):
        for eye in (left, right):
            cv2.ellipse(out, _pt(eye + [0, 0.42 * d]), (int(0.2 * d), int(0.1 * d)), 0, 0, 360, _BLUSH, -1,
                        cv2.LINE_AA)
    return out


def shift_hue(img: np.ndarray, degrees: float = 50.0, min_saturation: int = 50) -> np.ndarray:
    """채도가 있는 픽셀의 색조를 돌린다 — 변형(옷 · 머리색이 다른 그림) 흉내."""
    bgr = np.ascontiguousarray(img[:, :, :3])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sel = hsv[:, :, 1] >= min_saturation
    h = hsv[:, :, 0].astype(np.int32)
    h[sel] = (h[sel] + int(round(degrees / 2))) % 180
    hsv[:, :, 0] = h.astype(np.uint8)
    out = img.copy()
    out[:, :, :3] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


def test_portrait(base: Catalog, portrait: Portrait) -> np.ndarray:
    img = base.image(portrait)
    if portrait.variant != "a":
        img = shift_hue(img, 50.0 * (ord(portrait.variant[0]) - ord("a")))
    return draw_expression(img, portrait, int(portrait.emotion or "1"))


def build_test_catalog(base: Catalog, out_dir: str, force: bool = False) -> Catalog:
    """base의 초상마다 시험용 그림을 out_dir/<인물>/<변형>/<표정>.png로 쓰고, 그것을 가리키는 카탈로그를 낸다.
    배경은 base의 원본을 그대로 가리킨다. 이미 만든 것이 있으면 다시 쓰지 않는다(force로 다시)."""
    out_dir = os.path.abspath(out_dir)
    stamp = os.path.join(out_dir, "testkit-version.txt")
    fresh = force or not os.path.isfile(stamp) or open(stamp, encoding="utf-8").read().strip() != KIT_VERSION
    cat = Catalog("testkit", out_dir)
    for p in base.portraits:
        rel = os.path.join(p.character, p.variant, f"{p.emotion}.png")
        path = os.path.join(out_dir, rel)
        if fresh or not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_image(path, test_portrait(base, p))
        cat.portraits.append(Portrait(p.character, p.variant, p.emotion, rel, p.size, p.opaque_box))
    for b in base.backgrounds:
        cat.backgrounds.append(type(b)(b.key, base.abspath(b.path), b.size))
    if fresh:
        os.makedirs(out_dir, exist_ok=True)
        with open(stamp, "w", encoding="utf-8") as f:
            f.write(KIT_VERSION)
    cat.save(os.path.join(out_dir, "catalog.json"))
    return cat


def contact_sheet(cat: Catalog, character: str, height: int = 260) -> Optional[np.ndarray]:
    """한 인물의 표정 · 변형을 얼굴 근처만 잘라 한 장에 늘어놓는다(눈으로 확인용)."""
    tiles = []
    for p in cat.portraits:
        if p.character != character:
            continue
        img = cat.image(p)
        left, right, mouth = face_points(p)
        d = float(np.linalg.norm(right - left))
        c = (left + right) / 2
        x0, y0 = int(c[0] - 1.6 * d), int(c[1] - 1.2 * d)
        crop = img[max(0, y0):y0 + int(2.6 * d), max(0, x0):x0 + int(3.2 * d)]
        a = crop[:, :, 3:4] / 255.0
        tile = (crop[:, :, :3] * a + 200 * (1 - a)).astype(np.uint8)
        s = height / tile.shape[0]
        tile = cv2.resize(tile, (int(tile.shape[1] * s), height), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"{p.variant}/{p.emotion}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        tiles.append(tile)
    return np.hstack(tiles) if tiles else None
