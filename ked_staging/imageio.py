"""한글 경로에서도 깨지지 않는 이미지 입출력 (OpenCV는 Windows에서 비ASCII 경로를 못 연다)."""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def read_image(path: str, alpha: bool = False) -> Optional[np.ndarray]:
    """BGR(A) uint8 배열. alpha=True면 알파 채널을 살린다(없으면 255로 채운다)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    flag = cv2.IMREAD_UNCHANGED if alpha else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        return None
    if alpha:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def write_image(path: str, img: np.ndarray, params: Optional[list] = None) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise IOError(f"이미지를 인코딩하지 못했습니다: {path}")
    tmp = path + ".tmp" + ext
    buf.tofile(tmp)
    os.replace(tmp, path)
