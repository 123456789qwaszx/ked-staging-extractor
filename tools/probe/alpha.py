"""투명도 — 그 프레임의 인물 자리·카메라로 '화면 = a·인물 + (1-a)·배경'을 몸 화소에서 푼다.

python tools/probe/alpha.py <영상> <인물 원본.png> <배경 원본.png> <track.json> <t0> <t1>
주의: 자리를 한 곳에 고정하면 미끄러지는 동안 틀린다 — track.json의 그 프레임 변환을 쓴다.
"""
import argparse, json, os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import frame_at, load, load_bg

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("sprite"); ap.add_argument("bg"); ap.add_argument("track")
ap.add_argument("t0", type=float); ap.add_argument("t1", type=float)
a = ap.parse_args()
spr = load(a.sprite, 0.5, alpha=True)
bg, _ = load_bg(a.bg)
for r in json.load(open(a.track)):
    if not (a.t0 <= r["t"] <= a.t1) or "chr" not in r or "cam" not in r:
        continue
    s, _, tx, ty, _ = r["chr"]
    cs, _, cx, cy = r["cam"]
    S_ = cv2.warpAffine(spr, np.float32([[s, 0, tx], [0, s, ty]]), (960, 540))
    B_ = cv2.warpAffine(bg, np.float32([[cs, 0, cx], [0, cs, cy]]), (960, 540))
    opaque = S_[:, :, 3] == 255
    ys = np.nonzero(opaque)[0]
    if len(ys) < 50:
        continue
    body = np.zeros_like(opaque)
    body[int(ys.min() + 0.45 * (ys.max() - ys.min())):400] = True   # 얼굴(표정)은 뺀다
    m = opaque & body
    F = frame_at(a.video, r["t"]).astype(np.float32)[m]
    Sv = S_[:, :, :3].astype(np.float32)[m]
    Bv = B_.astype(np.float32)[m]
    d = Sv - Bv
    al = float(((F - Bv) * d).sum() / max((d * d).sum(), 1e-6))
    res = float(np.abs(F - (al * Sv + (1 - al) * Bv)).mean())
    print(r["t"], round(al, 3), "오차", round(res, 1))
