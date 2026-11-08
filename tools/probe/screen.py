"""화면 색 — 배경 원본을 맞댄 뒤 채널마다 '화면 = 감쇠·원본 + 더함'을 맞춘다. 어느 원본이 더 맞는지도 본다.

python tools/probe/screen.py <영상> <배경1.png> [배경2.png ...] --at 139 147 150
감쇠가 평소(영상 자체 밝기, 약 0.9)보다 낮으면 어둡게·검은 화면, 짝이 없으면 배경 모름.
"""
import argparse, os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import feats, frame_at, load_bg, match, sim_params, stage_mask

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("bgs", nargs="+"); ap.add_argument("--at", type=float, nargs="+", required=True)
a = ap.parse_args()
mask = stage_mask()
bgs = [(os.path.basename(p), *load_bg(p)) for p in a.bgs]
bgs = [(n, b, feats(b)) for n, b, _ in bgs]
for t in a.at:
    f = frame_at(a.video, t)
    ff = feats(f, mask)
    row = [t, "밝기", round(float(f[:400].mean()), 1)]
    for n, b, fb in bgs:
        r = match(fb, ff)
        if r is None:
            row.append((n, "짝 없음")); continue
        M, inl, _ = r
        W = cv2.warpAffine(b, M, (960, 540))
        valid = (cv2.warpAffine(np.ones(b.shape[:2], np.uint8), M, (960, 540)) > 0) & (mask > 0)
        if valid.sum() < 1000:
            row.append((n, "겹침 적음")); continue
        X, Y = W[valid].astype(np.float32), f[valid].astype(np.float32)
        gains, errs = [], []
        for c in range(3):
            x, y = X[:, c], Y[:, c]
            g, o = np.polyfit(x, y, 1)
            keep = np.abs(y - (g * x + o)) < np.percentile(np.abs(y - (g * x + o)), 60)
            g, o = np.polyfit(x[keep], y[keep], 1)
            gains.append(round(float(g), 2)); errs.append(float(np.median(np.abs(y - (g * x + o)))))
        row.append((n, "짝", inl, "배율", round(sim_params(M)[0], 3), "감쇠", gains, "오차", round(float(np.mean(errs)), 1)))
    print(row)
