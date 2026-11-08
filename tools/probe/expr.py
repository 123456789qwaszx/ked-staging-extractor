"""표정 — 그 프레임의 정확한 변환으로 표정 그림을 전부 제자리에 그리고, 쌍마다 서로 다른 화소에서만 견주는 리그전.

python tools/probe/expr.py <영상> <track.json> "<원본 폴더>/<인물>_<변형>_*.png" --at 53.967 57.467
모든 다른 표정을 이긴 표정이 있으면 확정. 멈춘 프레임만 본다(움직이는 중에는 오차가 세 배로 뛴다).
"""
import argparse, glob, os, re, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import frame_at

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("track"); ap.add_argument("pattern"); ap.add_argument("--at", type=float, nargs="+", required=True)
a = ap.parse_args()
import json
track = {round(r["t"], 3): r for r in json.load(open(a.track))}
files = [f for f in sorted(glob.glob(a.pattern)) if re.search(r"_(\d+)\.png$", f)]
E = {re.search(r"_(\d+)\.png$", f).group(1): cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in files}
for t in a.at:
    r = track.get(round(t, 3))
    if not r or "chr" not in r:
        print(t, "추적 없음"); continue
    s, _, tx, ty, _ = r["chr"]
    M = np.float32([[s, 0, tx * 2], [0, s, ty * 2]])   # 원본(전체 해상도) → 1080p 프레임
    f = frame_at(a.video, t, 1.0).astype(np.float32)
    W = {n: cv2.warpAffine(e, M, (1920, 1080)).astype(np.float32) for n, e in E.items()}
    names = list(W)
    wins = {n: 0 for n in names}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            d = (np.abs(W[x][:, :, :3] - W[y][:, :, :3]).max(-1) > 40) & (W[x][:, :, 3] > 200) & (W[y][:, :, 3] > 200)
            if d.sum() < 30:
                continue
            ex = np.abs(f[d] - W[x][:, :, :3][d]).mean()
            ey = np.abs(f[d] - W[y][:, :, :3][d]).mean()
            wins[x if ex < ey else y] += 1
    best = sorted(wins.items(), key=lambda kv: -kv[1])[:4]
    print(t, best, "/", len(names) - 1, "확정" if best[0][1] == len(names) - 1 else "모름")
