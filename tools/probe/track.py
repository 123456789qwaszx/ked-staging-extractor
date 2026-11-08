"""인물 추적 — 원본 그림(몸 특징점)을 프레임마다 찾고, 카메라를 빼서 배경 위 자리로 바꾼다.

python tools/probe/track.py <영상> <인물 원본.png> <배경 원본.png> <t0> <t1> <out.json>
줄마다 {"t", "cam":[배율,회전,x,y], "chr":[배율,회전,x,y,맞는 짝], "onbg":[배율,x,y]}
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import feats, frames, load, load_bg, match, sim_params, stage_mask

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("sprite"); ap.add_argument("bg")
ap.add_argument("t0", type=float); ap.add_argument("t1", type=float); ap.add_argument("out")
ap.add_argument("--min-inliers", type=int, default=15)
a = ap.parse_args()
spr = load(a.sprite, 0.5, alpha=True)
fs = feats(spr[:, :, :3], (spr[:, :, 3] > 200).astype(np.uint8) * 255)
bg, _ = load_bg(a.bg)
fb = feats(bg)
mask = stage_mask()
out = []
for t, f in frames(a.video, a.t0, a.t1, 1):
    ff = feats(f, mask)
    cam = match(fb, ff)
    ch = match(fs, ff, 0.8)
    rec = {"t": round(t, 3)}
    if cam:
        rec["cam"] = [round(v, 4) for v in sim_params(cam[0])]
    if ch and ch[1] >= a.min_inliers:
        M = ch[0]
        s, r, tx, ty = sim_params(M)
        rec["chr"] = [round(s, 4), round(r, 2), round(tx, 2), round(ty, 2), ch[1]]
        if cam:  # 인물 자리 = 카메라의 역 × 인물 변환
            P = np.linalg.inv(np.vstack([cam[0], [0, 0, 1]])) @ np.vstack([M, [0, 0, 1]])
            rec["onbg"] = [round(float(np.hypot(P[0, 0], P[1, 0])), 4), round(float(P[0, 2]), 2), round(float(P[1, 2]), 2)]
    out.append(rec)
json.dump(out, open(a.out, "w"))
print("frames", len(out), "seen", sum("chr" in r for r in out))
