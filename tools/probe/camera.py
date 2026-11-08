"""카메라 — 배경 원본을 프레임마다 맞대어 배율·회전·이동을 잰다.

python tools/probe/camera.py <영상> <배경 원본.png> <t0> <t1> <out.json> [--step 3]
출력: [시각, 배율, 회전(도), x, y, 맞는 짝, 후보 짝] (배율은 줄인 배경 기준, 위치는 작업 배율 px)
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import feats, frames, load_bg, match, sim_params, stage_mask

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("bg"); ap.add_argument("t0", type=float); ap.add_argument("t1", type=float)
ap.add_argument("out"); ap.add_argument("--step", type=int, default=3)
a = ap.parse_args()
bg, _ = load_bg(a.bg)
fb = feats(bg)
mask = stage_mask()
out = []
for t, f in frames(a.video, a.t0, a.t1, a.step):
    r = match(fb, feats(f, mask))
    if r is None:
        out.append([round(t, 3), None]); continue
    M, inl, good = r
    s, rot, tx, ty = sim_params(M)
    out.append([round(t, 3), round(s, 4), round(rot, 2), round(tx, 2), round(ty, 2), inl, good])
json.dump(out, open(a.out, "w"))
for o in out[::max(1, len(out) // 20)]:
    print(o)
