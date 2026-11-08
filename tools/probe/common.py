"""실영상 관측 시험 공용 — 프레임 읽기, 특징점 짝, 닮음 변환.

다른 게임 관측 재설계(P4) 1단계 전에 '이 기술이 실영상에서 되나'를 확인한 스크립트다.
작업 배율은 0.5(1920x1080 → 960x540). 숫자를 1080p로 볼 때는 위치에 2를 곱한다.
"""
import cv2
import numpy as np

S = 0.5            # 작업 배율
STAGE_ROWS = 400   # 960x540 기준 대사창 위 무대 영역(행)
_sift = cv2.SIFT_create(4000)
_bf = cv2.BFMatcher()


def feats(img, mask=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return _sift.detectAndCompute(g, mask)


def stage_mask():
    m = np.zeros((int(1080 * S), int(1920 * S)), np.uint8)
    m[:STAGE_ROWS] = 255
    return m


def load(path, scale=1.0, alpha=False):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED if alpha else cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit(f"그림을 못 읽음: {path}")
    if scale != 1.0:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return im


def load_bg(path):
    """배경 원본을 작업 폭(960)으로 줄인다. 돌려주는 k는 줄인 배율."""
    bg = load(path)
    k = 960 / bg.shape[1] if bg.shape[1] > 960 else 1.0
    return cv2.resize(bg, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), k


def match(f1, f2, ratio=0.75, thresh=2.5):
    """특징점 짝 → RANSAC 닮음 변환. (M, 맞는 짝 수, 후보 짝 수) 또는 None."""
    k1, d1 = f1
    k2, d2 = f2
    if d1 is None or d2 is None:
        return None
    pairs = _bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thresh, maxIters=4000)
    if M is None:
        return None
    return M, int(inl.sum()), len(good)


def sim_params(M):
    s = float(np.hypot(M[0, 0], M[1, 0]))
    r = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    return s, r, float(M[0, 2]), float(M[1, 2])


def frames(video, t0, t1, step=1):
    """(시각, 작업 배율 프레임)을 차례로. step=3이면 10fps."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
    i = 0
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, f = cap.read()
        if not ok or t > t1:
            break
        if i % step == 0:
            yield t, cv2.resize(f, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
        i += 1


def frame_at(video, t, scale=S):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    if not ok:
        raise SystemExit(f"프레임 없음: {t}")
    return f if scale == 1.0 else cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
