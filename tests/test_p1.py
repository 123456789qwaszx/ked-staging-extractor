"""P1 — 시험용 초상 · 줄 경계 · 시험 녹화 → 추출 → 평가."""
import os
import tempfile
import unittest

import cv2
import numpy as np

from ked_staging.catalog import import_runtime
from ked_staging.dialogue import BoxLayout, BoxPainter, find_font
from ked_staging.lines import detect_lines, track_ink
from tests.support import MOTIONS, REPO, RUNTIME, TUNING, needs_oracle, needs_runtime

SCENARIO = os.path.join(REPO, "testbed", "scenarios", "p1_basic.yarn")


def _has_font() -> bool:
    try:
        find_font()
        return True
    except FileNotFoundError:
        return False


needs_font = unittest.skipUnless(_has_font(), "한글 글꼴이 없습니다(KED_FONT)")


@needs_runtime
class TestkitTests(unittest.TestCase):
    def test_expressions_differ_only_near_the_face(self):
        from ked_staging.testkit import face_points, test_portrait
        base = import_runtime(RUNTIME)
        p1, p2 = base.portrait("harin", "a", "01"), base.portrait("harin", "a", "02")
        a, b = test_portrait(base, p1), test_portrait(base, p2)
        ys, xs = np.nonzero(np.abs(a.astype(int) - b.astype(int)).max(2) > 0)
        left, right, mouth = face_points(p1)
        d = np.linalg.norm(right - left)
        self.assertGreater(len(xs), 500)
        self.assertLess(xs.max() - xs.min(), 3 * d)          # 얼굴 근처에만
        self.assertLess(abs(np.median(ys) - mouth[1]), 1.5 * d)
        vb = test_portrait(base, base.portrait("harin", "b", "01"))
        changed = np.abs(a.astype(int) - vb.astype(int)).max(2) > 30
        self.assertGreater(changed.mean(), 0.05)              # 변형은 몸 전체가 다르다


@needs_runtime
@needs_font
class LineBoundaryTests(unittest.TestCase):
    def test_heads_within_a_frame(self):
        lay = BoxLayout.from_tuning(TUNING).scaled(960, 540)
        painter = BoxPainter(lay)
        fps, cps = 30.0, 24.0
        script = [("하린", "첫 번째 줄입니다.", 0.5), ("", "두 번째 줄은 나레이션.", 2.2), ("시구레", "세 번째.", 4.0)]
        end = 5.5
        rng = np.random.default_rng(1)
        bg = np.tile(np.linspace(40, 200, 960, dtype=np.float32)[None, :, None], (540, 1, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lines.mp4")
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
            for n in range(int(end * fps)):
                t = n / fps
                img = np.clip(bg + rng.normal(0, 2, bg.shape), 0, 255).astype(np.uint8)
                cur = [s for s in script if s[2] <= t + 1e-9]
                if cur:
                    spk, body, head = cur[-1]
                    img = painter.paint(img, spk, body, int(np.floor((t - head) * cps + 1e-6)))
                w.write(img)
            w.release()
            spans = detect_lines(track_ink(path, lay))
        self.assertEqual(len(spans), 3)
        for span, (_, _, head) in zip(spans, script):
            self.assertLessEqual(abs(span.head - head), 1.01 / fps)
        self.assertEqual([s.kind for s in spans], ["onset", "clear", "clear"])


@needs_oracle
@needs_font
class P1ScenarioTests(unittest.TestCase):
    def test_extract_matches_truth(self):
        from ked_staging.evaluate import evaluate
        from ked_staging.gte import rows_from_truth, write_like_gte
        from ked_staging.oracle import Oracle, route_lines
        from ked_staging.pipeline import extract
        from ked_staging.synthetic import write_scenario
        from ked_staging.testkit import build_test_catalog
        with tempfile.TemporaryDirectory() as tmp, Oracle(TUNING, MOTIONS) as oracle:
            cat = build_test_catalog(import_runtime(RUNTIME), os.path.join(tmp, "kit"))
            lines = route_lines(oracle.fold_yarn([SCENARIO]))
            video = os.path.join(tmp, "p1.mp4")
            truth = write_scenario(video, lines, cat, (oracle.width, oracle.height), BoxLayout.from_tuning(TUNING),
                                   size=(960, 540), route=[SCENARIO])
            xlsx = write_like_gte(os.path.join(tmp, "gte.xlsx"), rows_from_truth(truth))
            staging = extract(video, oracle, cat, TUNING, tmp, xlsx, node="P1", log=lambda m: None)
            report = evaluate(truth, staging, oracle)
        self.assertEqual(report["matched"], len(truth["lines"]))
        self.assertLessEqual(report["headFramesMax"], 1.0)
        self.assertEqual(report["stateMatch"], 1.0, [r for r in report["lines"] if r["state"]])
        self.assertEqual(report["textExact"], 1.0)


if __name__ == "__main__":
    unittest.main()
