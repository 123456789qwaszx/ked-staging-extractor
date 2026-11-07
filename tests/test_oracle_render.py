"""오라클 + 예측 화면 + 시험장 — 오라클 빌드와 런타임 저장소가 있을 때만 돈다."""
import os
import tempfile
import unittest

import numpy as np

from ked_staging.catalog import import_runtime
from ked_staging.oracle import Oracle, route_lines
from ked_staging.render import composite, coverages, render, render_layers
from ked_staging.synthetic import shifted, write_recording
from ked_staging.testbed import Recording, run_recording, summarize, visual_states
from tests.support import MOTIONS, RUNTIME, TUNING, needs_oracle, yarn

ROUTE = [yarn("festival_eve_Festival_Classroom.yarn"), yarn("festival_eve_Festival_Staffroom.yarn")]


@needs_oracle
class OracleRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = Oracle(TUNING, MOTIONS)
        cls.catalog = import_runtime(RUNTIME)
        cls.lines = route_lines(cls.oracle.fold_yarn(ROUTE))

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()

    def test_lines_follow_the_script(self):
        self.assertEqual(self.lines[0]["node"], "Festival_Classroom")
        first = self.lines[0]["snapshot"]
        keys = [s["key"] for s in first.slots]
        self.assertIn("bg_main", keys)
        self.assertEqual(keys[0], "bg_main")  # 배경(far 층)이 맨 먼저 그려진다

    def test_quad_is_the_drawn_sprite(self):
        slot = next(s for s in self.lines[1]["snapshot"].slots if s["key"] == "s1")
        x, y, w, h = slot["bounds"]
        self.assertAlmostEqual(w / h, slot["spriteAspect"], places=2)   # preserveAspect로 줄인 그림
        self.assertGreater(slot["rect"][2], w)                          # rect는 스트레치라 더 넓다

    def test_fold_matches_candidate_api(self):
        snaps = self.oracle.fold([self.lines[0]["commands"]])
        a = {s["key"]: s.get("bounds") for s in snaps[0].slots}
        b = {s["key"]: s.get("bounds") for s in self.lines[0]["snapshot"].slots}
        self.assertEqual(a, b)

    def test_render_and_coverage(self):
        snap = self.lines[1]["snapshot"]
        img = render(snap, self.catalog, self.oracle.width, self.oracle.height, scale=0.25)
        self.assertEqual(img.shape, (270, 480, 3))
        layers = render_layers(snap, self.catalog, self.oracle.width, self.oracle.height, scale=0.25)
        np.testing.assert_array_equal(composite(layers, 480, 270), img)
        cover = coverages(layers)
        total = sum(cover.values())
        self.assertLessEqual(float(total.max()), 1.0 + 1e-5)  # 보이는 정도의 합은 1을 넘지 않는다


@needs_oracle
class TestbedSyntheticTests(unittest.TestCase):
    def test_alignment_and_planted_offset(self):
        catalog = import_runtime(RUNTIME)
        with Oracle(TUNING, MOTIONS) as oracle, tempfile.TemporaryDirectory() as tmp:
            states = visual_states(route_lines(oracle.fold_yarn(ROUTE)))
            snaps = [s.snapshot for s in states]
            # 첫 상태는 뒤 상태와 합쳐지지 않는다(합쳐진 구간은 뒤쪽 프레임에서 잰다).
            snaps[0] = shifted(snaps[0], "s1", -4.0, 1.5)
            video = os.path.join(tmp, "syn.mp4")
            write_recording(video, snaps, catalog, oracle.width, oracle.height, hold=0.8, fade=0.2)
            results = run_recording(Recording(video, ROUTE), oracle, catalog, TUNING, log=lambda m: None)
        summary = summarize(results)
        self.assertEqual(summary["statesFound"], summary["states"])
        planted = [o for r in results for o in r.offsets if o.distance > 1.0]
        self.assertEqual(len(planted), 1)
        self.assertAlmostEqual(planted[0].dx, -4.0, delta=0.25)
        self.assertAlmostEqual(planted[0].dy, 1.5, delta=0.25)


if __name__ == "__main__":
    unittest.main()
