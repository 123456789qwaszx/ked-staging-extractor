"""시간과 이징(P2) — 진행도 맞추기 · 커맨드에 쓰기 · 시간 축 시험 녹화 → 추출 → 평가."""
import os
import tempfile
import unittest

import numpy as np

from ked_staging.ease import EaseTable
from ked_staging.timing import Channel, apply_timing, fit_progress
from ked_staging.tokens import FPS
from tests.support import GOLDEN, MOTIONS, REPO, RUNTIME, TUNING, needs_oracle, needs_runtime

SCENARIO = os.path.join(REPO, "testbed", "scenarios", "p2_timing.yarn")


@needs_runtime
class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = EaseTable.load(GOLDEN)

    def check(self, ease, start_fr, dur_fr, lag=0.004, noise=0.004):
        rng = np.random.default_rng(start_fr * 100 + dur_fr)
        t = np.arange(0, 2.5, 1 / 30) + lag
        p = np.clip((t - start_fr / FPS) / (dur_fr / FPS), 0, 1)
        v = self.table.evaluate(ease, p) + rng.normal(0, noise, len(t))
        s, d, e, rmse = fit_progress(t - lag, v, np.ones(len(t)), self.table)
        self.assertLessEqual(abs(d * FPS - dur_fr), 1.0, (ease, d * FPS))
        self.assertLessEqual(abs(s * FPS - start_fr), 1.0, (ease, s * FPS))
        self.assertEqual(e, ease)
        self.assertLess(rmse, 0.02)

    def test_recovers_common_eases(self):
        for ease, start, dur in [("OutCubic", 0, 15), ("InOutSine", 0, 25), ("Linear", 8, 20), ("OutBack", 0, 38),
                                 ("InOutQuad", 15, 30), ("OutQuart", 0, 23)]:
            self.check(ease, start, dur)

    def test_fixed_ease_for_fades(self):
        t = np.arange(0, 1.5, 1 / 30)
        v = self.table.evaluate("OutCubic", np.clip(t / (18 / FPS), 0, 1))
        s, d, e, _ = fit_progress(t, v, np.ones(len(t)), self.table, ["OutCubic"])
        self.assertEqual((round(s * FPS), round(d * FPS), e), (0, 18, "OutCubic"))


class _FakeOracle:
    _params = {"place_left": {"duration": 2, "ease": 3}, "size": {"duration": 3, "ease": 4},
               "fade_in": {"duration": 1}}

    def param_index(self, name, type_name):
        return self._params.get(name, {}).get(type_name)


class ApplyTests(unittest.TestCase):
    def test_durations_eases_and_waits(self):
        line = [{"name": "face", "args": ["s1", "2"]},
                {"name": "place_left", "args": ["s1", "bust", "15fr"]},
                {"name": "size", "args": ["s1", "front", "bust", "15fr"]},
                {"name": "fade_in", "args": ["s2", "23fr"]}]
        move = Channel("s1", "move", [1, 2], 0.0, 25 / FPS, "InOutSine", 0.01)
        fade = Channel("s2", "alpha", [3], 15 / FPS, 20 / FPS, "OutCubic", 0.01)
        out = apply_timing(line, [move, fade], _FakeOracle())
        self.assertEqual(out, [
            {"name": "face", "args": ["s1", "2"]},
            {"name": "place_left", "args": ["s1", "bust", "25fr", "InOutSine"]},
            {"name": "size", "args": ["s1", "front", "bust", "25fr", "InOutSine"]},
            {"name": "pause", "args": ["15fr"]},
            {"name": "fade_in", "args": ["s2", "20fr"]}])

    def test_default_ease_is_left_out(self):
        line = [{"name": "place_left", "args": ["s1", "bust", "15fr", "OutBack"]}]
        out = apply_timing(line, [Channel("s1", "move", [0], 0.0, 0.5, "OutCubic", 0.01)], _FakeOracle())
        self.assertEqual(out[0]["args"], ["s1", "bust", "15fr"])


@needs_oracle
class P2ScenarioTests(unittest.TestCase):
    def test_timing_matches_truth(self):
        from ked_staging.catalog import import_runtime
        from ked_staging.dialogue import BoxLayout
        from ked_staging.evaluate import evaluate, evaluate_timing
        from ked_staging.oracle import Oracle, route_lines
        from ked_staging.pipeline import extract
        from ked_staging.synthetic import write_scenario
        from ked_staging.testkit import build_test_catalog
        with tempfile.TemporaryDirectory() as tmp, Oracle(TUNING, MOTIONS) as oracle:
            cat = build_test_catalog(import_runtime(RUNTIME), os.path.join(tmp, "kit"))
            lines = route_lines(oracle.fold_yarn([SCENARIO]))
            video = os.path.join(tmp, "p2.mp4")
            truth = write_scenario(video, lines, cat, (oracle.width, oracle.height), BoxLayout.from_tuning(TUNING),
                                   size=(960, 540), route=[SCENARIO], oracle=oracle)
            staging = extract(video, oracle, cat, TUNING, tmp, None, node="P2", log=lambda m: None)
            state = evaluate(truth, staging, oracle)
            timing = evaluate_timing(truth, staging, oracle)
        self.assertEqual(state["stateMatch"], 1.0, [r for r in state["lines"] if r["state"]])
        self.assertTrue(timing["pass"], timing["problems"])
        self.assertGreaterEqual(timing["durationWithin"], 0.9)


if __name__ == "__main__":
    unittest.main()
