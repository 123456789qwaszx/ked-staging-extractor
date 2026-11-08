"""남은 어휘 · 화면 효과(P3) — 근사 모양 · 효과 상태 → 커맨드 · 플래시 · 노이즈 · 시험 녹화 → 추출 → 평가."""
import os
import tempfile
import unittest

import numpy as np

from ked_staging.effects import (EffectDetector, EffectPresets, EffectState, ObservedEffects, VignetteMasks,
                                 apply_flash, apply_noise, effect_commands, visual_hypotheses)
from ked_staging.evaluate import families
from ked_staging.vocab import Finding, apply_findings
from tests.support import MOTIONS, REPO, RUNTIME, TUNING, needs_oracle, needs_runtime

SCENARIOS = [os.path.join(REPO, "testbed", "scenarios", n) for n in ("p3_vocab.yarn", "p3_effects.yarn")]


class CommandTests(unittest.TestCase):
    def test_effect_commands_follow_state_changes(self):
        a = ObservedEffects()
        b = ObservedEffects(visual={"s2": ("dim", 1.0)}, vignette=("tension", 1.0))
        self.assertEqual(effect_commands(a, b), [
            {"name": "char_visual_dim", "args": ["s2", "1"]},
            {"name": "screen_vignette", "args": ["tension", "1"]}])
        c = ObservedEffects(visual={"s1": ("silhouette", 0.5)}, noise=("default", 0.25), flash=("soft", 1.0))
        self.assertEqual(effect_commands(b, c), [
            {"name": "char_visual_silhouette", "args": ["s1", "0.5"]},
            {"name": "char_visual_clear", "args": ["s2"]},
            {"name": "screen_vignette_clear", "args": []},
            {"name": "screen_noise", "args": ["default", "0.25"]},
            {"name": "screen_flash", "args": ["soft", "1"]}])
        self.assertEqual(effect_commands(c, ObservedEffects(visual={"s1": ("silhouette", 0.5)}, blur={"far": 1.0})), [
            {"name": "screen_noise_clear", "args": []},
            {"name": "screen_blur", "args": ["far", "1"]}])

    def test_families_ignore_header_shot_reset_and_clears(self):
        line = [{"name": "shot_reset", "args": ["0fr"]}, {"name": "screen_vignette_clear", "args": []},
                {"name": "char_visual_clear", "args": ["s1"]}, {"name": "motion", "args": ["s1", "nod"]},
                {"name": "slide_in", "args": ["s2", "right", "4u", "16fr"]}]
        self.assertEqual(families(line, header=True), {"char_visual_clear", "motion:nod", "slide_in"})
        self.assertIn("shot", families(line))

    def test_slide_finding_matches_fade_length(self):
        line = [{"name": "fade_in", "args": ["s2", "12fr"]}]
        slide = {"name": "slide_in", "args": ["s2", "right", "4u", "16fr"]}
        out = apply_findings(line, [Finding("s2", "slide_in", [slide], 0.8)])
        self.assertEqual(out, [slide, {"name": "fade_in", "args": ["s2", "16fr"]}])

    def test_state_ramps(self):
        st = EffectState()
        st.apply_commands([{"name": "char_visual_dim", "args": ["s1", "1", "6fr"]},
                           {"name": "screen_vignette", "args": ["tension", "1", "8fr"]}], head=1.0)
        self.assertAlmostEqual(st.visual["s1"].value(1.0 + 3 / 24), 0.5, places=3)
        self.assertEqual(st.vignette.value(5.0), 1.0)
        st.apply_commands([{"name": "char_visual_clear", "args": ["s1"]}, {"name": "screen_vignette_clear", "args": []}],
                          head=5.0)
        self.assertEqual(st.visual["s1"].value(9.0), 0.0)
        self.assertEqual(st.vignette.value(9.0), 0.0)
        self.assertTrue(st.active(5.0))


@needs_runtime
class ShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = EffectPresets.load(TUNING)
        rng = np.random.default_rng(3)
        base = cv2_blur(rng.uniform(40, 200, (90, 160, 3)).astype(np.float32))
        cls.img = base
        cls.mask = np.ones((90, 160), np.float32)

    def test_vignette_undo_inverts_apply(self):
        vm = VignetteMasks(self.p)
        for hyp in [("tension", 1.0), ("memory", 0.5), ("focus", 2.0)]:
            back = vm.undo(vm.apply(self.img, hyp), hyp)
            self.assertLess(float(np.abs(back - self.img).max()), 1e-3, hyp)
        self.assertIn(("tension", 1.0), vm.hypotheses())
        self.assertNotIn("clear", {n for n, _ in vm.hypotheses()})

    def test_visual_hypotheses_skip_rim_only_presets(self):
        names = {n for n, _ in visual_hypotheses(self.p)}
        self.assertTrue({"dim", "silhouette", "defocus"} <= names)
        self.assertFalse(names & {"clear", "test", "focus", "inner_rim", "outer_rim"})

    def test_flash_found_and_ramps_ignored(self):
        det = EffectDetector(self.p, self.mask, 1.0)
        times = [i / 30 for i in range(18)]
        models = [self.img] * len(times)
        frames = [(t, apply_flash(self.img, self.p.flash["soft"], 1.0, t)) for t in times]
        hit = det.flash(frames, models, self.img)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[:2], ("soft", 1.0))
        # 비네트가 서서히 걸리는 줄 — 앞 화면과 뒤 화면 사이를 오갈 뿐이다.
        vm = VignetteMasks(self.p)
        ramp = [(t, vm.apply(self.img, ("tension", min(1.0, t / 0.33)))) for t in times]
        self.assertIsNone(det.flash(ramp, models, self.img))

    def test_noise_needs_more_than_floor(self):
        det = EffectDetector(self.p, self.mask, 1.0)
        rng = np.random.default_rng(5)
        floor = det.residual_sigma(self.img + rng.normal(0, 1.0, self.img.shape).astype(np.float32), self.img)
        clean = self.img + rng.normal(0, 1.0, self.img.shape).astype(np.float32)
        self.assertIsNone(det.noise(clean, self.img, floor))
        noisy = apply_noise(self.img, self.p.noise["default"], 0.5, rng)
        hit = det.noise(noisy, self.img, floor)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "default")
        self.assertAlmostEqual(hit[1], 0.5, delta=0.25)


def cv2_blur(img):
    import cv2
    return cv2.GaussianBlur(img, (0, 0), 3.0)


@needs_oracle
class P3ScenarioTests(unittest.TestCase):
    def run_scenario(self, yarn):
        from ked_staging.catalog import import_runtime
        from ked_staging.dialogue import BoxLayout
        from ked_staging.evaluate import evaluate, evaluate_timing, evaluate_vocab
        from ked_staging.oracle import Oracle, route_lines
        from ked_staging.pipeline import extract
        from ked_staging.synthetic import write_scenario
        from ked_staging.testkit import build_test_catalog
        with tempfile.TemporaryDirectory() as tmp, Oracle(TUNING, MOTIONS) as oracle:
            cat = build_test_catalog(import_runtime(RUNTIME), os.path.join(tmp, "kit"))
            lines = route_lines(oracle.fold_yarn([yarn]))
            video = os.path.join(tmp, "p3.mp4")
            truth = write_scenario(video, lines, cat, (oracle.width, oracle.height), BoxLayout.from_tuning(TUNING),
                                   size=(960, 540), route=[yarn], oracle=oracle,
                                   effect_presets=EffectPresets.load(TUNING))
            staging = extract(video, oracle, cat, TUNING, tmp, None, node="P3", log=lambda m: None)
            return (evaluate(truth, staging, oracle), evaluate_timing(truth, staging, oracle),
                    evaluate_vocab(truth, staging))

    def test_vocab_and_effects(self):
        for yarn in SCENARIOS:
            state, timing, vocab = self.run_scenario(yarn)
            with self.subTest(os.path.basename(yarn)):
                self.assertEqual(state["stateMatch"], 1.0, [r for r in state["lines"] if r["state"]])
                self.assertTrue(timing["pass"], timing["problems"])
                self.assertTrue(vocab["pass"], vocab["problems"])


if __name__ == "__main__":
    unittest.main()
