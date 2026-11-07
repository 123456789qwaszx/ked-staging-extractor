import unittest

import numpy as np

from ked_staging.ease import DEFAULT_EASE, EaseTable
from tests.support import GOLDEN, needs_runtime


@needs_runtime
class EaseTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = EaseTable.load(GOLDEN)

    def test_has_the_dotween_set(self):
        self.assertEqual(len(self.table.names), 35)
        self.assertIn(DEFAULT_EASE, self.table)
        self.assertIn("Linear", self.table)

    def test_endpoints(self):
        for name in self.table.names:
            if "Flash" in name:
                continue  # Flash 계열은 깜빡임이라 끝값 규칙이 다르다
            self.assertAlmostEqual(float(self.table.evaluate(name, 0.0)), 0.0, places=5, msg=name)
            self.assertAlmostEqual(float(self.table.evaluate(name, 1.0)), 1.0, places=5, msg=name)

    def test_interpolates_exactly_at_samples_and_clamps(self):
        s = self.table.samples["OutCubic"]
        t = np.arange(len(s)) / (len(s) - 1)
        np.testing.assert_allclose(self.table.evaluate("OutCubic", t), s, atol=1e-12)
        self.assertEqual(float(self.table.evaluate("OutCubic", -0.5)), float(s[0]))
        self.assertEqual(float(self.table.evaluate("OutCubic", 2.0)), float(s[-1]))

    def test_outcubic_matches_formula(self):
        t = np.linspace(0, 1, 97)
        np.testing.assert_allclose(self.table.evaluate("OutCubic", t), 1 - (1 - t) ** 3, atol=2e-5)

    def test_overshooting_curves(self):
        over = set(self.table.overshooting())
        self.assertIn("OutBack", over)
        self.assertNotIn("OutCubic", over)

    def test_canonical_name(self):
        self.assertEqual(self.table.canonical("outcubic"), "OutCubic")


if __name__ == "__main__":
    unittest.main()
