import unittest

from ked_staging import tokens


class TokenTests(unittest.TestCase):
    def test_duration_forms(self):
        self.assertAlmostEqual(tokens.parse_duration("24fr"), 1.0)
        self.assertAlmostEqual(tokens.parse_duration("0.5s"), 0.5)
        self.assertAlmostEqual(tokens.parse_duration("0.5"), 0.5)   # 맨 숫자 = 초
        with self.assertRaises(ValueError):
            tokens.parse_duration("fast")

    def test_duration_token_is_frames(self):
        self.assertEqual(tokens.duration_token(0.75), "18fr")
        self.assertEqual(tokens.duration_token(0.0), "0fr")
        self.assertEqual(tokens.duration_token(-1), "0fr")

    def test_units(self):
        self.assertAlmostEqual(tokens.parse_units("1u"), 40.0)       # 1920 / 48
        self.assertAlmostEqual(tokens.parse_units("-2.5u"), -100.0)
        self.assertEqual(tokens.unit_token(100.0), "2.5u")
        self.assertEqual(tokens.unit_token(-80.0), "-2u")
        self.assertEqual(tokens.unit_token(41.0), "1u")


if __name__ == "__main__":
    unittest.main()
