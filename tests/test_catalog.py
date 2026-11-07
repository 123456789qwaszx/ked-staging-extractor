import os
import tempfile
import unittest

from ked_staging.catalog import Catalog, emotion_code, import_runtime
from tests.support import RUNTIME, needs_runtime


class EmotionCodeTests(unittest.TestCase):
    def test_same_rule_as_core(self):
        self.assertEqual(emotion_code("1"), "01")
        self.assertEqual(emotion_code("02"), "02")
        self.assertEqual(emotion_code("e1"), "")    # 표정 별칭은 show만 벗긴다
        self.assertEqual(emotion_code(""), "")


@needs_runtime
class RuntimeCatalogTests(unittest.TestCase):
    def test_import_and_round_trip(self):
        problems = []
        cat = import_runtime(RUNTIME, problems)
        self.assertGreater(len(cat.portraits), 0, problems)
        self.assertGreater(len(cat.backgrounds), 0, problems)
        p = cat.portraits[0]
        self.assertIsNotNone(cat.portrait(p.character, p.variant, str(int(p.emotion))))
        self.assertEqual(cat.image(p).shape[2], 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "catalog.json")
            cat.save(path)
            again = Catalog.load(path)
        self.assertEqual(again.to_json(), cat.to_json())


if __name__ == "__main__":
    unittest.main()
