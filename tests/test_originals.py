"""다른 게임 — 원본 모드(catalog-originals)와 대사 엑셀로 줄 세우기."""
import json
import os
import shutil
import tempfile
import unittest


from ked_staging.lines import LineSpan, spans_from_text
from ked_staging.originals import ident, read_folder, stance
from tests.support import needs_oracle


class FolderTests(unittest.TestCase):
    def test_names_variants_emotions_and_backgrounds(self):
        tmp = tempfile.mkdtemp()
        try:
            for name in ("Kasimira_a_14.png", "Kasimira_a_02.png", "Chaton_b_3.png", "Desert_Scorpion_Male_Grunt_a_02.png",
                         "Desert_Scorpion_Male_Grunt_a_02 (1).png", "BG_mirage_street_daylight.png", "notes.txt",
                         "Weird Name_a_05.png"):
                open(os.path.join(tmp, name), "wb").close()
            sprites, bgs = read_folder(tmp)
            self.assertEqual(sorted(sprites[("kasimira", "a")]), ["02", "14"])
            self.assertEqual(list(sprites[("chaton", "b")]), ["03"])
            self.assertEqual(list(sprites[("desert_scorpion_male_grunt", "a")]), ["02"])   # ' (1)' 사본은 뺀다
            self.assertIn(("weird_name", "a"), sprites)
            self.assertEqual(list(bgs), ["mirage_street_daylight"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ident_follows_runtime_key_rule(self):
        self.assertEqual(ident("Desert_Scorpion-Male"), "desert_scorpion_male")
        self.assertEqual(ident("3rd"), "x_3rd")

    def test_stance_is_the_most_common_spot(self):
        hits = [{"scale": 0.92, "x": 666.0, "y": 85.0}, {"scale": 0.921, "x": 667.0, "y": 86.0},
                {"scale": 0.92, "x": 665.0, "y": 85.0}, {"scale": 0.80, "x": 100.0, "y": 140.0}]
        st = stance(hits)
        self.assertEqual((st["lines"], st["seen"]), (3, 4))
        self.assertAlmostEqual(st["x"], 666.0)


class SpansFromTextTests(unittest.TestCase):
    def test_phantom_splits_and_merged_lines_follow_the_text(self):
        # 잉크 줄: 0~3초 줄 하나, 3초에 헛 지움(3.0~3.3), 3.3~9초에 두 대사가 붙어 하나로 잡혔다.
        ink = [LineSpan(0.0, 1.0, 3.0, "onset", 100), LineSpan(3.0, 3.0, 3.3, "clear", 20),
               LineSpan(3.3, 4.0, 9.0, "clear", 120)]
        read = [1.2, 4.1, 6.5, 6.6]                      # 6.6은 같은 대사를 거듭 읽은 것
        out = spans_from_text(ink, read, 10.0)
        self.assertEqual(len(out), 3)
        self.assertEqual([round(s.head, 2) for s in out], [0.0, 3.3, 6.1])   # 6.5에는 잉크 머리가 없다 → 0.4초 앞
        self.assertEqual([round(s.end, 2) for s in out], [3.3, 6.1, 9.0])
        self.assertTrue(all(s.head <= s.typed <= s.end for s in out))


@needs_oracle
class OriginalsScenarioTests(unittest.TestCase):
    def test_unknown_game_with_originals(self):
        from ked_staging.__main__ import main
        tmp = tempfile.mkdtemp()
        try:
            code = main(["p4-check", "--originals", "--out", tmp])
            report = json.load(open(os.path.join(tmp, "report.json"), encoding="utf-8"))
            self.assertEqual(code, 0, ({k: v for k, v in report.items() if k != "lines"},
                                       [r for r in report["lines"] if r["state"]]))
            self.assertGreater(report["meanIoU"], 0.95)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
