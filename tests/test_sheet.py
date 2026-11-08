"""엑셀 연출 열 ↔ .yarn(P5) — 고치지 않은 왕복은 바이트 그대로, 고친 칸만 다시 쓴다."""
import glob
import os
import shutil
import tempfile
import unittest

from openpyxl import load_workbook

from ked_staging.sheet import SheetError, apply_sheet, export_sheet, parse_yarn
from tests.support import REPO, YARN_DIR, needs_runtime

SAMPLE = """title: Demo
---
<<shot_reset 0fr>>
<<bg_spawn bg_main class_day>>
<<slot s1 stage00 mid>>
<<cast s1 harin a 1>>

<<bg_fade_in bg_main 24fr>>
교실은 조용하다. #line:ln_1

// 확인: 무언가
<<place_center s1 bust 0fr>>
<<size s1 mid bust 0fr>>
<<fade_in s1 18fr>>
하린: 안녕! #line:ln_2

<<face s1 2>>
<<pause 6fr>>
<<motion s1 nod>>
하린: 좋은 아침. #line:ln_3

-> 같이 간다
    <<face s1 3>>
    하린: 좋아!
<<fade_out s1 14fr>>
===
"""


def _write(tmp: str, text: str, name: str = "demo.yarn") -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path


def _edit(xlsx: str, key: str, **cells) -> None:
    wb = load_workbook(xlsx)
    ws = wb["연출"]
    head = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2):
        if row[head.index("열쇠")].value == key:
            for col, value in cells.items():
                row[head.index(col)].value = value
    wb.save(xlsx)


class SheetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.yarn = _write(self.tmp, SAMPLE)
        self.xlsx = os.path.join(self.tmp, "demo.xlsx")
        export_sheet(self.yarn, self.xlsx)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rows(self):
        wb = load_workbook(self.xlsx)
        ws = wb["연출"]
        head = [c.value for c in ws[1]]
        return [dict(zip(head, r)) for r in ws.iter_rows(min_row=2, values_only=True)]

    def test_rows_units_motion_checks(self):
        rows = {r["열쇠"]: r for r in self.rows()}
        self.assertEqual(rows["Demo#2"]["연출"], "@등장 s1 center bust mid 18fr")
        self.assertEqual(rows["Demo#2"]["확인"], "무언가")
        self.assertEqual((rows["Demo#3"]["연출"], rows["Demo#3"]["모션"]), ("face s1 2", "pause 6fr\nmotion s1 nod"))
        self.assertEqual(rows["Demo#4"]["종류"], "선택지")
        self.assertEqual(rows["Demo#5"]["연출"], "face s1 3")
        self.assertEqual(rows["Demo#0"]["연출"], "fade_out s1 14fr")

    def test_untouched_roundtrip_is_identical(self):
        text, changed = apply_sheet(self.xlsx)
        self.assertEqual((text, changed), (SAMPLE, []))

    def test_edits_rewrite_only_those_blocks(self):
        _edit(self.xlsx, "Demo#2", 연출="@등장 s1 left bust front 12fr")
        _edit(self.xlsx, "Demo#3", 모션="motion s1 bow")
        _edit(self.xlsx, "Demo#5", 연출="<<face s1 4>>")
        _edit(self.xlsx, "Demo#1", 화자="해설", 대사="교실은 아직 조용하다.")
        text, changed = apply_sheet(self.xlsx)
        self.assertEqual(changed, ["Demo#1", "Demo#2", "Demo#3", "Demo#5"])
        self.assertIn("해설: 교실은 아직 조용하다. #line:ln_1\n", text)
        self.assertIn("// 확인: 무언가\n<<place_left s1 bust 0fr>>\n<<size s1 front bust 0fr>>\n<<fade_in s1 12fr>>\n"
                      "하린: 안녕! #line:ln_2", text)
        self.assertIn("<<face s1 2>>\n<<motion s1 bow>>\n하린: 좋은 아침.", text)
        self.assertIn("-> 같이 간다\n    <<face s1 4>>\n    하린: 좋아!", text)
        # 나머지는 그대로
        self.assertEqual(text.count("\n"), SAMPLE.count("\n") - 1)   # pause 줄이 빠졌다

    def test_custom_unit_from_sheet(self):
        wb = load_workbook(self.xlsx)
        wb["단위"].append(["끄덕", "slot", "face $slot 2\nmotion $slot nod", "웃으며 끄덕"])
        wb.save(self.xlsx)
        _edit(self.xlsx, "Demo#3", 연출="@끄덕 s1", 모션=None)
        text, _ = apply_sheet(self.xlsx)
        self.assertIn("<<face s1 2>>\n<<motion s1 nod>>\n하린: 좋은 아침.", text)

    def test_refusals(self):
        _edit(self.xlsx, "Demo#5", 연출="@없는단위 s1")
        with self.assertRaises(SheetError):
            apply_sheet(self.xlsx)
        _edit(self.xlsx, "Demo#5", 연출="fcae s1 3")
        with self.assertRaises(SheetError):
            apply_sheet(self.xlsx, known_commands={"face", "fade_out"})
        _edit(self.xlsx, "Demo#5", 연출="face s1 3")
        _edit(self.xlsx, "Demo#4", 대사="-> 안 간다")
        with self.assertRaises(SheetError):           # 선택지 줄은 이 보기에서 고치지 않는다
            apply_sheet(self.xlsx)
        _edit(self.xlsx, "Demo#4", 대사="-> 같이 간다")
        with open(self.yarn, "a", encoding="utf-8") as f:
            f.write("\n")
        with self.assertRaises(SheetError):           # 내보낸 뒤 .yarn이 바뀜
            apply_sheet(self.xlsx)
        apply_sheet(self.xlsx, force=True)
        wb = load_workbook(self.xlsx)
        wb["연출"].delete_rows(3)
        wb.save(self.xlsx)
        with self.assertRaises(SheetError):           # 행을 지움
            apply_sheet(self.xlsx, force=True)

    def test_crlf_is_kept(self):
        crlf = _write(self.tmp, SAMPLE.replace("\n", "\r\n"), "crlf.yarn")
        x = os.path.join(self.tmp, "crlf.xlsx")
        export_sheet(crlf, x)
        self.assertEqual(apply_sheet(x)[0], SAMPLE.replace("\n", "\r\n"))
        _edit(x, "Demo#5", 연출="face s1 4")
        text, _ = apply_sheet(x)
        self.assertIn("    <<face s1 4>>\r\n", text)
        self.assertNotIn("\n", text.replace("\r\n", ""))


class RoundTripTests(unittest.TestCase):
    def check(self, paths):
        self.assertTrue(paths)
        with tempfile.TemporaryDirectory() as tmp:
            for y in paths:
                with open(y, encoding="utf-8", newline="") as f:
                    original = f.read()
                self.assertEqual(parse_yarn(original).render(), original)
                x = os.path.join(tmp, "t.xlsx")
                export_sheet(y, x)
                text, changed = apply_sheet(x, y)
                self.assertEqual((text == original, changed), (True, []), y)

    def test_scenarios(self):
        self.check(sorted(glob.glob(os.path.join(REPO, "testbed", "scenarios", "*.yarn"))))

    @needs_runtime
    def test_runtime_samples(self):
        self.check(sorted(glob.glob(os.path.join(YARN_DIR, "*.yarn"))))


if __name__ == "__main__":
    unittest.main()
