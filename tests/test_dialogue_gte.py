"""대사창 자리 · 게임텍스트추출기 엑셀 읽기와 줄 배정."""
import os
import tempfile
import unittest

from ked_staging.dialogue import BoxLayout, gte_profile, split_speaker
from ked_staging.gte import GteRow, attach, read_xlsx, write_like_gte
from tests.support import TUNING, needs_runtime


class SpeakerTests(unittest.TestCase):
    def test_split(self):
        self.assertEqual(split_speaker("하린: 좋은 아침!"), ("하린", "좋은 아침!"))
        self.assertEqual(split_speaker("아침 교실은 조용하다."), ("", "아침 교실은 조용하다."))
        self.assertEqual(split_speaker("시간: 23:40에 만나"), ("시간", "23:40에 만나"))


@needs_runtime
class LayoutTests(unittest.TestCase):
    def test_bottom_preset_in_canvas_pixels(self):
        lay = BoxLayout.from_tuning(TUNING)
        x0, y0, x1, y1 = lay.line
        self.assertAlmostEqual(x0, 0.08 * 1920, places=2)
        self.assertAlmostEqual(y1, (1 - 0.08) * 1080, places=2)       # 앵커는 y 위 방향 → 캔버스는 아래
        self.assertLess(lay.name[3], y0)                                # 이름표는 본문 위
        self.assertLess(lay.stage_bottom(), lay.name[1])
        half = lay.scaled(960, 540)
        self.assertAlmostEqual(half.line[2], x1 / 2, places=3)

    def test_gte_profile_matches_layout(self):
        prof = gte_profile(BoxLayout.from_tuning(TUNING))
        kinds = [r["type"] for r in prof["regions"]]
        self.assertEqual(kinds, ["area", "speaker"])
        area, speaker = prof["regions"]
        self.assertTrue(area["rect"][0] <= speaker["rect"][0] and area["rect"][1] <= speaker["rect"][1])


class GteTests(unittest.TestCase):
    def test_roundtrip_and_attach(self):
        rows = [GteRow(1, 1.2, "대사", "하린", "첫 줄"),
                GteRow(2, 3.4, "장소/시간", "", "교실 · 아침"),
                GteRow(3, 3.6, "나레이션", "", "둘째 줄 앞"),
                GteRow(4, 4.1, "나레이션", "", "둘째 줄 뒤"),
                GteRow(5, 75.25, "대사", "시구레", "1분 넘어서")]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_like_gte(os.path.join(tmp, "t.xlsx"), rows)
            back = read_xlsx(path)
        self.assertEqual([r.text for r in back], [r.text for r in rows])
        self.assertAlmostEqual(back[-1].t, 75.25, places=2)          # mm:ss.ss
        heads, ends = [1.0, 3.0, 5.0, 70.0], [3.0, 5.0, 7.0, 80.0]
        texts = attach(heads, ends, back)
        self.assertEqual((texts[0].speaker, texts[0].text), ("하린", "첫 줄"))
        self.assertEqual(texts[1].text, "둘째 줄 앞 둘째 줄 뒤")           # 한 줄에 둘 → 이어 붙이고 확인
        self.assertTrue(any("한 줄에 붙었" in r for r in texts[1].review))
        self.assertEqual(texts[1].notes, ["장소/시간: 교실 · 아침"])
        self.assertEqual(texts[2].rows, [])                               # 아무것도 안 붙은 줄
        self.assertTrue(texts[2].review)
        self.assertEqual(texts[3].speaker, "시구레")


if __name__ == "__main__":
    unittest.main()
