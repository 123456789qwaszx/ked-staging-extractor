"""역해석 — 관측 상태 열 → 정본 커맨드."""
import unittest

from ked_staging.emit import command_text, yarn_text
from ked_staging.inverse import build_script
from ked_staging.observe import ObservedSlot, ObservedStage


def slot(c, place="center", depth="mid", emotion="01", variant="a", mirror=False, move=(0.0, 0.0), focus="bust"):
    return ObservedSlot(c, variant, emotion, place, focus, depth, mirror, move, 3.0, 5000)


def stage(bg, *slots):
    return ObservedStage(0.0, bg, 0.9, list(slots), 3.0, 0.0)


class InverseTests(unittest.TestCase):
    def script(self, *stages):
        sc = build_script(list(stages))
        return [command_text(c) for c in sc.header], [[command_text(c) for c in l] for l in sc.lines]

    def test_header_and_first_appearance(self):
        head, lines = self.script(stage("class_day"), stage("class_day", slot("harin", "right")))
        self.assertEqual(head, ["<<shot_reset 0fr>>", "<<bg_spawn bg_main class_day>>",
                                "<<slot s1 stage00 mid>>", "<<cast s1 harin a 1>>"])
        self.assertEqual(lines[0], ["<<bg_fade_in bg_main 24fr>>"])
        self.assertEqual(lines[1], ["<<place_right s1 bust 0fr>>", "<<size s1 mid bust 0fr>>", "<<fade_in s1 18fr>>"])

    def test_changes_while_visible(self):
        a = slot("harin", "left")
        _, lines = self.script(
            stage("x", a),
            stage("x", slot("harin", "left", emotion="03")),                      # 표정
            stage("x", slot("harin", "left", emotion="03", move=(-2.0, 0.0))),   # 이동만
            stage("x", slot("harin", "left", emotion="03", move=(-2.0, 0.0), depth="front")),  # 이동 중 크기
            stage("x", slot("harin", "center", emotion="03", depth="front", mirror=True)),     # 자리 · 반전
            stage("y"))
        self.assertEqual(lines[1], ["<<face s1 3>>"])
        self.assertEqual(lines[2], ["<<move_by s1 -2u 0u 12fr>>"])
        # 이동이 남은 채로 크기만 바꾸면 처음 세운 자리와 달라진다 → 다시 세운다
        self.assertEqual(lines[3], ["<<place_left s1 bust 12fr>>", "<<size s1 front bust 12fr>>",
                                    "<<move_by s1 -2u 0u 12fr>>"])
        self.assertEqual(lines[4], ["<<mirror s1 left>>", "<<place_center s1 bust 12fr>>"])
        self.assertEqual(lines[5], ["<<bg_sprite bg_main y>>", "<<fade_out s1 18fr>>"])

    def test_reappear_with_new_variant(self):
        _, lines = self.script(stage(None, slot("sigure")), stage(None), stage(None, slot("sigure", variant="b", emotion="02")))
        self.assertEqual(lines[1], ["<<fade_out s1 18fr>>"])
        self.assertEqual(lines[2], ["<<pose s1 b>>", "<<face s1 2>>", "<<fade_in s1 18fr>>"])

    def test_yarn_text_escapes(self):
        self.assertEqual(yarn_text("a #b {c}\n d // e"), "a \\#b \\{c\\} d /\\/ e")


if __name__ == "__main__":
    unittest.main()
