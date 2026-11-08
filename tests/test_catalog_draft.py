"""다른 게임(P4) — 초안 고치기 · 조각 맞대기 · 덩어리 나누기 · 튜닝 겹본 · 모션 초안 · '모르는 게임' 왕복.

EditorTests는 컴포지터 tests/Vn.Authoring.Tests/Catalog/CatalogDraftTests.cs와 같은 초안 · 같은 기대값이다 —
두 쪽 고치기 연산의 뜻이 같다는 계약(docs/P4.md)을 양쪽에서 확인한다.
"""
import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from ked_staging.catalogdraft import DraftEditor, DraftError, write_overlay
from ked_staging.motiondraft import peak, to_motion, write_authoring
from ked_staging.scan import Piece, align, cluster, expression_distance, identity_distance, split_blob
from tests.support import TUNING, needs_oracle, needs_runtime

DRAFT = {
    "schema": "ked-staging-draft/1", "video": "v.mp4", "videoSize": [1280, 720], "stageBottom": 424,
    "boxProfile": None, "futureField": {"kept": True},
    "lines": [{"index": 0, "background": "bg01", "frame": "frames/l0000.jpg"}, {"index": 1, "background": "bg01"}],
    "backgrounds": [{"key": "bg01", "plate": "bg/bg01.png", "lines": [0, 1], "excluded": False}],
    "observations": [
        {"id": "o1", "line": 0, "time": 1.0, "bbox": [0, 0, 50, 80], "png": "obs/o1.png", "area": 1000, "edge": False,
         "background": "bg01", "character": "c01", "variant": "a", "emotion": "01", "excluded": False, "representative": False},
        {"id": "o2", "line": 1, "time": 2.0, "bbox": [0, 0, 50, 80], "png": "obs/o2.png", "area": 1200, "edge": False,
         "background": "bg01", "character": "c01", "variant": "a", "emotion": "01", "excluded": False, "representative": False},
        {"id": "o3", "line": 1, "time": 2.0, "bbox": [0, 0, 50, 80], "png": "obs/o3.png", "area": 900, "edge": True,
         "background": "bg01", "character": "c01", "variant": "a", "emotion": "02", "excluded": False, "representative": False},
        {"id": "o4", "line": 0, "time": 1.0, "bbox": [0, 0, 50, 80], "png": "obs/o4.png", "area": 800, "edge": False,
         "background": "bg01", "character": "c02", "variant": "a", "emotion": "01", "excluded": False, "representative": False},
        {"id": "o5", "line": 1, "time": 2.0, "bbox": [0, 0, 50, 80], "png": "obs/o5.png", "area": 700, "edge": False,
         "background": "bg01", "character": "c02", "variant": "a", "emotion": "02", "excluded": False, "representative": False},
    ],
}


def editor() -> DraftEditor:
    return DraftEditor(json.loads(json.dumps(DRAFT)), "draft.json")


def emotions(d: DraftEditor, character: str, variant: str = "a"):
    return [e for c, v, e, _ in d.groups() if c == character and v == variant]


class EditorTests(unittest.TestCase):
    def test_names_follow_rules_and_clash_asks_for_merge(self):
        d = editor()
        d.rename_character("c01", "Harin")
        self.assertEqual(d.characters(), ["c02", "harin"])
        self.assertRaises(DraftError, d.rename_character, "c02", "harin")
        self.assertRaises(DraftError, d.rename_character, "c02", "2bad")

    def test_merge_same_variant_appends_emotions(self):
        d = editor()
        d.merge_characters("c02", "c01")
        self.assertEqual(emotions(d, "c01"), ["01", "02", "03", "04"])

    def test_merge_as_variant_keeps_emotions(self):
        d = editor()
        d.merge_characters("c02", "c01", "b")
        self.assertEqual(emotions(d, "c01", "b"), ["01", "02"])
        self.assertRaises(DraftError, d.merge_characters, "c01", "c03", "bb")

    def test_move_exclude_representative_undo(self):
        d = editor()
        self.assertEqual(d.move(["o2"], "c01", "a"), "03")
        d.exclude(["o1"])
        self.assertEqual([o["id"] for o in d.observations if o.get("excluded")], ["o1"])
        self.assertEqual(d.representative("c01", "a", "02")["id"], "o3")   # 대표가 없으면 가장 넓은 조각
        d.set_representative("o3")
        self.assertTrue(d.find("o3")["representative"])
        self.assertTrue(d.undo())
        self.assertTrue(d.undo())
        self.assertFalse(any(o.get("excluded") for o in d.observations))
        self.assertTrue(d.undo())
        self.assertEqual(d.find("o2")["emotion"], "01")
        self.assertFalse(d.undo())

    def test_background_rename_follows_lines_and_keeps_unknown_fields(self):
        d = editor()
        d.rename_background("bg01", "classroom")
        tmp = tempfile.mkdtemp()
        try:
            path = d.save(os.path.join(tmp, "draft.json"))
            saved = json.load(open(path, encoding="utf-8"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertEqual(saved["backgrounds"][0]["key"], "classroom")
        self.assertTrue(all(l["background"] == "classroom" for l in saved["lines"]))
        self.assertTrue(all(o["background"] == "classroom" for o in saved["observations"]))
        self.assertTrue(saved["futureField"]["kept"])

    def test_other_schema_is_refused(self):
        self.assertRaises(DraftError, DraftEditor, dict(DRAFT, schema="other/1"), "draft.json")


def figure(w=120, h=200, color=(60, 120, 200), head=None, seed=0) -> Piece:
    """사람 모양 조각(머리 원 + 몸통 사다리꼴) — 무늬가 있어 맞대기가 하나로 정해진다."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 5), h // 7, 1, -1)
    cv2.fillPoly(mask, [np.array([[w // 4, h // 3], [3 * w // 4, h // 3], [w - 4, h - 1], [3, h - 1]])], 1)
    img[:] = color
    yy, xx = np.mgrid[0:h, 0:w]
    img += (np.sin(xx / 5.0) * 25 + np.cos(yy / 7.0) * 25)[:, :, None]
    img += rng.normal(0, 1.5, img.shape)
    if head is not None:                     # 표정: 머리 쪽 작은 표시
        cv2.line(img, (w // 2 - 12, h // 5 + head), (w // 2 + 12, h // 5 - head), (20, 20, 20), 3)
    return Piece(np.clip(img, 0, 255).astype(np.float32), mask.astype(bool))


def crop(p: Piece, x0, y0, x1, y1) -> Piece:
    return Piece(p.color[y0:y1, x0:x1].copy(), p.mask[y0:y1, x0:x1].copy())


class PieceTests(unittest.TestCase):
    def test_align_recovers_offset_even_when_cut(self):
        a = figure()
        b = crop(a, 17, 0, 120, 200)              # 왼쪽이 화면 끝에 잘린 같은 인물
        self.assertEqual(align(a, b), (17, 0))
        self.assertLess(identity_distance(a.at(0.5), b.at(0.5), (8, 0)), 0.05)

    def test_identity_and_expression_distances(self):
        a, b = figure(head=0), figure(head=6)
        other = figure(color=(200, 60, 60), seed=3)
        o = align(a.at(0.5), b.at(0.5))
        self.assertLess(identity_distance(a.at(0.5), b.at(0.5), o), 0.08)
        self.assertGreater(identity_distance(a.at(0.5), other.at(0.5), align(a.at(0.5), other.at(0.5)) or (0, 0)), 0.2)
        self.assertGreater(expression_distance(a, b, (0, 0), (0, 80), 3), 30)
        self.assertEqual(expression_distance(a, figure(head=0), (0, 0), (0, 80), 3), 0.0)

    def test_cluster_complete_and_apart(self):
        d = {(0, 1): 0.1, (0, 2): 0.9, (1, 2): 0.2}
        dist = lambda i, j: d[(min(i, j), max(i, j))]
        self.assertEqual(sorted(map(sorted, cluster([0, 1, 2], dist, 0.5))), [[0, 1, 2]])
        self.assertEqual(sorted(map(sorted, cluster([0, 1, 2], dist, 0.5, complete=True))), [[0, 1], [2]])
        self.assertEqual(sorted(map(sorted, cluster([0, 1, 2], dist, 0.5, apart=lambda i, j: {i, j} == {0, 1}))),
                         [[0], [1, 2]])

    def test_split_blob_into_front_and_rest(self):
        back, front = figure(color=(200, 80, 40), seed=1), figure(color=(40, 160, 90), seed=2)
        w = 200
        color = np.zeros((200, w, 3), np.float32)
        mask = np.zeros((200, w), bool)
        for p, x in ((back, 70), (front, 10)):       # 앞 인물을 나중에 그린다
            m = p.mask
            color[:, x:x + 120][m] = p.color[m]
            mask[:, x:x + 120] |= m
        blob = Piece(color, mask)
        at = align(blob, front, min_overlap=0.8)
        self.assertEqual(at, (10, 0))
        f, rest = split_blob(blob, front, at, 1.0)
        self.assertGreater(f.sum(), 0.9 * front.area)
        self.assertGreater(rest.sum(), 0.15 * blob.area)
        self.assertFalse((f & rest).any())


@needs_runtime
class OverlayTests(unittest.TestCase):
    def test_overlay_adds_and_replaces_dimensions(self):
        tmp = tempfile.mkdtemp()
        try:
            out = write_overlay(TUNING, os.path.join(tmp, "tuning"), {("c01", "a", "01"): (640, 1230),
                                                                       ("harin", "a", "01"): (10, 20)},
                                {"bg01": (1280, 720)})
            p = json.load(open(os.path.join(out, "portrait-dimensions.json"), encoding="utf-8"))
            harin = [e for e in p["entries"] if (e["character"], e["variant"], e["emotion"]) == ("harin", "a", "01")]
            self.assertEqual([(e["width"], e["height"]) for e in harin], [(10.0, 20.0)])
            self.assertIn({"character": "c01", "variant": "a", "emotion": "01", "width": 640.0, "height": 1230.0},
                          p["entries"])
            b = json.load(open(os.path.join(out, "background-dimensions.json"), encoding="utf-8"))
            self.assertIn({"key": "bg01", "width": 1280.0, "height": 720.0}, b["entries"])
            self.assertTrue(os.path.isfile(os.path.join(out, "ease-golden.json")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class MotionDraftTests(unittest.TestCase):
    def test_hop_becomes_track_move_y_with_peak(self):
        t = np.arange(0, 1.0, 1 / 30)
        hop = np.where((t >= 0.2) & (t <= 0.7), np.sin(np.clip((t - 0.2) / 0.5, 0, 1) * np.pi), 0.0)
        ds = 1.7
        sample = {"times": list(t), "dx": [0.0] * len(t), "dy": list(-hop * 30 * ds), "depthScale": ds,
                  "cameraScale": 1.0}
        m = to_motion("draft_l003_s1", sample)
        self.assertEqual([c["node"] for c in m["channels"]], ["CharacterPortrait_Track_Move_Y"])
        self.assertEqual((m["rig"], m["end"]), ("character", "restore"))
        self.assertAlmostEqual(peak(m)[1], 30.0, delta=1.0)
        keys = m["channels"][0]["y"]["keys"]
        self.assertEqual((keys[0]["v"], keys[-1]["v"]), (0.0, 0.0))
        self.assertLess(len(keys), len(t) // 2)          # 키를 줄였다
        self.assertIsNone(to_motion("none", dict(sample, dy=[0.2] * len(t))))
        tmp = tempfile.mkdtemp()
        try:
            doc = json.load(open(write_authoring(os.path.join(tmp, "v.motion-authoring.json"), [m]), encoding="utf-8"))
            self.assertEqual(doc["schema"], "motion-authoring/2")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@needs_oracle
class P4ScenarioTests(unittest.TestCase):
    def test_unknown_game_roundtrip(self):
        from ked_staging.__main__ import main
        tmp = tempfile.mkdtemp()
        try:
            code = main(["p4-check", "--out", tmp])
            report = json.load(open(os.path.join(tmp, "report.json"), encoding="utf-8"))
            self.assertEqual(code, 0, ({k: v for k, v in report.items() if k != "lines"},
                                       [r for r in report["lines"] if r["state"]]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
