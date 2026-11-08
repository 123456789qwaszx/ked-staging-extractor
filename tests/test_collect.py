"""원본 수집 — 가짜 위키(네트워크 없음)로 manifest 만들기를 시험한다. 이름은 지어낸 것이다."""
import json
import os
import shutil
import tempfile
import unittest
import urllib.parse

from ked_staging import collect

API = "https://wiki.example/w/api.php"


def info(name, size=100, sha1=None):
    return {"url": f"https://static.example/{name}", "descriptionurl": f"https://wiki.example/wiki/File:{name}",
            "size": size, "width": 10, "height": 20, "mime": "image/png", "sha1": sha1 or f"sha-{name}",
            "timestamp": "2026-01-01T00:00:00Z"}


class FakeWiki:
    """카테고리 → 하위 카테고리 · 파일. 파일 목록은 page 개씩 나눠 continue로 준다."""

    def __init__(self, subcats, files, page=2, failures=None):
        self.subcats = subcats
        self.files = files
        self.page = page
        self.failures = list(failures or [])
        self.urls = []

    def __call__(self, url, headers):
        self.urls.append(url)
        assert headers["User-Agent"] == "test-agent"
        if self.failures:
            return self.failures.pop(0)
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        assert q["format"] == "json" and q["maxlag"] == "5"
        if q.get("list") == "categorymembers":
            body = {"query": {"categorymembers": [{"title": t} for t in self.subcats.get(q["cmtitle"], [])]}}
        else:
            names = self.files.get(q["gcmtitle"], [])
            start = int(q.get("gcmcontinue", "0"))
            chunk = names[start:start + self.page]
            body = {"query": {"pages": [{"title": "File:" + n, **({"imageinfo": [info(n.replace(" ", "_"))]}
                                                                  if not n.startswith("!") else {})}
                                        for n in chunk]}}
            if start + self.page < len(names):
                body["continue"] = {"gcmcontinue": str(start + self.page), "continue": "gcmcontinue||"}
        return 200, {}, json.dumps(body).encode()


CONFIG = {
    "api": API,
    "sources": [
        {"kind": "sprites", "category": "Category:Sprites", "subcategories": True, "identitySuffix": " sprites",
         "parse": "sprite"},
        {"kind": "backgrounds", "category": "Category:Backgrounds"},
    ],
}


def wiki(**kw):
    return FakeWiki(
        subcats={"Category:Sprites": ["Category:Mira sprites", "Category:*** sprites", "Category:Mira Bright Ver sprites",
                                      "Category:Alina sprites", "Category:Firstlight Noya sprites"]},
        files={
            "Category:Mira sprites": ["Mira a 02.png", "Mira a 03.png", "Mira b 02.png", "Mira portrait.png",
                                      "Mira awakened 02.png", "Mira Bunny Girl 3.png", "Mira aa 02.png"],
            "Category:*** sprites": ["*** a 02.png"],
            "Category:Mira Bright Ver sprites": ["Mira Bright Ver a 02.png", "Mira a 02.png"],
            "Category:Alina sprites": ["Alina a 02.png", "Firstlight Noya a 02.png"],     # 위키가 잘못 단 카테고리
            "Category:Firstlight Noya sprites": ["Firstlight Noya b 02.png"],
            "Category:Backgrounds": ["BG park dusk.png", "BG park night.png", "!broken.png"],
        }, **kw)


def api(transport, sleeps=None):
    return collect.Api(API, "test-agent", delay=0, transport=transport,
                       sleep=(sleeps.append if sleeps is not None else (lambda s: None)), log=lambda m: None)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.m = collect.build_manifest(CONFIG, api(wiki()), log=lambda m: None)
        self.by = {f["title"]: f for f in self.m["files"]}

    def test_follows_continue_and_subcategories(self):
        self.assertEqual(self.m["summary"]["kinds"]["sprites"]["files"], 12)
        self.assertEqual(self.m["summary"]["kinds"]["backgrounds"]["files"], 2)
        self.assertIn("File:Mira portrait.png", self.by)          # 4개를 2개씩 — 두 번째 쪽도 읽었다

    def test_identity_is_the_category_not_the_file_name(self):
        f = self.by["File:Mira Bright Ver a 02.png"]
        self.assertEqual(f["identity"], "Mira Bright Ver")
        self.assertEqual(f["path"], "sprites/Mira Bright Ver/Mira_Bright_Ver_a_02.png")
        self.assertEqual(f["sprite"], {"name": "Mira Bright Ver", "variant": "a", "emotion": "02"})

    def test_provenance_is_kept_per_file(self):
        f = self.by["File:Mira a 02.png"]
        self.assertEqual((f["sourceType"], f["url"], f["sha1"], f["category"]),
                         ("mediawiki", "https://static.example/Mira_a_02.png", "sha-Mira_a_02.png", "Category:Mira sprites"))

    def test_windows_unsafe_names_are_replaced_but_the_original_name_stays(self):
        f = self.by["File:*** a 02.png"]
        self.assertEqual(f["path"], "sprites/___/____a_02.png")
        self.assertEqual(f["name"], "***_a_02.png")
        self.assertEqual(f["identity"], "***")

    def test_a_file_in_two_categories_belongs_to_the_one_its_name_names(self):
        # 정렬 순서로는 "Mira Bright Ver"가 먼저 열린다 — 그래도 주인은 이름이 가리키는 "Mira"
        self.assertEqual(self.by["File:Mira a 02.png"]["identity"], "Mira")

    def test_identity_comes_from_the_file_name_and_the_category_is_kept_as_provenance(self):
        f = self.by["File:Firstlight Noya a 02.png"]
        self.assertEqual((f["identity"], f["category"]), ("Firstlight Noya", "Category:Alina sprites"))
        self.assertEqual(f["path"], "sprites/Firstlight Noya/Firstlight_Noya_a_02.png")
        self.assertEqual(f["sprite"], {"name": "Firstlight Noya", "variant": "a", "emotion": "02"})
        self.assertEqual(self.m["mismatched"], [{"title": "File:Firstlight Noya a 02.png", "identity": "Firstlight Noya",
                                                 "category": "Category:Alina sprites"}])

    def test_resolve_identity_takes_the_longest_known_name(self):
        known = ["Mira", "Mira Bright Ver", "Bernina"]
        self.assertEqual(collect.resolve_identity("Mira Bright Ver a 02", known), "Mira Bright Ver")
        self.assertEqual(collect.resolve_identity("Mira a 02", known), "Mira")
        self.assertEqual(collect.resolve_identity("Bernina Bunny Girl 02", known), "Bernina")
        self.assertIsNone(collect.resolve_identity("Miracle a 02", known))

    def test_variants_can_be_words(self):
        self.assertEqual(self.by["File:Mira awakened 02.png"]["sprite"], {"name": "Mira", "variant": "awakened", "emotion": "02"})
        self.assertEqual(self.by["File:Mira Bunny Girl 3.png"]["sprite"], {"name": "Mira", "variant": "bunny_girl", "emotion": "03"})
        self.assertEqual(self.by["File:Mira aa 02.png"]["sprite"]["variant"], "aa")

    def test_nothing_is_dropped_silently(self):
        self.assertEqual([u["title"] for u in self.m["unparsed"]], ["File:Mira portrait.png"])
        self.assertIsNone(self.by["File:Mira portrait.png"]["sprite"])     # 규칙 밖이어도 받는다
        self.assertEqual(self.m["duplicates"], [{"title": "File:Mira a 02.png", "kept": "Category:Mira sprites",
                                                 "also": "Category:Mira Bright Ver sprites"}])
        self.assertEqual([p["title"] for p in self.m["problems"]], ["File:!broken.png"])

    def test_backgrounds_are_not_parsed_as_sprites(self):
        f = self.by["File:BG park dusk.png"]
        self.assertEqual(f["path"], "backgrounds/BG_park_dusk.png")
        self.assertNotIn("sprite", f)


class ApiTests(unittest.TestCase):
    def test_waits_for_retry_after_on_maxlag_and_429(self):
        maxlag = (200, {"retry-after": "7"}, json.dumps({"error": {"code": "maxlag", "info": "lag"}}).encode())
        busy = (429, {}, b"")
        w = FakeWiki({}, {"Category:Backgrounds": ["BG a.png"]}, failures=[maxlag, busy])
        sleeps = []
        files = collect.category_files(api(w, sleeps), "Category:Backgrounds")
        self.assertEqual([f["title"] for f in files], ["File:BG a.png"])
        self.assertEqual(sleeps, [7.0, 10.0])          # Retry-After 그대로 · 없으면 5 × 2^시도

    def test_other_errors_stop(self):
        w = FakeWiki({}, {}, failures=[(200, {}, json.dumps({"error": {"code": "badvalue", "info": "x"}}).encode())])
        with self.assertRaises(collect.ApiError):
            api(w).get({"action": "query"})
        w = FakeWiki({}, {}, failures=[(404, {}, b"")])
        with self.assertRaises(collect.ApiError):
            api(w).get({"action": "query"})

    def test_gives_up_after_retries(self):
        w = FakeWiki({}, {}, failures=[(503, {"retry-after": "1"}, b"")] * 10)
        a = collect.Api(API, "test-agent", delay=0, transport=w, sleep=lambda s: None, retries=2, log=lambda m: None)
        with self.assertRaises(collect.ApiError):
            a.get({"action": "query"})
        self.assertEqual(len(w.urls), 3)


class NameTests(unittest.TestCase):
    def test_safe_name(self):
        self.assertEqual(collect.safe_name("a:b?.png"), "a_b_.png")
        self.assertEqual(collect.safe_name("CON.png"), "_CON.png")
        self.assertEqual(collect.safe_name("x. "), "x")

    def test_parse_sprite(self):
        self.assertEqual(collect.parse_sprite("File:Olympia (Puppet) b 7.png"),
                         {"name": "Olympia (Puppet)", "variant": "b", "emotion": "07"})
        self.assertIsNone(collect.parse_sprite("File:Mira icon.png"))

    def test_parse_sprite_with_known_identity(self):
        self.assertEqual(collect.parse_sprite("File:Olympia (Puppet) a cbt cn1 02.png", "Olympia (Puppet)"),
                         {"name": "Olympia (Puppet)", "variant": "a_cbt_cn1", "emotion": "02"})
        self.assertIsNone(collect.parse_sprite("File:Mira a 02.png", "Nyx"))          # 다른 정체
        self.assertIsNone(collect.parse_sprite("File:Mira 02.png", "Mira"))           # 변형이 없다
        self.assertIsNone(collect.parse_sprite("File:Mira a  02.png", "Mira"))        # 빈 낱말


class DiffAndWriteTests(unittest.TestCase):
    def test_diff_reports_added_changed_removed(self):
        old = {"files": [{"title": "A", "sha1": "1"}, {"title": "B", "sha1": "2"}]}
        new = {"files": [{"title": "B", "sha1": "3"}, {"title": "C", "sha1": "4"}]}
        self.assertEqual(collect.diff(old, new), {"added": ["C"], "changed": ["B"], "removed": ["A"]})

    def test_write_is_atomic_and_round_trips(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "sub", "manifest.json")
            collect.write_json(path, {"files": [{"title": "가"}]})
            self.assertEqual(collect.load_json(path), {"files": [{"title": "가"}]})
            self.assertFalse(os.path.exists(path + ".part"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
