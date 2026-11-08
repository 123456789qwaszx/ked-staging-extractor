"""원본 수집 — MediaWiki 위키의 카테고리에서 원본 그림의 목록(manifest)을 만든다.

범용 수집기다. 위키 주소 · 카테고리 · 분류는 설정 파일이 정한다.

설정 파일
  {
    "api": "https://<위키>/w/api.php",
    "userAgent": "ked-originals/0.1",     도구 이름만 — 개인 연락처를 넣지 않는다
    "delay": 1.0,                         요청 사이 간격(초)
    "sources": [
      {"kind": "sprites", "category": "Category:…", "subcategories": true, "identitySuffix": " sprites",
       "parse": "sprite"},
      {"kind": "backgrounds", "category": "Category:…"},
      {"kind": "cg/main", "category": "Category:…"}
    ]
  }

  kind            보관소 안의 폴더(sprites · backgrounds · cg/main …)
  subcategories   true면 그 카테고리의 하위 카테고리 하나하나가 인물이다 — 하위 카테고리 이름들이 "알려진 정체"가 된다
  identitySuffix  하위 카테고리 이름에서 떼어 낼 꼬리("<이름> sprites" → "<이름>")
  parse "sprite"  파일 이름을 "<정체> <변형> <표정>"으로 푼다. 정체 = 파일 이름 앞부분과 가장 길게 맞는 알려진 정체,
                  마지막 숫자가 표정, 그 사이 전부가 변형이다(a · aa · awakened · skin1 · "Bunny Girl" …).
                  파일 이름은 게임 리소스 이름이고 위키 카테고리는 사람이 단 것이라 틀릴 수 있다(2026-11-08 소유자 결정) —
                  카테고리는 출처로 남기고, 정체와 다르면 mismatched로 보고한다. 못 풀면 버리지 않고 unparsed로 남긴다

manifest 한 줄이 그 파일의 출처 기록이다(sourceType "mediawiki" — 카탈로그 툴 문서의 공통 수집 결과).
로컬 파일 이름은 위키 파일 이름 그대로(공백 → _)이고, Windows에서 쓸 수 없는 글자만 _로 바꾼다(원래 이름은 name에 남는다).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Callable, Dict, Iterator, List, Optional, Tuple

FORMAT = "ked-originals-manifest/1"
IIPROP = "url|size|mime|sha1|timestamp"
SPRITE_TITLE = re.compile(r"^(?P<name>.+?) (?P<variant>[A-Za-z]) (?P<emotion>\d{1,3})$")   # 정체를 모를 때만
_EMOTION = re.compile(r"^\d{1,3}$")
_WINDOWS_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}

# (url, 헤더) → (상태 코드, 헤더(소문자 키), 본문)
Transport = Callable[[str, Dict[str, str]], Tuple[int, Dict[str, str], bytes]]


class ApiError(RuntimeError):
    pass


def urllib_transport(url: str, headers: Dict[str, str]) -> Tuple[int, Dict[str, str], bytes]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        body = e.read() if e.fp else b""
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, body


class Api:
    """MediaWiki API — 한 번에 하나씩, 요청 사이 간격, maxlag · 429 · 503이면 Retry-After만큼 기다렸다 다시."""

    def __init__(self, api: str, user_agent: str, delay: float = 1.0, transport: Optional[Transport] = None,
                 sleep: Callable[[float], None] = time.sleep, retries: int = 5, log: Callable[[str], None] = print):
        self.api = api
        self.user_agent = user_agent
        self.delay = delay
        self.transport = transport or urllib_transport
        self.sleep = sleep
        self.retries = retries
        self.log = log
        self.requests = 0
        self._last: Optional[float] = None

    def get(self, params: Dict[str, str]) -> dict:
        q = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        url = self.api + "?" + urllib.parse.urlencode(q)
        for attempt in range(self.retries + 1):
            if self._last is not None:
                gap = self.delay - (time.monotonic() - self._last)
                if gap > 0:
                    self.sleep(gap)
            status, headers, body = self.transport(url, {"User-Agent": self.user_agent})
            self._last = time.monotonic()
            self.requests += 1
            if status in (429, 503):
                reason = f"HTTP {status}"
            elif status != 200:
                raise ApiError(f"HTTP {status}: {url}")
            else:
                data = json.loads(body.decode("utf-8"))
                err = data.get("error")
                if not err:
                    return data
                if err.get("code") != "maxlag":
                    raise ApiError(f"{err.get('code')}: {err.get('info')} ({url})")
                reason = "서버 지연(maxlag)"
            if attempt == self.retries:
                break
            wait = _retry_after(headers, attempt)
            self.log(f"  {reason} — {wait:.0f}초 기다렸다 다시")
            self.sleep(wait)
        raise ApiError(f"{self.retries}번 다시 해도 실패: {url}")

    def query_all(self, params: Dict[str, str]) -> Iterator[dict]:
        """continue를 끝까지 따라가며 응답을 하나씩 내준다."""
        cont: Dict[str, str] = {}
        while True:
            data = self.get({"action": "query", **params, **cont})
            yield data
            if "continue" not in data:
                return
            cont = data["continue"]


def _retry_after(headers: Dict[str, str], attempt: int) -> float:
    try:
        return max(1.0, float(headers.get("retry-after", "")))
    except ValueError:
        return float(min(60, 5 * 2 ** attempt))


def subcategories(api: Api, category: str) -> List[str]:
    out: List[str] = []
    for data in api.query_all({"list": "categorymembers", "cmtitle": category, "cmtype": "subcat", "cmlimit": "500"}):
        out += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
    return sorted(out)


def category_files(api: Api, category: str) -> List[dict]:
    """카테고리의 파일 → [{title, imageinfo}] (제목 순). 응답이 나뉘어 와도 제목으로 합친다."""
    pages: Dict[str, dict] = {}
    for data in api.query_all({"generator": "categorymembers", "gcmtitle": category, "gcmtype": "file",
                               "gcmlimit": "500", "prop": "imageinfo", "iiprop": IIPROP}):
        for p in data.get("query", {}).get("pages", []):
            cur = pages.setdefault(p["title"], {"title": p["title"]})
            if p.get("imageinfo"):
                cur["imageinfo"] = p["imageinfo"][0]
    return [pages[t] for t in sorted(pages)]


def file_name(title: str) -> str:
    """"File:Aeloria a 02.png" → "Aeloria a 02.png"의 위키 저장 이름 "Aeloria_a_02.png"."""
    return title.split(":", 1)[1].replace(" ", "_")


def safe_name(name: str) -> str:
    """Windows에서 쓸 수 있는 이름 — 금지 글자만 _로, 끝의 점 · 공백을 떼고, 예약 이름이면 앞에 _."""
    s = _WINDOWS_BAD.sub("_", name).rstrip(". ") or "_"
    stem = s.split(".", 1)[0].upper()
    return "_" + s if stem in _WINDOWS_RESERVED else s


def identity(category: str, suffix: str = "") -> str:
    name = category.split(":", 1)[1] if ":" in category else category
    if suffix and name.endswith(suffix):
        name = name[: -len(suffix)]
    return name.strip()


def _stem(title: str) -> str:
    return os.path.splitext(title.split(":", 1)[1])[0]


def parse_sprite(title: str, who: Optional[str] = None) -> Optional[dict]:
    """"File:Mira Bunny Girl 02.png" + 정체 "Mira" → {name "Mira", variant "bunny_girl", emotion "02"}.
    정체를 모르면 한 글자 변형만 알아본다. 규칙 밖이면 None."""
    stem = _stem(title)
    if who:
        if not stem.startswith(who + " "):
            return None
        tokens = stem[len(who) + 1:].split(" ")
        if len(tokens) < 2 or not _EMOTION.match(tokens[-1]) or not all(tokens[:-1]):
            return None
        return {"name": who, "variant": "_".join(tokens[:-1]).lower(), "emotion": f"{int(tokens[-1]):02d}"}
    m = SPRITE_TITLE.match(stem)
    if not m:
        return None
    return {"name": m.group("name"), "variant": m.group("variant").lower(), "emotion": f"{int(m.group('emotion')):02d}"}


def resolve_identity(stem: str, known: List[str]) -> Optional[str]:
    """파일 이름 앞부분과 가장 길게 맞는 알려진 정체("Mira Bright Ver a 02" → "Mira Bright Ver", "Mira"보다 길다)."""
    hits = [k for k in known if stem.startswith(k + " ")]
    return max(hits, key=len) if hits else None


def build_manifest(config: dict, api: Api, log: Callable[[str], None] = print) -> dict:
    # 1) 모은다 — 제목마다 그 파일이 나온 카테고리 전부
    found: Dict[str, List[dict]] = {}
    order: List[str] = []
    problems: List[dict] = []
    known: Dict[int, List[str]] = {}        # 출처마다 알려진 정체
    for i, src in enumerate(config["sources"]):
        kind = src["kind"].strip("/")
        cats = subcategories(api, src["category"]) if src.get("subcategories") else [src["category"]]
        if src.get("subcategories"):
            known[i] = [identity(c, src.get("identitySuffix", "")) for c in cats]
        log(f"{kind}: 카테고리 {len(cats)}개")
        for n, cat in enumerate(cats, 1):
            who = identity(cat, src.get("identitySuffix", "")) if src.get("subcategories") else None
            for p in category_files(api, cat):
                ii = p.get("imageinfo")
                if not ii or not ii.get("url"):
                    problems.append({"title": p["title"], "category": cat, "reason": "원본 정보가 없다"})
                    continue
                if p["title"] not in found:
                    found[p["title"]] = []
                    order.append(p["title"])
                found[p["title"]].append({"src": src, "index": i, "kind": kind, "category": cat, "identity": who,
                                          "ii": ii})
            if src.get("subcategories") and (n % 20 == 0 or n == len(cats)):
                log(f"  {n}/{len(cats)} · 파일 {len(order)}개까지")

    # 2) 정체를 정한다 — 파일 이름이 정본(가장 긴 알려진 정체). 파일 이름으로 못 정하면 카테고리의 정체.
    #    여러 카테고리에 있으면 그 정체의 카테고리를, 없으면 처음 나온 곳을 출처 카테고리로 남긴다
    files: List[dict] = []
    paths: Dict[str, str] = {}
    duplicates: List[dict] = []
    unparsed: List[dict] = []
    mismatched: List[dict] = []
    for title in order:
        hits = found[title]
        first = hits[0]
        who = first["identity"]
        if first["index"] in known:
            who = resolve_identity(_stem(title), known[first["index"]]) or first["identity"]
        owner = next((h for h in hits if h["identity"] == who), first)
        sprite = parse_sprite(title, who)
        for h in hits:
            if h is not owner:
                duplicates.append({"title": title, "kept": owner["category"], "also": h["category"]})
        if owner["identity"] is not None and owner["identity"] != who:
            mismatched.append({"title": title, "identity": who, "category": owner["category"]})
        ii, kind = owner["ii"], owner["kind"]
        name = file_name(title)
        local = kind + ("/" + safe_name(who) if who is not None else "") + "/" + safe_name(name)
        if local.lower() in paths:       # Windows는 대소문자를 가리지 않는다
            problems.append({"title": title, "category": owner["category"],
                             "reason": f"로컬 경로가 겹친다: {local} ← {paths[local.lower()]}"})
            continue
        paths[local.lower()] = title
        entry = {
            "title": title, "name": name, "path": local, "kind": kind, "category": owner["category"],
            "identity": who, "sourceType": "mediawiki", "url": ii["url"],
            "page": ii.get("descriptionurl"), "sha1": ii.get("sha1"), "size": ii.get("size"),
            "width": ii.get("width"), "height": ii.get("height"), "mime": ii.get("mime"),
            "timestamp": ii.get("timestamp"),
        }
        if owner["src"].get("parse") == "sprite":
            entry["sprite"] = sprite
            if sprite is None:
                unparsed.append({"title": title, "category": owner["category"]})
        files.append(entry)
    return {
        "format": FORMAT,
        "api": config["api"],
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summarize(files),
        "files": files,
        "unparsed": unparsed,
        "mismatched": mismatched,
        "duplicates": duplicates,
        "problems": problems,
    }


def summarize(files: List[dict]) -> dict:
    kinds: Dict[str, dict] = {}
    for f in files:
        k = kinds.setdefault(f["kind"], {"files": 0, "bytes": 0})
        k["files"] += 1
        k["bytes"] += f.get("size") or 0
    return {"files": len(files), "bytes": sum(k["bytes"] for k in kinds.values()), "kinds": kinds}


def diff(old: dict, new: dict) -> dict:
    """앞 manifest와 견준다 — 새 파일 · 바뀐 파일(SHA-1이 다르다) · 위키에서 사라진 파일."""
    a = {f["title"]: f for f in old.get("files", [])}
    b = {f["title"]: f for f in new.get("files", [])}
    return {
        "added": sorted(t for t in b if t not in a),
        "changed": sorted(t for t in b if t in a and a[t].get("sha1") != b[t].get("sha1")),
        "removed": sorted(t for t in a if t not in b),
    }


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: dict) -> None:
    """원자적으로 — 임시 파일에 다 쓴 뒤 바꿔 끼운다(쓰다 끊겨도 앞 파일이 남는다)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n}B"
