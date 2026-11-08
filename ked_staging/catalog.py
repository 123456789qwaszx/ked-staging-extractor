"""카탈로그 — "무엇이 화면에 설 수 있나". 두 모드가 같은 형식을 쓴다(설계 문서 '데이터 형식').

런타임 모드는 원본 리소스를 그대로 가리킨다(규칙 1 — 원본이 있으면 원본으로, 복사하지 않는다).
다른 게임 모드는 카탈로그 툴이 같은 형식의 파일을 쓴다(P4).

catalog.json
{
  "schema": "ked-staging-catalog/1",
  "source": "runtime" | "video",
  "root": "<경로들이 기대는 폴더>",
  "portraits": [{"character","variant","emotion","path","size":[w,h],"opaqueBox":[x,y,w,h]}],
  "backgrounds": [{"key","path","size":[w,h]}]
}
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .imageio import read_image

SCHEMA = "ked-staging-catalog/1"
IMAGE_CACHE = 24          # 원래 크기 그림을 이만큼만 쥔다(원본 초상 수십 장이면 1GB를 넘는다) — 줄인 그림은 Painter가 따로 쥔다

# 런타임 저장소 안에서 리소스를 찾는 자리 (ResourcesSpriteResolver 규약: 초상 <캐릭터>/<변형>/<표정>.png, 배경 Backgrounds/<키>.png)
RUNTIME_PORTRAIT_DIRS = ["Assets/Samples/Content/Portraits", "Assets/Samples/Portraits"]
RUNTIME_BACKGROUND_DIRS = ["Assets/Samples/Content/Resources/Backgrounds", "Assets/Resources/Backgrounds"]


def emotion_code(value: str) -> str:
    """코어 PortraitKeyNormalizer.EmotionCode와 같다 — 숫자 한두 자리를 두 자리로. 아니면 빈 문자열."""
    v = (value or "").strip()
    if len(v) == 2 and v.isdigit():
        return v
    if len(v) == 1 and v.isdigit():
        return "0" + v
    return ""


@dataclass
class Portrait:
    character: str
    variant: str
    emotion: str
    path: str
    size: Tuple[int, int]
    opaque_box: Tuple[int, int, int, int]


@dataclass
class Background:
    key: str
    path: str
    size: Tuple[int, int]


@dataclass
class Catalog:
    source: str
    root: str
    portraits: List[Portrait] = field(default_factory=list)
    backgrounds: List[Background] = field(default_factory=list)
    _cache: "OrderedDict[str, np.ndarray]" = field(default_factory=OrderedDict, repr=False)

    # ── 조회 ──
    def portrait(self, character: str, variant: str, emotion: str) -> Optional[Portrait]:
        c, v, e = (character or "").lower(), (variant or "a").lower(), emotion_code(emotion)
        for p in self.portraits:
            if p.character == c and p.variant == v and p.emotion == e:
                return p
        return None

    def background(self, key: str) -> Optional[Background]:
        for b in self.backgrounds:
            if b.key == key:
                return b
        return None

    def image(self, entry) -> np.ndarray:
        """BGRA 이미지(캐시)."""
        path = self.abspath(entry.path)
        img = self._cache.get(path)
        if img is None:
            img = read_image(path, alpha=True)
            if img is None:
                raise FileNotFoundError(path)
            self._cache[path] = img
            while len(self._cache) > IMAGE_CACHE:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(path)
        return img

    def abspath(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(self.root, path))

    # ── 저장 · 읽기 ──
    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "source": self.source,
            "root": self.root,
            "portraits": [
                {"character": p.character, "variant": p.variant, "emotion": p.emotion, "path": p.path,
                 "size": list(p.size), "opaqueBox": list(p.opaque_box)} for p in self.portraits],
            "backgrounds": [{"key": b.key, "path": b.path, "size": list(b.size)} for b in self.backgrounds],
        }

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Catalog":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{SCHEMA} 형식이 아닙니다: {data.get('schema')!r}")
        root = data.get("root") or os.path.dirname(os.path.abspath(path))
        cat = cls(data["source"], root)
        cat.portraits = [Portrait(p["character"], p["variant"], p["emotion"], p["path"], tuple(p["size"]),
                                  tuple(p["opaqueBox"])) for p in data.get("portraits", [])]
        cat.backgrounds = [Background(b["key"], b["path"], tuple(b["size"])) for b in data.get("backgrounds", [])]
        return cat


def import_runtime(runtime_root: str, problems: Optional[List[str]] = None) -> Catalog:
    """ked-presentation-runtime 저장소에서 카탈로그를 만든다. 경로는 저장소 기준 상대 경로로 둔다."""
    problems = problems if problems is not None else []
    root = os.path.abspath(runtime_root)
    cat = Catalog("runtime", root)
    for rel in RUNTIME_PORTRAIT_DIRS:
        base = os.path.join(root, rel)
        if not os.path.isdir(base):
            continue
        for character in sorted(os.listdir(base)):
            cdir = os.path.join(base, character)
            if not os.path.isdir(cdir):
                continue
            for variant in sorted(os.listdir(cdir)):
                vdir = os.path.join(cdir, variant)
                if not os.path.isdir(vdir):
                    continue
                for name in sorted(os.listdir(vdir)):
                    stem, ext = os.path.splitext(name)
                    if ext.lower() != ".png":
                        continue
                    code = emotion_code(stem)
                    if not code:
                        problems.append(f"표정 코드가 아닌 파일은 건너뜁니다: {os.path.join(rel, character, variant, name)}")
                        continue
                    path = os.path.join(vdir, name)
                    img = read_image(path, alpha=True)
                    if img is None:
                        problems.append(f"읽지 못했습니다: {path}")
                        continue
                    ys, xs = np.nonzero(img[:, :, 3] > 8)
                    box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) \
                        if len(xs) else (0, 0, 0, 0)
                    if cat.portrait(character, variant, code):
                        problems.append(f"같은 초상이 두 폴더에 있습니다(앞의 것을 씁니다): {character}/{variant}/{code}")
                        continue
                    cat.portraits.append(Portrait(character.lower(), variant.lower(), code,
                                                  os.path.relpath(path, root), (img.shape[1], img.shape[0]), box))
    for rel in RUNTIME_BACKGROUND_DIRS:
        base = os.path.join(root, rel)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            path = os.path.join(base, name)
            img = read_image(path)
            if img is None:
                problems.append(f"읽지 못했습니다: {path}")
                continue
            if cat.background(stem):
                problems.append(f"같은 배경 키가 두 폴더에 있습니다(앞의 것을 씁니다): {stem}")
                continue
            cat.backgrounds.append(Background(stem, os.path.relpath(path, root), (img.shape[1], img.shape[0])))
    if not cat.portraits:
        problems.append("초상을 하나도 찾지 못했습니다 — RUNTIME_PORTRAIT_DIRS를 확인하세요.")
    if not cat.backgrounds:
        problems.append("배경을 하나도 찾지 못했습니다 — RUNTIME_BACKGROUND_DIRS를 확인하세요.")
    return cat
