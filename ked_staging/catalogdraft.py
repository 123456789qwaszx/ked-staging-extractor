"""다른 게임(P4) — 초안 고치기(DraftEditor) · 초안 → 카탈로그 + 튜닝 겹본(build_catalog).

초안(draft.json, ked-staging-draft/1)은 scan이 쓰고, 사람이 컴포지터의 카탈로그 툴(Vn.CatalogTool —
Vn.Authoring/Catalog/CatalogDraft) 또는 `draft` 명령으로 고친다. 두 쪽 연산의 뜻은 같다(docs/P4.md가 계약).
JSON 객체를 직접 고쳐 모르는 필드는 그대로 남긴다.

catalog-build
  초상    런타임은 그림의 종횡비만 쓴다(preserveAspect로 슬롯 사각형 높이에 맞춰 가운데에 그린다). 그래서
          기준 자리(place · bust · mid) 사각형 높이 × 영상 배율을 그림 높이로 두고, 가장 자주 선 자리가
          가장 가까운 place_* 에 정확히 오게 투명 여백을 두른다. 머리가 사각형 위로 넘치면 front · close를 쓴다.
  몸      그 인물의 모든 조각을 맞대어 겹친 중앙값으로 채운다(잘리거나 가린 자리도 메운다).
  표정    몸 그림에서 머리 쪽만 그 표정의 대표 조각으로 바꾼다 — 몸까지 덮으면 조각마다의 반 화소 어긋남 · 가림이
          표정 차이보다 커져 관측의 표정 고르기를 흔든다.
  튜닝    런타임 ExportedTuning을 복사하고 치수표(portrait- · background-dimensions)에 새 그림을 더한다.
          오라클은 이 겹본으로 새 인물을 접는다 — 오라클 코드는 그대로다.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .catalog import Background, Catalog, Portrait
from .imageio import read_image, write_image
from .scan import HEAD_PART, SCHEMA, Piece, align, load_piece

NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
VARIANT = re.compile(r"^[a-z]$")
DEPTH_TRY = ("mid", "front", "close")     # 머리가 넘치면 다음 것
PLACES = ("left", "center", "right")


class DraftError(ValueError):
    pass


def check_name(name: str) -> str:
    n = (name or "").strip().lower()
    if not NAME.match(n):
        raise DraftError(f"이름은 영문 소문자로 시작하고 영문 소문자 · 숫자 · _만 쓴다(32자까지): '{name}'")
    return n


def emotion_code(code) -> str:
    v = str(code if code is not None else "").strip()
    if v.isdigit() and 1 <= int(v) <= 99:
        return f"{int(v):02d}"
    raise DraftError(f"표정 번호는 1~99: '{code}'")


def _num(e: str) -> int:
    return int(e) if (e or "").isdigit() else 0


# ── 고치기 ────────────────────────────────────────────────────

class DraftEditor:
    """초안 고치기 — 컴포지터 CatalogDraft와 같은 연산(이름 · 합치기 · 옮기기 · 빼기 · 대표 · 배경 · 되돌리기)."""

    def __init__(self, data: dict, path: str):
        if data.get("schema") != SCHEMA:
            raise DraftError(f"{SCHEMA} 형식이 아니다: '{data.get('schema')}'")
        self.data = data
        self.path = path
        self._undo: List[dict] = []

    @classmethod
    def load(cls, path: str) -> "DraftEditor":
        if os.path.isdir(path):
            path = os.path.join(path, "draft.json")
        with open(path, encoding="utf-8-sig") as f:
            return cls(json.load(f), path)

    @property
    def folder(self) -> str:
        return os.path.dirname(os.path.abspath(self.path))

    def abspath(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(self.folder, rel))

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)
            f.write("\n")
        os.replace(tmp, path)
        return path

    # 조회
    @property
    def observations(self) -> List[dict]:
        return self.data.setdefault("observations", [])

    @property
    def backgrounds(self) -> List[dict]:
        return self.data.setdefault("backgrounds", [])

    @property
    def live(self) -> List[dict]:
        return [o for o in self.observations if not o.get("excluded")]

    def characters(self) -> List[str]:
        return sorted({o["character"] for o in self.live})

    def groups(self) -> List[Tuple[str, str, str, List[dict]]]:
        g: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
        for o in self.live:
            g[(o["character"], o.get("variant", "a"), o.get("emotion", "01"))].append(o)
        return [(c, v, e, g[(c, v, e)]) for c, v, e in sorted(g)]

    def find(self, oid: str) -> dict:
        for o in self.observations:
            if o.get("id") == oid:
                return o
        raise DraftError(f"조각이 없다: {oid}")

    def next_emotion(self, character: str, variant: str) -> str:
        used = {_num(o.get("emotion", "")) for o in self.observations
                if o["character"] == character and o.get("variant", "a") == variant}
        k = 1
        while k in used:
            k += 1
        return emotion_code(k)

    def representative(self, character: str, variant: str, emotion: str) -> Optional[dict]:
        group = [o for o in self.live if o["character"] == character and o.get("variant", "a") == variant
                 and o.get("emotion", "01") == emotion]
        marked = [o for o in group if o.get("representative")]
        return marked[0] if marked else (max(group, key=lambda o: o.get("area", 0)) if group else None)

    # 되돌리기
    def _snapshot(self) -> None:
        self._undo.append(copy.deepcopy(self.data))
        del self._undo[:-50]

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.data = self._undo.pop()
        return True

    # 고치기
    def rename_character(self, old: str, new: str) -> None:
        name = check_name(new)
        if name != old and any(o["character"] == name for o in self.observations):
            raise DraftError(f"이미 있는 이름이다 — 합치려면 '합치기'를 쓴다: {name}")
        self._snapshot()
        for o in self.observations:
            if o["character"] == old:
                o["character"] = name

    def merge_characters(self, source: str, target: str, as_variant: Optional[str] = None) -> None:
        """source를 target으로. as_variant가 없으면 같은 변형으로(표정 번호는 target 뒤로 이어 붙인다),
        있으면 source의 변형 a를 그 변형으로(옷이 다른 같은 인물 — 표정 번호는 그대로)."""
        if source == target:
            return
        variant = None
        if as_variant is not None and str(as_variant).strip():
            variant = str(as_variant).strip().lower()
            if not VARIANT.match(variant):
                raise DraftError(f"변형은 영문 소문자 한 자: '{as_variant}'")
        self._snapshot()
        moved = [o for o in self.observations if o["character"] == source]
        if variant is not None:
            for o in moved:
                o["variant"] = variant if o.get("variant", "a") == "a" else o.get("variant", "a")
                o["character"] = target
            return
        top: Dict[str, int] = defaultdict(int)
        for o in self.observations:
            if o["character"] == target:
                v = o.get("variant", "a")
                top[v] = max(top[v], _num(o.get("emotion", "")))
        remap: Dict[Tuple[str, str], str] = {}
        for o in sorted(moved, key=lambda o: (o.get("variant", "a"), o.get("emotion", "01"))):
            key = (o.get("variant", "a"), o.get("emotion", "01"))
            if key not in remap:
                top[key[0]] += 1
                remap[key] = emotion_code(top[key[0]])
            o["character"] = target
            o["emotion"] = remap[key]

    def move(self, ids: Iterable[str], character: str, variant: str = "a", emotion: Optional[str] = None) -> str:
        """조각들을 (인물, 변형, 표정)으로. 표정이 비면 새 번호. 새 인물 이름도 된다. → 쓴 표정 번호."""
        name = check_name(character)
        v = (variant or "a").strip().lower() or "a"
        if not VARIANT.match(v):
            raise DraftError(f"변형은 영문 소문자 한 자: '{variant}'")
        targets = [self.find(i) for i in ids]
        self._snapshot()
        e = self.next_emotion(name, v) if emotion is None or not str(emotion).strip() else emotion_code(emotion)
        for o in targets:
            o["character"], o["variant"], o["emotion"], o["representative"] = name, v, e, False
        return e

    def merge_emotions(self, character: str, variant: str, source: str, target: str) -> None:
        t = emotion_code(target)
        self._snapshot()
        for o in self.observations:
            if o["character"] == character and o.get("variant", "a") == variant and o.get("emotion") == source:
                o["emotion"], o["representative"] = t, False

    def exclude(self, ids: Iterable[str], excluded: bool = True) -> None:
        targets = [self.find(i) for i in ids]
        self._snapshot()
        for o in targets:
            o["excluded"] = excluded

    def set_representative(self, oid: str) -> None:
        t = self.find(oid)
        self._snapshot()
        for o in self.observations:
            if (o["character"], o.get("variant", "a"), o.get("emotion")) == (t["character"], t.get("variant", "a"),
                                                                              t.get("emotion")):
                o["representative"] = o["id"] == oid

    def rename_background(self, old: str, new: str) -> None:
        key = check_name(new)
        if key != old and any(b.get("key") == key for b in self.backgrounds):
            raise DraftError(f"이미 있는 배경 이름이다: {key}")
        self._snapshot()
        for b in self.backgrounds:
            if b.get("key") == old:
                b["key"] = key
        for line in self.data.get("lines", []):
            if line.get("background") == old:
                line["background"] = key
        for o in self.observations:
            if o.get("background") == old:
                o["background"] = key

    def exclude_background(self, key: str, excluded: bool = True) -> None:
        self._snapshot()
        for b in self.backgrounds:
            if b.get("key") == key:
                b["excluded"] = excluded


# ── 카탈로그 만들기 ───────────────────────────────────────────

def _premultiplied_warp(rgba: np.ndarray, m: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    pre = np.concatenate([rgba[:, :, :3].astype(np.float32) * a, a * 255.0], axis=2)
    out = cv2.warpAffine(pre, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = out[:, :, 3:4] / 255.0
    color = out[:, :, :3] / np.maximum(alpha, 1e-6)
    res = np.concatenate([color, out[:, :, 3:4]], axis=2)
    res[alpha[:, :, 0] < 1e-3] = 0
    return np.clip(res + 0.5, 0, 255).astype(np.uint8)


def _paste(canvas: np.ndarray, piece: Piece, x: int, y: int) -> None:
    """piece의 불투명 화소를 canvas(BGRA float32) (x, y)에 덮어쓴다."""
    h, w = piece.mask.shape
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    m = piece.mask[y0 - y:y1 - y, x0 - x:x1 - x]
    region = canvas[y0:y1, x0:x1]
    region[m, :3] = piece.color[y0 - y:y1 - y, x0 - x:x1 - x][m]
    region[m, 3] = 255.0


class _Identity:
    """한 (인물, 변형)의 조각들을 한 좌표(기준 조각의 왼쪽 위가 원점)에 맞댄 것."""

    def __init__(self, ed: DraftEditor, character: str, variant: str, members: List[dict], log):
        self.character, self.variant = character, variant
        self.members = members
        self.pieces = {o["id"]: load_piece(ed.abspath(o["png"])) for o in members}
        whole = [o for o in members if not o.get("edge") and not o.get("split")] or members
        ref = max(whole, key=lambda o: o.get("area", 0))
        self.ref = ref["id"]
        self.at: Dict[str, Tuple[int, int]] = {self.ref: (0, 0)}
        for o in members:
            if o["id"] == self.ref:
                continue
            found = align(self.pieces[self.ref], self.pieces[o["id"]], scale=1.0)
            if found is None:
                log(f"  {character}/{variant}: 조각 {o['id']}를 맞대지 못해 뺍니다")
                continue
            self.at[o["id"]] = found
        xs0 = [self.at[i][0] for i in self.at]
        ys0 = [self.at[i][1] for i in self.at]
        xs1 = [self.at[i][0] + self.pieces[i].mask.shape[1] for i in self.at]
        ys1 = [self.at[i][1] + self.pieces[i].mask.shape[0] for i in self.at]
        self.origin = (min(xs0), min(ys0))
        self.size = (max(xs1) - self.origin[0], max(ys1) - self.origin[1])

    def local(self, oid: str) -> Tuple[int, int]:
        x, y = self.at[oid]
        return x - self.origin[0], y - self.origin[1]

    def body(self, video_size: Tuple[int, int], stage_bottom: int) -> np.ndarray:
        """모든 조각을 겹친 몸(BGRA float32). 색은 불투명한 조각의 중앙값, 투명도는 그 자리를 본 조각의 다수결."""
        W, H = self.size
        colors = np.full((len(self.at), H, W, 3), np.nan, np.float32)
        opaque = np.zeros((H, W), np.float32)
        seen = np.zeros((H, W), np.float32)
        by_id = {o["id"]: o for o in self.members}
        for k, oid in enumerate(self.at):
            p, o = self.pieces[oid], by_id[oid]
            x, y = self.local(oid)
            h, w = p.mask.shape
            colors[k, y:y + h, x:x + w][p.mask] = p.color[p.mask]
            opaque[y:y + h, x:x + w] += p.mask
            if o.get("split"):
                seen[y:y + h, x:x + w] += p.mask          # 가린 자리는 모른다 — 보인 자리만 센다
                continue
            # 그 줄 화면에서 이 좌표가 영상 무대 안이었나(화면 끝에 잘린 자리는 모른다).
            bx, by = o["bbox"][0] - x, o["bbox"][1] - y
            gx0, gy0 = max(0, -bx), max(0, -by)
            gx1, gy1 = min(W, video_size[0] - bx), min(H, stage_bottom - by)
            if gx1 > gx0 and gy1 > gy0:
                seen[gy0:gy1, gx0:gx1] += 1
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(colors, axis=0)
        alpha = (opaque > 0) & (opaque >= 0.5 * np.maximum(seen, 1))
        out = np.zeros((H, W, 4), np.float32)
        out[:, :, :3] = np.nan_to_num(med)
        out[:, :, 3] = alpha * 255.0
        return out

    def stance(self) -> Tuple[float, float]:
        """가장 자주 선 자리 — 좌표 원점이 영상 어디에 왔나(영상 화소). 3px 안은 같은 자리로 본다."""
        spots = []
        for o in self.members:
            if o["id"] not in self.at:
                continue
            x, y = self.local(o["id"])
            spots.append((o["bbox"][0] - x, o["bbox"][1] - y))
        best = None
        for s in spots:
            near = [t for t in spots if abs(t[0] - s[0]) <= 3 and abs(t[1] - s[1]) <= 3]
            if best is None or len(near) > len(best):
                best = near
        return float(np.median([t[0] for t in best])), float(np.median([t[1] for t in best]))


def _slot_line(character: str, variant: str, place: str, focus: str, depth: str) -> List[dict]:
    return [{"name": "slot", "args": ["s1", "stage00", "mid"]},
            {"name": "cast", "args": ["s1", character, variant, "1"]},
            {"name": f"place_{place}", "args": ["s1", focus, "0fr"]},
            {"name": "size", "args": ["s1", depth, focus, "0fr"]},
            {"name": "fade_in", "args": ["s1", "0fr"]}]


def _slot(oracle, character: str, variant: str, place: str, focus: str, depth: str) -> dict:
    snap = oracle.fold([_slot_line(character, variant, place, focus, depth)])[0]
    return next(s for s in snap.slots if s["key"] == "s1")


def _dims_json(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def write_overlay(runtime_tuning: str, out_dir: str, portraits: Dict[Tuple[str, str, str], Tuple[int, int]],
                  backgrounds: Dict[str, Tuple[int, int]]) -> str:
    """런타임 ExportedTuning 복사본에 새 그림의 치수를 더한다(같은 이름은 바꾼다)."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(runtime_tuning, out_dir)
    p = os.path.join(out_dir, "portrait-dimensions.json")
    data = _dims_json(p)
    entries = [e for e in data.get("entries", []) if (e.get("character"), e.get("variant"), e.get("emotion"))
               not in portraits]
    for (c, v, e), (w, h) in sorted(portraits.items()):
        entries.append({"character": c, "variant": v, "emotion": e, "width": float(w), "height": float(h)})
    data["entries"] = entries
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    p = os.path.join(out_dir, "background-dimensions.json")
    data = _dims_json(p)
    entries = [e for e in data.get("entries", []) if e.get("key") not in backgrounds]
    for k, (w, h) in sorted(backgrounds.items()):
        entries.append({"key": k, "width": float(w), "height": float(h)})
    data["entries"] = entries
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return out_dir


def build_catalog(draft_path: str, out_dir: str, runtime_tuning: str, oracle_command: Optional[Sequence[str]] = None,
                  log: Callable[[str], None] = print) -> Tuple[str, str]:
    """초안 → out_dir/catalog.json(+ portraits/ · backgrounds/) · out_dir/tuning(겹본). → (카탈로그 경로, 튜닝 폴더)."""
    from .observe import depth_presets
    from .oracle import Oracle

    ed = DraftEditor.load(draft_path)
    d = ed.data
    vw, vh = (int(x) for x in d["videoSize"])
    bottom = int(d["stageBottom"])
    os.makedirs(out_dir, exist_ok=True)
    cat = Catalog("video", os.path.abspath(out_dir))
    bg_dims: Dict[str, Tuple[int, int]] = {}
    for b in ed.backgrounds:
        if b.get("excluded"):
            continue
        img = read_image(ed.abspath(b["plate"]))
        if img is None:
            raise FileNotFoundError(b["plate"])
        rel = os.path.join("backgrounds", f"{b['key']}.png")
        write_image(os.path.join(out_dir, rel), img)
        cat.backgrounds.append(Background(b["key"], rel, (img.shape[1], img.shape[0])))
        bg_dims[b["key"]] = (img.shape[1], img.shape[0])

    idents: List[_Identity] = []
    by_cv: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for o in ed.live:
        by_cv[(o["character"], o.get("variant", "a"))].append(o)
    for (c, v), members in sorted(by_cv.items()):
        idents.append(_Identity(ed, c, v, members, log))

    tuning = os.path.join(out_dir, "tuning")
    depths_have = depth_presets(runtime_tuning)
    depth_try = [x for x in DEPTH_TRY if x in depths_have] or depths_have[:1]
    # 1단: 자리를 고르고 그림 크기를 정한다(사각형은 종횡비와 무관하다 — 임시 치수로 잰다).
    temp = {(i.character, i.variant, "01"): (100, 100) for i in idents}
    write_overlay(runtime_tuning, tuning, temp, bg_dims)
    plans = []
    with Oracle(tuning, command=oracle_command) as oracle:
        v = vh / float(oracle.height)
        for ident in idents:
            body = ident.body((vw, vh), bottom)
            rows = np.nonzero(body[:, :, 3].max(1) > 0)[0]
            cols = np.nonzero(body[:, :, 3].max(0) > 0)[0]
            if len(rows) == 0:
                log(f"  {ident.character}/{ident.variant}: 불투명한 자리가 없어 건너뜁니다")
                continue
            sx, sy = ident.stance()
            # 몸이 선 자리(오라클 캔버스 px)
            top = (sy + rows.min()) / v
            x0, x1 = (sx + cols.min()) / v, (sx + cols.max() + 1) / v
            depth = depth_try[-1]
            for cand in depth_try:
                if _slot(oracle, ident.character, ident.variant, "center", "bust", cand)["rect"][1] <= top - 2:
                    depth = cand
                    break
            centers = {}
            for p in PLACES:
                r = _slot(oracle, ident.character, ident.variant, p, "bust", depth)["rect"]
                centers[p] = (r[0] + r[2] / 2.0, r)
            place = min(PLACES, key=lambda p: abs(centers[p][0] - (x0 + x1) / 2.0))
            cx, rect = centers[place]
            H = int(round(rect[3] * v))
            half = max(abs(x0 - cx), abs(x1 - cx)) * v + 4
            W = int(min(2 * np.ceil(half), np.floor(H * rect[2] / rect[3])))
            plans.append((ident, body, (sx, sy), place, depth, (W, H)))
            log(f"  {ident.character}/{ident.variant}: 기준 자리 place_{place} · bust · {depth} · 그림 {W}×{H}")

    # 2단: 정한 크기로 겹본을 쓰고, 실제로 그려지는 자리(quad)에 맞춰 몸 · 표정을 붙인다.
    dims = {}
    emos: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for o in ed.live:
        key = (o["character"], o.get("variant", "a"))
        if o.get("emotion", "01") not in emos[key]:
            emos[key].append(o.get("emotion", "01"))
    for ident, _b, _s, _p, _d, size in plans:
        for e in emos[(ident.character, ident.variant)]:
            dims[(ident.character, ident.variant, e)] = size
    write_overlay(runtime_tuning, tuning, dims, bg_dims)
    with Oracle(tuning, command=oracle_command) as oracle:
        v = vh / float(oracle.height)
        for ident, body, (sx, sy), place, depth, (W, H) in plans:
            s = _slot(oracle, ident.character, ident.variant, place, "bust", depth)
            qx, qy, qw, qh = s["bounds"]
            k = W / qw / v                                  # 영상 화소 → 그림 화소
            m = np.float32([[k, 0, (sx / v - qx) * W / qw], [0, k, (sy / v - qy) * H / qh]])
            rows = np.nonzero(body[:, :, 3].max(1) > 0)[0]
            top, low = int(rows.min()), int(rows.max()) + 1
            head_end = int(np.ceil(top + HEAD_PART * (low - top)))
            for e in emos[(ident.character, ident.variant)]:
                canvas = body.copy()
                rep = ed.representative(ident.character, ident.variant, e)
                if rep is not None and rep["id"] in ident.at:
                    face = body.copy()
                    x, y = ident.local(rep["id"])
                    _paste(face, ident.pieces[rep["id"]], x, y)
                    canvas[:head_end] = face[:head_end]
                rgba = np.clip(canvas + 0.5, 0, 255).astype(np.uint8)
                img = _premultiplied_warp(rgba, m, (W, H))
                rel = os.path.join("portraits", ident.character, ident.variant, f"{e}.png")
                write_image(os.path.join(out_dir, rel), img)
                ys, xs = np.nonzero(img[:, :, 3] > 8)
                box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) \
                    if len(xs) else (0, 0, 0, 0)
                cat.portraits.append(Portrait(ident.character, ident.variant, e, rel, (W, H), box))
    path = os.path.join(out_dir, "catalog.json")
    cat.save(path)
    log(f"초상 {len(cat.portraits)} · 배경 {len(cat.backgrounds)} → {path}")
    return path, tuning
