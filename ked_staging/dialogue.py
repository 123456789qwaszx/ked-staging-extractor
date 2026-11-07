"""대사창 — 자리(surface-layout 프리셋), 합성 녹화용 그리기, 게임텍스트추출기 프로필.

자리는 런타임 DialogueSurfaceLayoutPresetDB(ExportedTuning/presets/surface-layout.json)의 앵커를 그대로 쓴다.
앵커는 화면 비율 · y 위 방향이고, 여기서는 캔버스 px(왼쪽 위 원점 · y 아래)로 바꿔 둔다.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]  # x0, y0, x1, y1 (px, 왼쪽 위 원점)

DEFAULT_PRESET = "bottom"
_SPEAKER = re.compile(r"^\s*([^:：\n]{1,24}?)\s*[:：]\s+(.*)$", re.S)


def split_speaker(text: str) -> Tuple[str, str]:
    """Yarn 대사 "화자: 본문" → (화자, 본문). 화자가 없으면 ("", 본문)."""
    m = _SPEAKER.match(text or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", (text or "").strip()


@dataclass
class BoxLayout:
    width: int
    height: int
    line: Rect
    name: Optional[Rect]
    line_font: float
    name_font: float
    preset: str = DEFAULT_PRESET

    @classmethod
    def from_tuning(cls, tuning_dir: str, width: int = 1920, height: int = 1080,
                    preset: str = DEFAULT_PRESET) -> "BoxLayout":
        path = os.path.join(tuning_dir, "presets", "surface-layout.json")
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        body = data.get("MonoBehaviour", data)
        for entry in body.get("entries", []):
            if entry.get("key") != preset:
                continue
            ref = 1080.0  # 글자 크기는 기준 해상도(1080p) 값이다
            s = height / ref

            def rect(lo, hi) -> Rect:
                return (lo["x"] * width, (1.0 - hi["y"]) * height, hi["x"] * width, (1.0 - lo["y"]) * height)

            name = rect(entry["nameAnchorMin"], entry["nameAnchorMax"]) if entry.get("useName") else None
            return cls(width, height, rect(entry["lineAnchorMin"], entry["lineAnchorMax"]), name,
                       float(entry.get("lineFontSize", 36.0)) * s, float(entry.get("nameFontSize", 28.0)) * s, preset)
        raise KeyError(f"surface-layout에 '{preset}' 프리셋이 없습니다: {path}")

    def scaled(self, width: int, height: int) -> "BoxLayout":
        sx, sy = width / self.width, height / self.height

        def sc(r: Optional[Rect]) -> Optional[Rect]:
            return None if r is None else (r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy)

        return BoxLayout(width, height, sc(self.line), sc(self.name), self.line_font * sy, self.name_font * sy,
                         self.preset)

    def panel(self) -> Rect:
        """대사창 판(이름표 + 본문)을 덮는 사각형 — 그리기와 대조 가림에 같이 쓴다."""
        x0, y0, x1, y1 = self.line
        if self.name:
            x0, y0 = min(x0, self.name[0]), min(y0, self.name[1])
            x1 = max(x1, self.name[2])
        mx, my = 0.02 * self.width, 0.025 * self.height
        return (max(0.0, x0 - mx), max(0.0, y0 - my), min(self.width, x1 + mx), min(self.height, y1 + my))

    def stage_bottom(self) -> int:
        """무대를 대조할 때 쓰는 아래 끝(px) — 그 아래는 대사창이다."""
        return int(np.floor(self.panel()[1]))


# ── 합성 녹화용 그리기 ──────────────────────────────────────────

_FONT_CANDIDATES = [
    os.environ.get("KED_FONT", ""),
    r"C:\Windows\Fonts\malgun.ttf",
    r"C:\Windows\Fonts\malgunbd.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


def find_font() -> str:
    for path in _FONT_CANDIDATES:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError("한글 글꼴을 찾지 못했습니다 — KED_FONT에 .ttf/.ttc 경로를 지정하세요.")


class BoxPainter:
    """대사창 판 · 이름 · 본문(타자기)을 프레임 위에 그린다. 글자 모양은 런타임과 다르다(자리 · 크기만 같다)."""

    def __init__(self, layout: BoxLayout, font_path: Optional[str] = None, panel_alpha: float = 0.78):
        from PIL import ImageFont  # 합성 녹화에만 필요하다
        self.layout = layout
        path = font_path or find_font()
        self.font = ImageFont.truetype(path, max(8, int(round(layout.line_font))))
        self.name_font = ImageFont.truetype(path, max(8, int(round(layout.name_font))))
        self.panel_alpha = panel_alpha
        self._wrap_cache: dict = {}

    def wrap(self, text: str) -> List[str]:
        if text in self._wrap_cache:
            return self._wrap_cache[text]
        width = self.layout.line[2] - self.layout.line[0]
        lines: List[str] = []
        for para in text.split("\n"):
            cur = ""
            for ch in para:
                if self.font.getlength(cur + ch) > width and cur:
                    lines.append(cur)
                    cur = ch
                else:
                    cur += ch
            lines.append(cur)
        self._wrap_cache[text] = lines
        return lines

    def paint(self, frame: np.ndarray, speaker: str, body: str, shown: int) -> np.ndarray:
        """frame(BGR uint8)에 대사창을 그린다. shown = 지금까지 찍힌 본문 글자 수."""
        from PIL import Image, ImageDraw
        x0, y0, x1, y1 = (int(round(v)) for v in self.layout.panel())
        out = frame.copy()
        region = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (region * (1 - self.panel_alpha) + np.array([28, 22, 20]) * self.panel_alpha
                             ).astype(np.uint8)
        img = Image.fromarray(out[y0:y1, x0:x1][:, :, ::-1])
        draw = ImageDraw.Draw(img)
        if speaker and self.layout.name:
            nx0, ny0, nx1, ny1 = self.layout.name
            draw.text((nx0 - x0, (ny0 + ny1) / 2 - y0), speaker, font=self.name_font, fill=(255, 214, 140),
                      anchor="lm")
        lx0, ly0, _lx1, _ly1 = self.layout.line
        remaining = max(0, shown)
        y = ly0 - y0
        step = self.layout.line_font * 1.35
        for row in self.wrap(body):
            if remaining <= 0:
                break
            part = row[:remaining]
            remaining -= len(row)
            draw.text((lx0 - x0, y), part, font=self.font, fill=(245, 245, 245))
            y += step
        out[y0:y1, x0:x1] = np.asarray(img)[:, :, ::-1]
        return out


# ── 게임텍스트추출기 프로필 ─────────────────────────────────────

def gte_profile(layout: BoxLayout, name: str = "ked-presentation-runtime (대사창 bottom)") -> dict:
    """게임텍스트추출기(GameTextExtractor) 게임 프로필 — 같은 대사창 자리를 비율 좌표로 적는다."""
    w, h = float(layout.width), float(layout.height)
    px0, py0, px1, py1 = layout.panel()
    regions = [{"type": "area", "rect": [round(px0 / w, 5), round(py0 / h, 5), round(px1 / w, 5), round(py1 / h, 5)],
                "label": "대사", "label_no_speaker": "나레이션"}]
    if layout.name:
        nx0, ny0, nx1, ny1 = layout.name
        my = 0.15 * (ny1 - ny0)
        regions.append({"type": "speaker", "rect": [round(nx0 / w, 5), round((ny0 - my) / h, 5),
                                                     round(nx1 / w, 5), round((ny1 + my) / h, 5)]})
    return {
        "name": name,
        "version": 1,
        "language": "kor",
        "sample_fps": 10.0,
        "stable_sec": 0.2,
        "change_threshold": 0.05,
        "force_ocr_sec": 1.2,
        "min_confidence": 40.0,
        "review_confidence": 80.0,
        "ellipsis": "……",
        "line_join": "auto",
        "regions": regions,
        "notes": f"ked-staging-extractor가 surface-layout '{layout.preset}' 프리셋에서 만든 프로필입니다.",
    }
