"""시간 · 거리 토큰 — 런타임 YarnDurationParser · UnitToken과 같은 규칙.

- 시간: `12fr`(24fps) · `0.5s` · `0.5`(맨 숫자 = 초)
- 거리: `3u` · `3`(맨 숫자 = u), 1u = 기준 무대 폭 / 48 (1920 → 40px)
"""
from __future__ import annotations

import math
import re

FPS = 24
UNITS_PER_STAGE_WIDTH = 48
DEFAULT_STAGE_WIDTH = 1920.0

_DURATION = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(fr|s)?\s*$", re.IGNORECASE)
_UNIT = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(u)?\s*$", re.IGNORECASE)


def parse_duration(token: str) -> float:
    """토큰 → 초. 못 읽으면 ValueError (런타임은 경고 후 생략값으로 가지만, 추출기는 자기가 낸 토큰만 읽는다)."""
    m = _DURATION.match(token or "")
    if not m:
        raise ValueError(f"시간 토큰이 아닙니다: {token!r}")
    value = float(m.group(1))
    return value / FPS if (m.group(2) or "").lower() == "fr" else value


def frames(seconds: float) -> int:
    """초 → 가장 가까운 24fps 프레임 수(0 이상)."""
    return max(0, int(round(seconds * FPS)))


def duration_token(seconds: float) -> str:
    """초 → 정본 시간 토큰. 언제나 프레임 단위(`18fr`)로 낸다 — 대본이 실제로 쓰는 표기다."""
    return f"{frames(seconds)}fr"


def pixels_per_unit(stage_width: float = DEFAULT_STAGE_WIDTH) -> float:
    return stage_width / UNITS_PER_STAGE_WIDTH


def parse_units(token: str, stage_width: float = DEFAULT_STAGE_WIDTH) -> float:
    """거리 토큰 → px (부호 유지)."""
    m = _UNIT.match(token or "")
    if not m:
        raise ValueError(f"거리 토큰이 아닙니다: {token!r}")
    return float(m.group(1)) * pixels_per_unit(stage_width)


def unit_token(pixels: float, stage_width: float = DEFAULT_STAGE_WIDTH, step: float = 0.25) -> str:
    """px → 정본 거리 토큰(`2.5u`). step 단위로 반올림한다(대본은 보통 0.25u보다 잘게 쓰지 않는다)."""
    units = round(pixels / pixels_per_unit(stage_width) / step) * step
    if math.isclose(units, round(units)):
        return f"{int(round(units))}u"
    return f"{units:g}u"
