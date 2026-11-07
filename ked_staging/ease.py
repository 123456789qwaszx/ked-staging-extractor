"""DOTween 이징 — 런타임이 내보낸 표본(ExportedTuning/ease-golden.json)이 유일한 정의다.

규칙 4: 이징은 런타임의 것만. 수식을 파이썬으로 다시 쓰지 않고, DOTween이 실제로 낸 257개 표본을
선형 보간한다(간격 1/256 — 영상 한 프레임보다 훨씬 촘촘하다).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

DEFAULT_EASE = "OutCubic"  # 커맨드 대부분의 생략값 (YarnEaseParser)


@dataclass(frozen=True)
class EaseTable:
    names: List[str]
    samples: Dict[str, np.ndarray]  # 이름 → 길이 N 배열, t[i] = i/(N-1)
    source: str
    dotween_version: str

    @classmethod
    def load(cls, path: str) -> "EaseTable":
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        if data.get("schema") != "ease-golden/1":
            raise ValueError(f"ease-golden/1 형식이 아닙니다: {data.get('schema')!r}")
        curves = data["eases"]
        names = [c["name"] for c in curves]
        samples = {c["name"]: np.asarray(c["samples"], dtype=np.float64) for c in curves}
        n = int(data["sampleCount"])
        bad = [k for k, v in samples.items() if len(v) != n]
        if bad:
            raise ValueError(f"표본 수가 {n}이 아닌 곡선: {bad}")
        return cls(names, samples, data.get("source", ""), data.get("dotweenVersion", ""))

    def __contains__(self, name: str) -> bool:
        return name in self.samples

    def evaluate(self, name: str, t) -> np.ndarray:
        """진행도 t∈[0,1] (배열 가능) → 이징 값. 범위 밖은 끝값으로 자른다(트윈은 시작 전·끝 뒤에 머문다)."""
        s = self.samples[name]
        x = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * (len(s) - 1)
        i = np.minimum(np.floor(x).astype(int), len(s) - 2)
        f = x - i
        return s[i] * (1.0 - f) + s[i + 1] * f

    def canonical(self, name: str) -> str:
        """대소문자를 무시하고 표의 이름으로. 모르면 KeyError."""
        low = {n.lower(): n for n in self.names}
        return low[name.strip().lower()]

    def overshooting(self) -> List[str]:
        """[0,1] 밖으로 나가는 곡선(Back · Elastic) — 끝값을 지나쳤다가 돌아오는 전이를 가려낼 때 쓴다."""
        return [n for n in self.names if self.samples[n].min() < -1e-6 or self.samples[n].max() > 1 + 1e-6]
