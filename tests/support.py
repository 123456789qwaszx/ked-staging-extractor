"""테스트가 기대는 바깥 자원 — 옆의 런타임 저장소와 오라클 빌드. 없으면 그 테스트만 건너뛴다."""
import os
import unittest

from ked_staging.oracle import find_oracle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.environ.get("KED_RUNTIME") or os.path.join(os.path.dirname(REPO), "ked-presentation-runtime")
TUNING = os.path.join(RUNTIME, "ExportedTuning")
MOTIONS = os.path.join(RUNTIME, "Assets", "Dialogue", "motions.json")
YARN_DIR = os.path.join(RUNTIME, "Assets", "Dialogue", "ChapterProgression")
GOLDEN = os.path.join(TUNING, "ease-golden.json")

needs_runtime = unittest.skipUnless(os.path.isdir(TUNING), f"런타임 저장소가 없습니다: {RUNTIME}")
needs_oracle = unittest.skipUnless(os.path.isdir(TUNING) and find_oracle(), "ked-stage-oracle 빌드가 없습니다")


def yarn(name: str) -> str:
    return os.path.join(YARN_DIR, name)
