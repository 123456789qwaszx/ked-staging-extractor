"""게임텍스트추출기(GameTextExtractor) 결과 엑셀 → 줄마다 화자 · 대사 글자.

엑셀 '텍스트' 시트의 열(머리글 이름으로 찾는다): No · 시간(mm:ss.ss, 그 대사를 처음 읽은 순간) · 형태 · 화자 ·
원문 · 번역 · 메모 · 확인 · 신뢰도. 번역가가 원문을 고친 엑셀을 그대로 넣어도 된다 — 원문 칸을 읽는다.

배정 규칙: 읽은 순간이 [줄 머리, 다음 줄 머리) 안에 드는 줄에 붙인다. 추출기는 대사가 멈춘 뒤에 읽으므로
읽은 순간은 언제나 그 줄 안에 있다. 한 줄에 둘 이상 붙으면 이어 붙이고 확인 표시, 아무것도 안 붙은 줄도 확인 표시.
대사창의 형태('대사' · '나레이션' · '주인공대사' · '주인공속말')만 줄 글자로, 그 밖(장소/시간 카드 · 장제목 · 메신저 ·
선택지 · UI)은 줄 글자가 아니라 메모로 붙인다. (추출기 3단계부터 이름표 없는 주인공 대사가 '나레이션' 대신
'주인공대사' · '주인공속말'로 나온다.)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

SHEET = "텍스트"
DIALOGUE_KINDS = ("대사", "나레이션", "주인공대사", "주인공속말")
_TIME = re.compile(r"^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*$")


@dataclass
class GteRow:
    no: int
    t: float
    kind: str
    speaker: str
    text: str
    review: str = ""
    conf: Optional[float] = None


@dataclass
class LineText:
    speaker: str = ""
    text: str = ""
    rows: List[int] = field(default_factory=list)       # 붙은 엑셀 줄 번호(No)
    notes: List[str] = field(default_factory=list)       # 장소/시간 카드 등
    review: List[str] = field(default_factory=list)      # 확인할 까닭


def parse_time(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    m = _TIME.match(str(value or ""))
    if not m:
        raise ValueError(f"시간 칸을 읽지 못했습니다: {value!r}")
    return float(m.group(1) or 0) * 60.0 + float(m.group(2))


def read_xlsx(path: str) -> List[GteRow]:
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[SHEET] if SHEET in wb.sheetnames else wb.worksheets[0]
    rows = ws.iter_rows(values_only=True)
    header = [str(v or "").strip() for v in next(rows)]

    def col(name: str) -> Optional[int]:
        return header.index(name) if name in header else None

    c_no, c_t, c_kind, c_spk, c_txt = col("No"), col("시간"), col("형태"), col("화자"), col("원문")
    c_rev, c_conf = col("확인"), col("신뢰도")
    if c_t is None or c_txt is None:
        raise ValueError(f"게임텍스트추출기 엑셀이 아닙니다(시간 · 원문 열 없음): {path}")
    out: List[GteRow] = []
    for i, r in enumerate(rows, 1):
        if r is None or all(v in (None, "") for v in r):
            continue
        try:
            t = parse_time(r[c_t])
        except ValueError:
            continue
        conf = r[c_conf] if c_conf is not None else None
        out.append(GteRow(
            no=int(r[c_no]) if c_no is not None and isinstance(r[c_no], (int, float)) else i,
            t=t,
            kind=str(r[c_kind] or "") if c_kind is not None else "대사",
            speaker=str(r[c_spk] or "").strip() if c_spk is not None else "",
            text=str(r[c_txt] or "").strip(),
            review=str(r[c_rev] or "") if c_rev is not None else "",
            conf=float(conf) if isinstance(conf, (int, float)) else None))
    wb.close()
    out.sort(key=lambda row: row.t)
    return out


def attach(heads: Sequence[float], ends: Sequence[float], rows: Sequence[GteRow]) -> List[LineText]:
    """줄 구간들([head, end))에 엑셀 줄을 붙인다."""
    texts = [LineText() for _ in heads]
    if not heads:
        return texts
    for row in rows:
        k = None
        for i, h in enumerate(heads):
            nxt = heads[i + 1] if i + 1 < len(heads) else float("inf")
            if h - 0.05 <= row.t < nxt:
                k = i
                break
        if k is None:
            k = 0  # 첫 줄 머리보다 앞 — 영상 앞머리에서 읽힌 것
        lt = texts[k]
        if row.kind and row.kind not in DIALOGUE_KINDS:
            lt.notes.append(f"{row.kind}: {row.text}")
            continue
        if row.t > ends[k] + 0.3:
            lt.review.append(f"엑셀 {row.no}번이 대사창이 닫힌 뒤에 읽혔습니다")
        if lt.rows:
            lt.review.append(f"엑셀 {lt.rows[-1]}번과 {row.no}번이 한 줄에 붙었습니다")
            lt.text = (lt.text + " " + row.text).strip()
            if row.speaker and row.speaker != lt.speaker:
                lt.review.append(f"화자가 둘입니다: {lt.speaker} / {row.speaker}")
        else:
            lt.text = row.text
            lt.speaker = row.speaker
        lt.rows.append(row.no)
        if row.review:
            lt.review.append(f"엑셀 {row.no}번 확인: {row.review}")
    for lt in texts:
        if not lt.rows:
            lt.review.append("붙은 대사가 없습니다(게임텍스트추출기가 못 읽었거나 줄이 아닌 구간)")
    return texts


def write_like_gte(path: str, rows: Sequence[GteRow]) -> str:
    """게임텍스트추출기와 같은 머리글의 '텍스트' 시트만 쓴다 — 시험용(정답에서 흉내 낸 엑셀)."""
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET
    ws.append(["No", "시간", "형태", "화자", "원문", "번역", "메모", "확인", "신뢰도", "스크린샷", "원본"])
    for r in rows:
        m, s = divmod(max(0.0, r.t), 60)
        ws.append([r.no, f"{int(m):02d}:{s:05.2f}", r.kind, r.speaker, r.text, "", "", r.review,
                   int(round(r.conf)) if r.conf is not None else 100])
    tmp = path + ".tmp.xlsx"
    wb.save(tmp)
    os.replace(tmp, path)
    return path


def rows_from_truth(truth: dict, delay: float = 0.25) -> List[GteRow]:
    """합성 녹화 정답 → 게임텍스트추출기가 냈을 법한 줄(다 찍히고 delay 뒤에 읽음)."""
    rows = []
    for i, line in enumerate(truth["lines"], 1):
        rows.append(GteRow(i, float(line["typed"]) + delay, "대사" if line["speaker"] else "나레이션",
                           line["speaker"], line["body"]))
    return rows
