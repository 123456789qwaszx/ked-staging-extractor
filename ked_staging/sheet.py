"""엑셀 연출 열 ↔ .yarn (P5) — .yarn이 값의 주인이고, 엑셀은 무손실로 왕복하는 고치기 보기다.

  export  .yarn → 연출 엑셀. 줄(대사 · 선택지)마다 한 행: 화자 · 대사 · 연출(그 줄 앞의 커맨드) · 모션 · 확인.
          노드 끝(마지막 줄 뒤 커맨드)도 한 행이다. '단위' 시트의 연출 단위(@등장 …)로 이어진 커맨드를 접는다.
  apply   원래 .yarn + 고친 엑셀 → 새 .yarn. 고친 행만 다시 쓰고 나머지 글자는 바이트 그대로 둔다.

행 하나(블록)는 그 줄과 그 앞의 커맨드 · 주석 · 빈 줄이다.
  연출   표현 커맨드(<<if>> 같은 흐름 커맨드가 아닌 것). 한 줄에 하나, << >>는 써도 되고 안 써도 된다.
         '@단위 인자…'는 '단위' 시트의 단계로 편다.
  모션   연출 끝의 motion · pause만으로 된 꼬리(모션이 하나는 있어야 한다). 들여올 때 연출 뒤에 붙는다.
  확인   '// 확인:' 주석(추출기가 단 확인할 까닭). 고치면 그 주석을 바꾼다 — 비우면 지운다.

고쳤는지는 칸 글자가 아니라 편 커맨드로 견준다 — 단위 시트를 고쳐도 손대지 않은 행은 그대로다.
들여오지 않는 것: 내보낸 뒤 .yarn이 바뀜(지문, --force로 무시) · 행을 넣거나 지움 · 선택지 줄 고치기 ·
흐름 커맨드와 섞인 행의 연출 고치기 · 모르는 단위 · 인자 수 틀림 · (오라클 표를 주면) 모르는 커맨드.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

SCHEMA = "ked-staging-sheet/1"
COLUMNS = ["노드", "행", "종류", "화자", "대사", "연출", "모션", "확인", "열쇠"]
CHECK_PREFIX = "// 확인:"
FLOW = {"if", "elseif", "else", "endif", "set", "declare", "jump", "detour", "return", "stop", "once", "endonce",
        "local", "enum", "case", "endenum"}
MOTION_TAIL = {"motion", "pause"}

# 기본 연출 단위 — (이름, 인자, 단계, 설명). 내보낼 때 '단위' 시트에 적고, 들여올 때는 시트의 것을 쓴다.
DEFAULT_UNITS: List[Tuple[str, str, List[str], str]] = [
    ("등장", "slot place focus depth dur",
     ["place_$place $slot $focus 0fr", "size $slot $depth $focus 0fr", "fade_in $slot $dur"],
     "자리 · 크기를 잡고 나타난다"),
    ("밀며등장", "slot place focus depth dir dist dur",
     ["place_$place $slot $focus 0fr", "size $slot $depth $focus 0fr", "slide_in $slot $dir $dist $dur",
      "fade_in $slot $dur"],
     "자리 · 크기를 잡고 미끄러져 들어오며 나타난다(샘플 대본 관례)"),
    ("배경", "key out in",
     ["bg_fade_out bg_main $out", "bg_sprite bg_main $key", "bg_fade_in bg_main $in"],
     "배경이 어두워졌다가 다른 그림으로 밝아진다"),
]

_COMMAND = re.compile(r"^(?P<indent>[ \t]*)<<(?P<body>.*)>>[ \t]*$")
_SPEAKER = re.compile(r"^(?P<speaker>[^\s:#<>\\][^:#<>\\]{0,31}?):[ \t]?(?P<rest>.*)$")
_TAGS = re.compile(r"^(?P<body>.*?)(?P<tags>(?:[ \t]+#[^\s#]+)+)?(?P<trail>[ \t]*)$")
_PARAM = re.compile(r"\$(\w+)")


class SheetError(ValueError):
    pass


def normalize(command: str) -> str:
    return " ".join(command.split())


# ── .yarn 읽기 ────────────────────────────────────────────────

@dataclass
class Block:
    node: str
    number: int                      # 노드 안 행 번호(1부터), 노드 끝은 0
    start: int                       # 파일 줄 번호 [start, end)
    end: int
    kind: str                        # 대사 | 선택지 | 노드 끝
    content: Optional[int] = None    # 그 줄(파일 줄 번호)
    speaker: str = ""
    text: str = ""
    tags: str = ""
    commands: List[Tuple[int, str]] = field(default_factory=list)   # (파일 줄, 커맨드 몸)
    checks: List[Tuple[int, str]] = field(default_factory=list)
    flow: List[int] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.node}#{self.number}"

    def split_commands(self) -> Tuple[List[str], List[str]]:
        """(연출, 모션) — 모션은 motion · pause만으로 된 꼬리(motion이 하나는 있다)."""
        cmds = [normalize(c) for _, c in self.commands]
        k = len(cmds)
        while k > 0 and cmds[k - 1].split(" ", 1)[0] in MOTION_TAIL:
            k -= 1
        tail = cmds[k:]
        if not any(c.split(" ", 1)[0] == "motion" for c in tail):
            return cmds, []
        return cmds[:k], tail


@dataclass
class YarnDoc:
    lines: List[Tuple[str, str]]     # (글자, 줄바꿈)
    blocks: List[Block]
    newline: str

    def render(self) -> str:
        return "".join(t + e for t, e in self.lines)


def _split_lines(text: str) -> List[Tuple[str, str]]:
    out = []
    for raw in text.splitlines(keepends=True):
        if raw.endswith("\r\n"):
            out.append((raw[:-2], "\r\n"))
        elif raw.endswith("\n") or raw.endswith("\r"):
            out.append((raw[:-1], raw[-1]))
        else:
            out.append((raw, ""))
    return out


def _parse_content(line: str) -> Tuple[str, str, str]:
    """대사 줄 → (화자, 대사, 태그 — 앞 공백 포함). 들여쓰기는 뺀다."""
    s = line.strip(" \t")
    m = _TAGS.match(s)
    body, tags = m.group("body"), (m.group("tags") or "")
    sm = _SPEAKER.match(body)
    if sm:
        return sm.group("speaker").strip(), sm.group("rest"), tags
    return "", body, tags


def parse_yarn(text: str) -> YarnDoc:
    lines = _split_lines(text)
    newline = "\r\n" if sum(e == "\r\n" for _, e in lines) > sum(e == "\n" for _, e in lines) else "\n"
    blocks: List[Block] = []
    i, n = 0, len(lines)
    while i < n:
        t = lines[i][0].lstrip("﻿").strip()
        if not t.startswith("title:"):
            i += 1
            continue
        title = t[len("title:"):].strip()
        while i < n and lines[i][0].strip() != "---":
            i += 1
        i += 1
        number = 0
        start = i
        cur = Block(title, 0, start, start, "")
        while i < n and lines[i][0].strip() != "===":
            raw = lines[i][0]
            s = raw.strip()
            m = _COMMAND.match(raw)
            if m and s.startswith("<<"):
                body = m.group("body").strip()
                name = body.split(" ", 1)[0] if body else ""
                if name in FLOW:
                    cur.flow.append(i)
                else:
                    cur.commands.append((i, body))
            elif s.startswith(CHECK_PREFIX):
                cur.checks.append((i, s[len(CHECK_PREFIX):].strip()))
            elif not s or s.startswith("//"):
                pass
            else:
                number += 1
                cur.number, cur.end, cur.content = number, i + 1, i
                if s.startswith("->") or s.startswith("=>"):
                    cur.kind, cur.text = "선택지", s
                else:
                    cur.kind = "대사"
                    cur.speaker, cur.text, cur.tags = _parse_content(raw)
                blocks.append(cur)
                cur = Block(title, 0, i + 1, i + 1, "")
            i += 1
        cur.end, cur.kind = i, "노드 끝"
        blocks.append(cur)
        i += 1
    return YarnDoc(lines, blocks, newline)


# ── 연출 단위 ─────────────────────────────────────────────────

@dataclass
class Unit:
    name: str
    params: List[str]
    steps: List[str]
    note: str = ""

    def expand(self, args: Sequence[str]) -> List[str]:
        if len(args) != len(self.params):
            raise SheetError(f"@{self.name}: 인자 {len(self.params)}개({' '.join(self.params)})인데 {len(args)}개를 줬습니다")
        values = dict(zip(self.params, args))
        out = []
        for step in self.steps:
            def sub(m):
                if m.group(1) not in values:
                    raise SheetError(f"@{self.name}: 단계에 모르는 인자 ${m.group(1)}")
                return values[m.group(1)]
            out.append(normalize(_PARAM.sub(sub, step)))
        return out

    def pattern(self, commands: Sequence[str], at: int) -> Optional[List[str]]:
        """commands[at:]가 이 단위로 시작하면 인자들. 아니면 None."""
        if at + len(self.steps) > len(commands):
            return None
        values: Dict[str, str] = {}
        for step, cmd in zip(self.steps, commands[at:at + len(self.steps)]):
            names: List[str] = []

            def rx(m):
                names.append(m.group(1))
                return r"(\S+)"
            regex = "^" + _PARAM.sub(rx, re.escape(normalize(step)).replace(r"\$", "$")) + "$"
            hit = re.match(regex, cmd)
            if not hit:
                return None
            for name, v in zip(names, hit.groups()):
                if values.setdefault(name, v) != v:
                    return None
        if set(values) != set(self.params):
            return None
        return [values[p] for p in self.params]


def default_units() -> Dict[str, Unit]:
    return {name: Unit(name, params.split(), steps, note) for name, params, steps, note in DEFAULT_UNITS}


def fold(commands: Sequence[str], units: Dict[str, Unit]) -> List[str]:
    """이어진 커맨드를 단위로 접는다(긴 단위 먼저). 편 결과가 원래와 글자까지 같을 때만 접는다."""
    order = sorted(units.values(), key=lambda u: -len(u.steps))
    out, i = [], 0
    while i < len(commands):
        for u in order:
            args = u.pattern(commands, i)
            if args is not None and u.expand(args) == list(commands[i:i + len(u.steps)]):
                out.append("@" + " ".join([u.name] + args))
                i += len(u.steps)
                break
        else:
            out.append(commands[i])
            i += 1
    return out


def unfold(cell: Optional[str], units: Dict[str, Unit]) -> List[str]:
    out = []
    for raw in (cell or "").replace("\r\n", "\n").split("\n"):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("@"):
            parts = s[1:].split()
            if not parts or parts[0] not in units:
                raise SheetError(f"모르는 연출 단위 '{s.split()[0]}' — '단위' 시트에 있어야 합니다")
            out += units[parts[0]].expand(parts[1:])
            continue
        if s.startswith("<<") and s.endswith(">>"):
            s = s[2:-2]
        s = normalize(s)
        if s:
            out.append(s)
    return out


def _cell(values: Sequence[str]) -> str:
    return "\n".join(values)


# ── 엑셀 ─────────────────────────────────────────────────────

def _fingerprint(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def block_row(b: Block, units: Dict[str, Unit]) -> dict:
    staging, motion = b.split_commands()
    return {"노드": b.node, "행": str(b.number) if b.number else "(노드 끝)", "종류": b.kind,
            "화자": b.speaker or None, "대사": b.text or None, "연출": _cell(fold(staging, units)) or None,
            "모션": _cell(motion) or None, "확인": _cell([c for _, c in b.checks]) or None, "열쇠": b.key}


def export_sheet(yarn: str, xlsx: str) -> int:
    """.yarn → 연출 엑셀. → 행 수."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    with open(yarn, encoding="utf-8", newline="") as f:
        doc = parse_yarn(f.read())
    units = default_units()
    wb = Workbook()
    ws = wb.active
    ws.title = "연출"
    ws.append(COLUMNS)
    wrap = Alignment(wrap_text=True, vertical="top")
    for b in doc.blocks:
        row = block_row(b, units)
        ws.append([row[c] for c in COLUMNS])
        r = ws.max_row
        for col in range(1, len(COLUMNS) + 1):
            ws.cell(r, col).alignment = wrap
        if row["확인"]:
            for col in range(1, len(COLUMNS) + 1):
                ws.cell(r, col).fill = PatternFill("solid", fgColor="FFF4C2")
    for col, width in zip("ABCDEFGHI", (16, 9, 8, 10, 48, 44, 28, 36, 18)):
        ws.column_dimensions[col].width = width
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.freeze_panes = "D2"

    wu = wb.create_sheet("단위")
    wu.append(["이름", "인자", "단계", "설명"])
    for name, params, steps, note in DEFAULT_UNITS:
        wu.append([name, params, "\n".join(steps), note])
        for col in range(1, 5):
            wu.cell(wu.max_row, col).alignment = wrap
    for col, width in zip("ABCD", (12, 36, 48, 40)):
        wu.column_dimensions[col].width = width

    wi = wb.create_sheet("정보")
    wi.append(["열쇠", "값", "설명"])
    wi.append(["schema", SCHEMA, "이 엑셀의 형식"])
    wi.append(["yarn", os.path.abspath(yarn), "값의 주인인 .yarn — 들여오기는 이 파일에 한다"])
    wi.append(["sha256", _fingerprint(yarn), "내보낼 때 .yarn의 지문 — 다르면 들여오지 않는다(다시 내보내세요)"])
    wi.append(["안내", "행을 넣거나 지우지 마세요. 연출 칸은 한 줄에 커맨드 하나, '@단위 인자…'는 '단위' 시트로 편다.", ""])
    wi.column_dimensions["B"].width = 80
    os.makedirs(os.path.dirname(os.path.abspath(xlsx)) or ".", exist_ok=True)
    wb.save(xlsx)
    return len(doc.blocks)


def read_sheet(xlsx: str) -> Tuple[dict, List[dict], Dict[str, Unit]]:
    """→ (정보, 행들, 단위)."""
    from openpyxl import load_workbook
    wb = load_workbook(xlsx)
    for name in ("연출", "단위", "정보"):
        if name not in wb.sheetnames:
            raise SheetError(f"'{name}' 시트가 없습니다 — sheet export로 만든 엑셀인가요?")
    info = {str(r[0]): r[1] for r in wb["정보"].iter_rows(min_row=2, values_only=True) if r and r[0]}
    if info.get("schema") != SCHEMA:
        raise SheetError(f"{SCHEMA} 형식이 아닙니다: {info.get('schema')!r}")
    ws = wb["연출"]
    head = [c.value for c in ws[1]]
    missing = [c for c in COLUMNS if c not in head]
    if missing:
        raise SheetError(f"'연출' 시트에 열이 없습니다: {', '.join(missing)}")
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r is None or all(v is None for v in r):
            continue
        rows.append({c: (None if r[head.index(c)] is None else str(r[head.index(c)])) for c in COLUMNS})
    units: Dict[str, Unit] = {}
    for r in wb["단위"].iter_rows(min_row=2, values_only=True):
        if not r or not r[0]:
            continue
        name = str(r[0]).strip().lstrip("@")
        steps = [s.strip() for s in str(r[2] or "").replace("\r\n", "\n").split("\n") if s.strip()]
        if not steps:
            raise SheetError(f"단위 '{name}'에 단계가 없습니다")
        units[name] = Unit(name, str(r[1] or "").split(), steps, str(r[3] or "") if len(r) > 3 else "")
    return info, rows, units


def _same(a: Optional[str], b: Optional[str]) -> bool:
    return (a or "").replace("\r\n", "\n").strip() == (b or "").replace("\r\n", "\n").strip()


def apply_sheet(xlsx: str, yarn: Optional[str] = None, force: bool = False,
                known_commands: Optional[Set[str]] = None) -> Tuple[str, List[str]]:
    """원래 .yarn + 고친 엑셀 → (새 .yarn 글자, 바뀐 행 열쇠들). 하나라도 틀리면 아무것도 들여오지 않는다(SheetError)."""
    info, rows, units = read_sheet(xlsx)
    path = yarn or info.get("yarn")
    if not path or not os.path.isfile(path):
        raise SheetError(f".yarn을 찾지 못했습니다: {path!r} (--yarn으로 주세요)")
    if not force and info.get("sha256") and _fingerprint(path) != info["sha256"]:
        raise SheetError("내보낸 뒤 .yarn이 바뀌었습니다 — 다시 내보내고 고치세요(그래도 들여오려면 --force)")
    with open(path, encoding="utf-8", newline="") as f:
        doc = parse_yarn(f.read())
    keys = [b.key for b in doc.blocks]
    got = [r["열쇠"] for r in rows]
    if got != keys:
        lost = [k for k in keys if k not in got]
        extra = [k for k in got if k not in keys]
        what = (f"빠진 행 {lost[:5]}" if lost else "") + (f" 모르는 행 {extra[:5]}" if extra else "")
        raise SheetError(f"행이 .yarn과 다릅니다 — 행을 넣거나 지우거나 옮기지 마세요. {what or '순서가 다릅니다'}")

    errors: List[str] = []
    edits: List[Tuple[Block, Optional[List[str]], Optional[List[str]], Optional[str]]] = []
    for b, row in zip(doc.blocks, rows):
        staging, motion = b.split_commands()
        try:
            new_cmds = unfold(row["연출"], units) + unfold(row["모션"], units)
        except SheetError as e:
            errors.append(f"{b.key}: {e}")
            continue
        cmds_changed = new_cmds != staging + motion
        new_checks = [s.strip() for s in (row["확인"] or "").replace("\r\n", "\n").split("\n") if s.strip()]
        checks_changed = new_checks != [c for _, c in b.checks]
        content_changed = False
        new_content = None
        if b.kind == "선택지":
            if not _same(row["대사"], b.text) or not _same(row["화자"], b.speaker):
                errors.append(f"{b.key}: 선택지 줄은 이 보기에서 고치지 않습니다(.yarn에서 고치세요)")
                continue
        elif b.kind == "대사":
            if not _same(row["대사"], b.text) or not _same(row["화자"], b.speaker):
                content_changed = True
                indent = re.match(r"[ \t]*", doc.lines[b.content][0]).group(0)
                speaker, text = (row["화자"] or "").strip(), (row["대사"] or "").strip()
                if not text:
                    errors.append(f"{b.key}: 대사를 비울 수 없습니다(줄을 지우려면 .yarn에서)")
                    continue
                new_content = indent + (f"{speaker}: {text}" if speaker else text) + b.tags
        elif not (_same(row["대사"], None) and _same(row["화자"], None)):
            errors.append(f"{b.key}: 노드 끝 행에는 대사를 쓰지 않습니다")
            continue
        if cmds_changed:
            if b.flow and b.commands and any(b.commands[0][0] < f < b.commands[-1][0] for f in b.flow):
                errors.append(f"{b.key}: 흐름 커맨드(<<if>> 등)와 섞인 행의 연출은 이 보기에서 고치지 않습니다")
                continue
            if known_commands is not None:
                bad = sorted({c.split(" ", 1)[0] for c in new_cmds} - set(known_commands))
                if bad:
                    errors.append(f"{b.key}: 모르는 커맨드 {', '.join(repr(x) for x in bad)}")
                    continue
        if cmds_changed or checks_changed or content_changed:
            edits.append((b, new_cmds if cmds_changed else None, new_checks if checks_changed else None, new_content))
    if errors:
        raise SheetError("들여오지 않았습니다:\n  " + "\n  ".join(errors))

    lines = list(doc.lines)
    for b, cmds, checks, content in reversed(edits):
        lines[b.start:b.end] = _rewrite(doc, b, cmds, checks, content)
    return "".join(t + e for t, e in lines), [b.key for b, *_ in edits]


def _rewrite(doc: YarnDoc, b: Block, cmds: Optional[List[str]], checks: Optional[List[str]],
             content: Optional[str]) -> List[Tuple[str, str]]:
    nl = doc.newline
    if b.content is not None:
        indent = re.match(r"[ \t]*", doc.lines[b.content][0]).group(0)
    elif b.commands:
        indent = re.match(r"[ \t]*", doc.lines[b.commands[0][0]][0]).group(0)
    else:
        indent = ""
    cmd_lines = [(f"{indent}<<{c}>>", nl) for c in cmds] if cmds is not None else []
    check_lines = [(f"{indent}{CHECK_PREFIX} {c}", nl) for c in checks] if checks is not None else []
    cmd_at = {i for i, _ in b.commands}
    check_at = {i for i, _ in b.checks}
    out: List[Tuple[str, str]] = []
    if checks is not None and not check_at:
        out += check_lines
    for i in range(b.start, b.end):
        if cmds is not None and i in cmd_at:
            if i == b.commands[0][0]:
                out += cmd_lines
            continue
        if checks is not None and i in check_at:
            if i == b.checks[0][0]:
                out += check_lines
            continue
        if i == b.content:
            if cmds is not None and not cmd_at:
                out += cmd_lines
            out.append((content, doc.lines[i][1]) if content is not None else doc.lines[i])
            continue
        out.append(doc.lines[i])
    if b.content is None and cmds is not None and not cmd_at:
        out += cmd_lines
    return out
