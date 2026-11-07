"""출력 — *.staging.json(추출의 모든 근거)과 .yarn(시험장 · 빠른 재생용 초안).

staging.json이 중심이다. 컴포지터로 가져갈 때도 이것을 쓴다(설계 결정 A: 추출기는 staging.json까지).
.yarn은 같은 내용을 런타임 대본 모양으로 편 것이다 — 게임에 들어갈 Yarn은 컴포지터만 낸다(K4).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from typing import List, Optional, Sequence

from .inverse import Script

SCHEMA = "ked-staging/1"
_YARN_ESCAPE = re.compile(r"([#{}\[\]<>\\])")


def yarn_text(text: str) -> str:
    """대사 한 줄을 Yarn 본문으로 — 줄바꿈은 띄어쓰기로, 특수 문자는 백슬래시로."""
    t = " ".join((text or "").split())
    t = _YARN_ESCAPE.sub(r"\\\1", t)
    return t.replace("//", "/\\/")


def command_text(cmd: dict) -> str:
    return "<<" + " ".join([cmd["name"]] + [str(a) for a in cmd.get("args", [])]) + ">>"


def node_name(text: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]", "_", text or "Extracted").strip("_") or "Extracted"
    return name if name[0].isalpha() else "N_" + name


def write_yarn(path: str, node: str, script: Script, lines: Sequence[dict], source: str = "") -> str:
    out: List[str] = [f"title: {node_name(node)}", "---"]
    out.append(f"// ked-staging-extractor P1 초안 — {os.path.basename(source)}. 시간 인자는 생략값이다(P2에서 잰다).")
    out += [command_text(c) for c in script.header]
    for i, line in enumerate(lines):
        out.append("")
        for note in line.get("review", []):
            out.append(f"// 확인: {note}")
        out += [command_text(c) for c in line.get("commands", [])]
        text = yarn_text(line.get("text") or "") or f"(대사 {i + 1})"
        speaker = yarn_text(line.get("speaker") or "")
        out.append(f"{speaker}: {text}" if speaker else text)
    out.append("===")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    return path


def staging_document(video: str, node: str, script: Script, lines: Sequence[dict], settings: dict,
                     text_source: Optional[str]) -> dict:
    return {
        "schema": SCHEMA,
        "createdAt": _dt.datetime.now().isoformat(timespec="seconds"),
        "video": os.path.abspath(video),
        "textSource": os.path.abspath(text_source) if text_source else None,
        "node": node_name(node),
        "settings": settings,
        "slots": script.slots,
        "header": script.header,
        "lines": list(lines),
    }


def write_json(path: str, doc: dict) -> str:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path
