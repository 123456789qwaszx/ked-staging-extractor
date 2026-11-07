"""ked-stage-oracle 클라이언트 — 컴포지터 저장소의 Vn.StageOracle을 띄워 두고 JSON 줄로 묻는다.

오라클은 "이 커맨드면 화면에 무엇이 어디에 있나"를 런타임 코어로 계산한다(규칙 5 — 화면 예측은 코어가 한다).
실행 파일 찾는 순서:
  1. 인자 / 환경 변수 KED_STAGE_ORACLE (exe · dll · 실행 파일 경로)
  2. 형제 저장소 ../ked-story-compositor/src/Vn.StageOracle/bin/{Release,Debug}/net10.0/ked-stage-oracle(.exe|.dll)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

PROTOCOL = "2"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


class OracleError(RuntimeError):
    pass


def find_oracle(explicit: Optional[str] = None) -> Optional[List[str]]:
    """실행 명령(리스트)을 돌려준다. 못 찾으면 None."""
    candidates: List[str] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("KED_STAGE_ORACLE"):
        candidates.append(os.environ["KED_STAGE_ORACLE"])
    sibling = os.path.join(os.path.dirname(_REPO), "ked-story-compositor", "src", "Vn.StageOracle", "bin")
    for config in ("Release", "Debug"):
        base = os.path.join(sibling, config, "net10.0", "ked-stage-oracle")
        candidates += [base + ".exe", base, base + ".dll"]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        if path.lower().endswith(".exe") and os.name != "nt":
            continue  # Windows 빌드를 다른 OS에서 본 것(예: 리눅스 VM에 붙인 폴더) — 실행할 수 없다
        if path.lower().endswith(".dll"):
            dotnet = shutil.which("dotnet")
            if dotnet:
                return [dotnet, path]
            continue
        return [path]
    return None


@dataclass
class Snapshot:
    """오라클이 낸 정착 화면 한 장 — 슬롯은 그리기 순서(먼 것 먼저)다."""
    slots: List[Dict[str, Any]]
    shot: Dict[str, Any]
    unhandled: List[Dict[str, Any]]

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Snapshot":
        return cls(list(data.get("slots", [])), dict(data.get("shot", {})), list(data.get("unhandled", [])))

    def visible_slots(self, min_alpha: float = 0.01) -> List[Dict[str, Any]]:
        return [s for s in self.slots if s.get("quad") and s.get("alpha", 0) >= min_alpha]


class Oracle:
    """`with Oracle(tuning_dir) as o: o.fold_yarn([...])`"""

    def __init__(self, tuning_dir: str, motions_path: Optional[str] = None, command: Optional[Sequence[str]] = None,
                 timeout: float = 60.0):
        cmd = list(command) if command else find_oracle()
        if not cmd:
            raise OracleError(
                "ked-stage-oracle을 찾지 못했습니다. ked-story-compositor에서 "
                "`dotnet build src/Vn.StageOracle -c Release`로 빌드하거나 KED_STAGE_ORACLE에 경로를 지정하세요.")
        args = cmd + ["serve", "--tuning", tuning_dir]
        if motions_path:
            args += ["--motions", motions_path]
        self._proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", bufsize=1)
        self._next_id = 0
        self._lock = threading.Lock()
        self._timeout = timeout
        self.stderr_lines: List[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        info = self.request({"op": "ping"})
        if info.get("protocol") != PROTOCOL:
            self.close()
            raise OracleError(f"오라클 프로토콜이 다릅니다: {info.get('protocol')} (기대 {PROTOCOL})")
        self.width = float(info["width"])
        self.height = float(info["height"])
        self.problems: List[str] = list(info.get("problems", []))

    def _drain_stderr(self) -> None:
        for line in self._proc.stderr:  # type: ignore[union-attr]
            self.stderr_lines.append(line.rstrip("\n"))

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._proc.poll() is not None:
                raise OracleError("오라클이 종료됐습니다: " + " / ".join(self.stderr_lines[-5:]))
            self._next_id += 1
            payload = dict(payload, id=self._next_id)
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")  # type: ignore[union-attr]
            self._proc.stdin.flush()  # type: ignore[union-attr]
            line = self._proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            raise OracleError("오라클이 답하지 않았습니다: " + " / ".join(self.stderr_lines[-5:]))
        response = json.loads(line)
        if response.get("id") != payload["id"]:
            raise OracleError(f"응답 순서가 어긋났습니다: {response.get('id')} != {payload['id']}")
        if not response.get("ok"):
            raise OracleError(response.get("error", "알 수 없는 오류"))
        return response

    # ── 편의 함수 ───────────────────────────────────────────────

    def groups(self, path: str) -> Dict[str, Any]:
        return self.request({"op": "groups", "path": os.path.abspath(path)})

    def fold_yarn(self, paths: Iterable[str], continuous: bool = True, node: Optional[str] = None) -> Dict[str, Any]:
        """yarn 파일들을 재생 순서대로 접는다 — 줄(대사)마다 정착 스냅샷이 붙는다."""
        req: Dict[str, Any] = {"op": "foldYarn", "paths": [os.path.abspath(p) for p in paths],
                               "continuous": continuous}
        if node:
            req["node"] = node
        return self.request(req)

    def fold(self, lines: Sequence[Sequence[Dict[str, Any]]]) -> List[Snapshot]:
        """커맨드 열(줄마다 [{name, args}])을 처음 상태에서 접는다 — 후보 커맨드를 시험할 때 쓴다."""
        res = self.request({"op": "fold", "lines": [{"commands": list(cmds)} for cmds in lines]})
        return [Snapshot.from_json(s) for s in res["snapshots"]]

    def timeline(self, lines: Sequence[Sequence[Dict[str, Any]]], times: Sequence[float],
                 line: Optional[int] = None) -> Dict[str, Any]:
        """줄 하나의 시간 축 — lines를 처음 상태에서 접되 line번째(기본 마지막) 줄은 times(줄 머리부터 초)마다
        스냅샷을 뜬다. → {schedule: [{index, name, args, start, duration, ease, wait, tween, fade}], end, snapshots}"""
        req: Dict[str, Any] = {"op": "timeline", "lines": [{"commands": list(c)} for c in lines],
                               "times": [float(t) for t in times]}
        if line is not None:
            req["line"] = int(line)
        res = self.request(req)
        res["snapshots"] = [Snapshot.from_json(s) for s in res["snapshots"]]
        return res

    def commands(self) -> Dict[str, Dict[str, Any]]:
        """커맨드 카탈로그(이름 → {id, parameters: [{name, type, required, default}]}). 인자 자리 = 파라미터 순서."""
        if not hasattr(self, "_commands"):
            res = self.request({"op": "commands"})
            self._commands = {c["name"]: c for c in res["commands"]}
        return self._commands

    def param_index(self, command: str, type_name: str) -> Optional[int]:
        """command의 type_name(예: "duration" · "ease") 파라미터가 몇 번째 인자인가."""
        info = self.commands().get(command)
        if not info:
            return None
        for i, p in enumerate(info["parameters"]):
            if p.get("type") == type_name:
                return i
        return None

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
                self._proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
                self._proc.wait()
        # 파이프를 모두 닫는다 — 안 닫으면 파이썬이 종료할 때 ResourceWarning을 낸다.
        self._stderr_thread.join(timeout=2)
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except OSError:
                pass

    def __enter__(self) -> "Oracle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def route_lines(folded: Dict[str, Any]) -> List[Dict[str, Any]]:
    """foldYarn 결과를 재생 순서의 줄 목록으로 편다. 각 줄: node, index, text, lineId, commands, snapshot."""
    out: List[Dict[str, Any]] = []
    for node in folded.get("nodes", []):
        for line in node.get("lines", []):
            out.append(dict(line, node=node.get("name"), snapshot=Snapshot.from_json(line["snapshot"])))
    return out
