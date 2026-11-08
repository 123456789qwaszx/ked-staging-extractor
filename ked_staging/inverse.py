"""역해석 — 줄마다 관측한 무대를 앞 줄과 비교해 정본 커맨드로 바꾼다.

정본 패턴(샘플 대본과 같은 모양):
  머리(노드 셋업)  shot_reset · bg_spawn bg_main <첫 배경> · 등장 순서대로 slot sN stage00 mid + cast sN <인물> <변형> <표정>
  배경            처음 보임 → (bg_sprite) + bg_fade_in · 바뀜 → bg_sprite · 사라짐 → bg_fade_out
  퇴장            fade_out sN
  인물            pose(변형) · face(표정) · mirror left/right · place_<자리> + size(크기) [+ move_by] · 등장이면 fade_in
  이동만 바뀜      move_by sN <차이>

자리는 관측이 이미 어휘(place × focus × size)로 골라 두었다. place는 그 순간 크기와 상관없이 focus를 화면 지점에
세우고 size는 focus를 지키며 키우므로, 앞 상태가 어떻든 place → size 두 줄이면 같은 자리에 선다(그 뒤 move_by).
시간 인자는 P1에서 생략값이다 — 숨어 있을 때 바꾸는 자리는 0fr, 보이는 채로 바꾸면 12fr, 등 · 퇴장은 18fr.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .observe import ObservedSlot, ObservedStage, emotion_arg
from .tokens import unit_token

BG_KEY = "bg_main"
APPEAR = "18fr"
MOVE = "12fr"
SNAP = "0fr"
BG_FADE = "24fr"


def _cmd(name: str, *args) -> dict:
    return {"name": name, "args": [str(a) for a in args]}


def _units(u: float) -> str:
    return unit_token(u * 40.0, stage_width=1920.0)


@dataclass
class _Actor:
    slot: str
    variant: str
    emotion: str
    visible: bool = False
    place: Optional[Tuple[str, str]] = None          # (place, focus)
    depth: str = "mid"
    move: Tuple[float, float] = (0.0, 0.0)
    mirror: bool = False
    extras: Tuple[str, ...] = ()


@dataclass
class Script:
    header: List[dict]
    lines: List[List[dict]]
    slots: Dict[str, str]                       # 인물 → 슬롯 키
    notes: List[List[str]] = field(default_factory=list)


def first_appearances(stages: Sequence[ObservedStage]) -> List[ObservedSlot]:
    seen: Dict[str, ObservedSlot] = {}
    for st in stages:
        for s in st.slots:
            seen.setdefault(s.character, s)
    return list(seen.values())


def build_script(stages: Sequence[ObservedStage]) -> Script:
    header: List[dict] = [_cmd("shot_reset", SNAP)]
    first_bg = next((st.background for st in stages if st.background), None)
    if first_bg:
        header.append(_cmd("bg_spawn", BG_KEY, first_bg))
    actors: Dict[str, _Actor] = {}
    slots: Dict[str, str] = {}
    for i, s in enumerate(first_appearances(stages), 1):
        key = f"s{i}"
        slots[s.character] = key
        header += [_cmd("slot", key, "stage00", "mid"), _cmd("cast", key, s.character, s.variant, emotion_arg(s.emotion))]
        actors[s.character] = _Actor(key, s.variant, s.emotion)

    bg_sprite = first_bg
    bg_visible = False
    shot = None
    out_lines: List[List[dict]] = []
    out_notes: List[List[str]] = []
    for st in stages:
        cmds: List[dict] = []
        notes: List[str] = []
        # 카메라 — 줄의 맨 앞(무대 전체가 함께 움직인다)
        if st.shot != shot:
            if st.shot is None:
                cmds.append(_cmd("shot_reset", MOVE))
            else:
                zoom, x, y = st.shot
                cmds.append(_cmd("shot_to", f"{zoom:g}", _units(x), _units(y), MOVE))
            shot = st.shot
        # 배경
        if st.background and not bg_visible:
            if st.background != bg_sprite:
                cmds.append(_cmd("bg_sprite", BG_KEY, st.background))
                bg_sprite = st.background
            cmds.append(_cmd("bg_fade_in", BG_KEY, BG_FADE))
            bg_visible = True
        elif st.background and st.background != bg_sprite:
            cmds.append(_cmd("bg_sprite", BG_KEY, st.background))
            bg_sprite = st.background
        elif not st.background and bg_visible:
            cmds.append(_cmd("bg_fade_out", BG_KEY, BG_FADE))
            bg_visible = False
        # 퇴장
        present = {s.character for s in st.slots}
        for c, a in actors.items():
            if a.visible and c not in present:
                cmds.append(_cmd("fade_out", a.slot, APPEAR))
                a.visible = False
        # 인물(그리기 순서대로 — 먼 것 먼저)
        for s in st.slots:
            a = actors[s.character]
            appearing = not a.visible
            if s.variant != a.variant:
                cmds.append(_cmd("pose", a.slot, s.variant))
                a.variant = s.variant
            if s.emotion != a.emotion:
                cmds.append(_cmd("face", a.slot, emotion_arg(s.emotion)))
                a.emotion = s.emotion
            if not s.emotion_sure:
                notes.append(f"{s.character}: 표정을 화면으로 가리지 못했습니다({s.emotion}로 둠)")
            if s.mirror != a.mirror:
                cmds.append(_cmd("mirror", a.slot, "left" if s.mirror else "right"))
                a.mirror = s.mirror
            dur = SNAP if appearing else MOVE
            first = a.place is None
            place_changed = (s.place, s.focus) != a.place
            depth_changed = s.depth != a.depth
            # place는 앞 이동(move_by)을 지우고 focus를 화면 지점에 세운다. size는 focus를 지키며 키우므로
            # 이동이 남아 있을 때 크기만 바꾸면 처음부터 세운 자리와 달라진다 → 그때는 다시 세운다.
            replace = (first or place_changed
                       or (s.depth != a.depth and (a.move != (0.0, 0.0) or s.move != (0.0, 0.0)))
                       or (s.move == (0.0, 0.0) and a.move != (0.0, 0.0)))   # 이동을 거두는 것은 place로(대본 관례)
            if replace:
                cmds.append(_cmd(f"place_{s.place}", a.slot, s.focus, dur))
                if first or depth_changed:
                    cmds.append(_cmd("size", a.slot, s.depth, s.focus, dur))
                if s.move != (0.0, 0.0):
                    cmds.append(_cmd("move_by", a.slot, _units(s.move[0]), _units(s.move[1]), dur))
                a.place, a.depth, a.move = (s.place, s.focus), s.depth, s.move
            elif s.depth != a.depth:
                cmds.append(_cmd("size", a.slot, s.depth, s.focus, dur))
                a.depth = s.depth
            elif s.move != a.move:
                dx, dy = s.move[0] - a.move[0], s.move[1] - a.move[1]
                cmds.append(_cmd("move_by", a.slot, _units(dx), _units(dy), dur))
                a.move = s.move
            if s.extras != a.extras:
                gained = [m for m in s.extras if m not in a.extras]
                if len(gained) == len(s.extras) - len(a.extras) and all(m in s.extras for m in a.extras):
                    cmds += [_cmd("motion", a.slot, m) for m in gained]
                else:
                    # commit 모션이 남긴 자세를 거둔다 — 샘플 모션(bow)은 초상 회전이라 char_rotate_to 0으로 되돌린다.
                    cmds.append(_cmd("char_rotate_to", a.slot, "0", SNAP if appearing else MOVE))
                    cmds += [_cmd("motion", a.slot, m) for m in s.extras]
                    notes.append(f"{s.character}: 남은 자세(commit 모션)를 char_rotate_to 0으로 거뒀습니다 — 확인")
                a.extras = s.extras
            if appearing:
                cmds.append(_cmd("fade_in", a.slot, APPEAR))
                a.visible = True
        notes += st.notes
        out_lines.append(cmds)
        out_notes.append(notes)
    return Script(header, out_lines, slots, out_notes)
