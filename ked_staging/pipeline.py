"""추출 한 번 — 영상(+ 게임텍스트추출기 엑셀) → 줄 · 대사 · 무대 관측 · 커맨드 → staging.json · .yarn.

  1. 줄 경계(lines)        대사창 본문 잉크로 줄 머리 · 다 찍힘 · 끝
  2. 대사 글자(gte)         엑셀 줄을 읽은 순간으로 줄에 붙인다(없으면 자리만)
  3. 관측(observe)          줄마다 끝 직전 정착 프레임 몇 장의 평균에서 배경 · 인물 · 표정 · 자리
  4. 역해석(inverse)        앞 줄과의 차이 → 정본 커맨드
  5. 출력(emit)
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .catalog import Catalog
from .dialogue import BoxLayout
from .emit import staging_document, write_json, write_yarn
from .gte import LineText, attach, read_xlsx
from .inverse import build_script
from .lines import LineSpan, detect_lines, track_ink
from .observe import ObservedStage, Observer
from .oracle import Oracle

SETTLE_FRAMES = 3          # 정착 프레임 평균 장수
SETTLE_LEAD = 0.05         # 다음 줄 머리보다 이만큼 앞에서 끝낸다(초)
MAX_TIMING_WINDOW = 5.5    # 시간을 잴 때 줄 머리부터 보는 길이(초) — 120fr보다 조금 길게
FLASH_WINDOW = 0.6         # 플래시를 찾는 줄 앞머리(초)


def settled_times(span: LineSpan, fps: float, frames: int = SETTLE_FRAMES) -> List[float]:
    """줄 끝 직전 프레임들 — 다 찍힌 뒤 · 끝 직전. 줄이 너무 짧으면 끝 직전 한 장."""
    end = span.end - SETTLE_LEAD
    start = max(span.head, span.typed)
    ts = [end - k / fps for k in range(frames)]
    ts = [t for t in ts if t >= start] or [max(span.head, end)]
    return sorted(ts)


def grab_frames(video: str, wanted: Sequence[Sequence[float]]) -> List[np.ndarray]:
    """줄마다 원하는 시각들의 프레임을 한 번 훑어 모아 평균한다."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"영상을 열지 못했습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    flat = sorted((t, i) for i, ts in enumerate(wanted) for t in ts)
    sums: List[Optional[np.ndarray]] = [None] * len(wanted)
    counts = [0] * len(wanted)
    k = 0
    index = 0
    last = None
    while k < len(flat):
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and index > 0:
            t = index / fps
        index += 1
        last = frame
        while k < len(flat) and flat[k][0] <= t + 0.5 / fps:
            i = flat[k][1]
            sums[i] = frame.astype(np.float32) if sums[i] is None else sums[i] + frame
            counts[i] += 1
            k += 1
    cap.release()
    out = []
    for s, c in zip(sums, counts):
        if s is None:
            out.append(last.astype(np.float32) if last is not None else None)
        else:
            out.append(s / c)
    return out


def stream_windows(video: str, windows: Sequence[Tuple[float, float]], size: Tuple[int, int]):
    """겹치지 않고 시간 순인 구간들의 프레임을 한 번 훑어 구간마다 내준다 → (구간 번호, [(구간 시작부터 초, 분석 배율 float32)])."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"영상을 열지 못했습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    order = sorted(range(len(windows)), key=lambda i: windows[i][0])
    k = 0
    bucket: List[Tuple[float, np.ndarray]] = []
    index = 0
    while k < len(order):
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and index > 0:
            t = index / fps
        index += 1
        while k < len(order) and t >= windows[order[k]][1]:
            yield order[k], bucket
            bucket = []
            k += 1
        if k < len(order) and t >= windows[order[k]][0]:
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.float32)
            bucket.append((t - windows[order[k]][0], small))
    if k < len(order):
        yield order[k], bucket
        for i in order[k + 1:]:
            yield i, []
    cap.release()


def measure_motion(video: str, oracle: Oracle, catalog: Catalog, tuning_dir: str, observer: Observer,
                   spans: Sequence[LineSpan], stages: Sequence[ObservedStage], script, log=print,
                   timing: bool = True, vocab: bool = True, settled: Optional[Sequence[np.ndarray]] = None,
                   drafts: Optional[list] = None, node: str = "Extracted"
                   ) -> Tuple[List[list], List[list], List[Optional[tuple]]]:
    """P2 · P3 — 줄마다 (1) 흐름(움직임 · 투명도)의 시작 · 길이 · 이징을 재고 (2) 정착 화면에 남지 않는 움직임
    (제자리 모션 · 슬라이드)을 찾아 script.lines를 고친다. 영상은 한 번만 훑는다. → (줄마다 흐름, 줄마다 찾은 것)

    drafts를 주면 어휘로 설명 못 한 제자리 움직임을 재서 모션 초안(motiondraft)을 거기에 더한다(P4)."""
    from .ease import EaseTable
    from .inverse import BG_KEY
    from .timing import TimingMeter, apply_timing, channels_of
    from .vocab import TransientFinder, apply_findings
    from .effects import EffectDetector, EffectPresets, VignetteMasks
    from .tokens import parse_duration
    table = EaseTable.load(os.path.join(tuning_dir, "ease-golden.json"))
    presets = EffectPresets.load(tuning_dir)
    meter = TimingMeter(oracle, catalog, observer.painter, observer.mask, table, presets, observer.px_scale)
    finder = TransientFinder(oracle, catalog, observer.painter, observer.mask, observer.motions)
    slot_of = script.slots

    def chars(st: Optional[ObservedStage]) -> set:
        return set() if st is None else {slot_of[s.character] for s in st.slots if s.character in slot_of}

    def visible(st: Optional[ObservedStage]) -> set:
        keys = chars(st)
        if st is not None and st.background:
            keys.add(BG_KEY)
        return keys

    def recolor_of(k: int) -> dict:
        """줄 머리에서 캐릭터 비주얼이 바뀌는 슬롯 → (앞, 뒤, 바뀌는 초). 길이는 그 커맨드의 생략값(추출은 길이를 안 잰다)."""
        prev_vis = stages[k - 1].visual if k else {}
        cur_vis = stages[k].visual
        out = {}
        for c in set(prev_vis) | set(cur_vis):
            a, b = prev_vis.get(c), cur_vis.get(c)
            if a == b or c not in slot_of:
                continue
            name = "char_visual_clear" if b is None else f"char_visual_{b[0]}"
            try:
                secs = parse_duration(oracle.param_default(name, "duration") or "6fr")
            except ValueError:
                secs = 0.25
            out[slot_of[c]] = (a, b, secs)
        return out

    header = list(script.header)
    work = [header + list(script.lines[0])] + [list(l) for l in script.lines[1:]]
    plans, windows = [], []
    for k, span in enumerate(spans):
        prev = stages[k - 1] if k else None
        shot_changed = (prev.shot if prev else None) != stages[k].shot
        chans = channels_of(work[k], visible(prev), visible(stages[k]), shot_changed) if timing else []
        plans.append(chans)
        wanted = bool(chans) or (vocab and bool(chars(prev) | chars(stages[k])))
        if wanted:
            windows.append((span.head, min(span.end, span.head + MAX_TIMING_WINDOW)))
        elif vocab:
            windows.append((span.head, min(span.end, span.head + FLASH_WINDOW)))   # 플래시만 본다
        else:
            windows.append((span.head, span.head))
    timings: List[list] = [[] for _ in spans]
    found: List[list] = [[] for _ in spans]
    flashes: List[Optional[tuple]] = [None for _ in spans]
    from .timing import compose
    fx = EffectDetector(presets, observer.mask, observer.px_scale)
    vmasks = VignetteMasks(presets)
    for k, frames in stream_windows(video, windows, observer.size):
        if not frames:
            continue
        # 앞 줄과 이 줄에 같은 비네트가 걸려 있으면 걷어 내고 잰다(효과는 근사).
        vig = stages[k].vignette
        steady = vig is not None and k > 0 and stages[k - 1].vignette == vig
        raw_frames = frames
        if steady:
            frames = [(t, vmasks.undo(f, vig)) for t, f in frames]
        if plans[k]:
            meter.measure_line(work[:k], work[k], plans[k], frames, recolor_of(k))
            work[k] = apply_timing(work[k], plans[k], oracle)
            timings[k] = plans[k]
            log(f"  {k:3d} 시간: " + ", ".join(
                f"{c.target}/{c.kind} {int(round(c.start * 24))}+{int(round(c.duration * 24))}fr {c.ease}"
                for c in plans[k]))
        if vocab:
            # 플래시 — 줄 앞머리에서 화면 전체가 앞 · 뒤 화면보다 확 밝아졌다 사그라드는가(근사 효과).
            # 화면 그대로 본다(플래시는 비네트 위에 얹힌다 — 비네트를 걷어 내면 가장자리 플래시가 부푼다).
            early = [(t, f) for t, f in raw_frames if t < FLASH_WINDOW]
            before_img = None
            if k > 0 and settled is not None and settled[k - 1] is not None:
                before_img = cv2.resize(settled[k - 1], observer.size, interpolation=cv2.INTER_AREA).astype(np.float32)
            if len(early) >= 3 and before_img is not None:
                snaps = oracle.timeline(work[:k] + [work[k]], [t for t, _ in early])["snapshots"]
                models = [compose(sn, catalog, observer.painter)[0] for sn in snaps]
                hit = fx.flash(early, models, before_img)
                if hit and hit[2] > 0.5:
                    flashes[k] = (hit[0], hit[1])
                    log(f"  {k:3d} 효과: screen_flash {hit[0]} {hit[1]:g}")
        if vocab:
            prev_st = stages[k - 1] if k else None
            before, after = chars(prev_st), chars(stages[k])
            moving = {c.target for c in plans[k] if c.kind == "move"}
            # 캐릭터 비주얼이 바뀐 인물은 줄 안에서 색이 서서히 바뀐다 — 지나가는 움직임 찾기에서 뺀다.
            recolored = set(recolor_of(k))
            static = sorted((before & after) - moving - recolored)
            hits = finder.find(work[:k], work[k], frames, static, sorted(after - before), sorted(before - after))
            screen_fx = flashes[k] is not None or (prev_st is not None and prev_st.vignette != stages[k].vignette)
            if screen_fx:
                hits = [h for h in hits if h.kind != "unexplained"]   # 화면 전체 효과가 남긴 어긋남이다
            if drafts is not None:
                from .motiondraft import measure as measure_draft, to_motion
                for h in hits:
                    if h.kind != "unexplained" or h.slot not in static:
                        continue
                    sample = measure_draft(oracle, catalog, observer.painter, observer.mask, work[:k], work[k],
                                           frames, h.slot)
                    name = f"draft_l{k:03d}_{h.slot}"
                    motion = to_motion(name, sample) if sample else None
                    if motion is not None:
                        drafts.append({"line": k, "slot": h.slot, "name": name, "motion": motion})
                        h.note = f"{h.note} → 모션 초안 {name}(.motion-authoring.json)"
                        log(f"  {k:3d} 모션 초안: {name}")
            if hits:
                work[k] = apply_findings(work[k], hits)
                found[k] = hits
                log(f"  {k:3d} 어휘: " + ", ".join(f"{h.slot} {h.kind} {h.commands[-1]['args'][1:] if h.commands else ''}"
                                                 for h in hits))
    script.lines[0] = work[0][len(header):]
    for k in range(1, len(work)):
        script.lines[k] = work[k]
    return timings, found, flashes


def _layers(snapshot, catalog: Catalog, painter) -> list:
    out = []
    for sl in snapshot.slots:
        if not sl.get("quad") or float(sl.get("alpha", 0)) <= 0.003:
            continue
        entry = catalog.background(sl.get("sprite") or "") if sl.get("background") else \
            catalog.portrait(sl.get("character") or "", sl.get("variant") or "a", sl.get("emotion") or "")
        if entry is None:
            continue
        color, alpha = painter.layer(entry, np.asarray(sl["quad"], np.float64), bool(sl.get("toolMirror")))
        out.append((sl["key"], sl.get("layer") or "mid", color, alpha * float(sl.get("alpha", 1.0)),
                    bool(sl.get("background"))))
    return out


def _insert_head(line: List[dict], cmds: List[dict]) -> List[dict]:
    """줄 머리 무리(첫 pause 앞)의 끝에 끼운다."""
    i = next((j for j, c in enumerate(line) if c["name"] == "pause"), len(line))
    return line[:i] + cmds + line[i:]


def detect_effects(oracle: Oracle, catalog: Catalog, tuning_dir: str, observer: Observer, frames: Sequence,
                   script, stages: Sequence[ObservedStage], flashes: Sequence, log=print,
                   averaged: Optional[Sequence[int]] = None) -> List[List[str]]:
    """P3 — 줄마다 화면 효과(근사)를 정해 줄 커맨드에 넣는다. 줄마다 메모.

    캐릭터 비주얼 · 비네트는 관측이 이미 정했다(그 효과를 입혀야 무대가 맞는다). 여기서는 효과를 입힌 예측과 정착 화면을
    견줘 노이즈(고주파 잔차) · 층 블러를 더 찾고, 시간 재기에서 찾은 플래시를 붙인다."""
    from .effects import EffectDetector, EffectPresets, ObservedEffects, VignetteMasks, effect_commands, visual_layer
    presets = EffectPresets.load(tuning_dir)
    det = EffectDetector(presets, observer.mask, observer.px_scale)
    vmasks = VignetteMasks(presets)
    slot_of = script.slots
    header = list(script.header)
    work = [header + list(script.lines[0])] + [list(l) for l in script.lines[1:]]
    snaps = oracle.fold(work)
    smalls, models, layer_sets, stds, observed, raws = [], [], [], [], [], []
    for frame, snap, st in zip(frames, snaps, stages):
        cur = ObservedEffects()
        cur.visual = {slot_of[c]: v for c, v in st.visual.items() if c in slot_of}
        cur.vignette = st.vignette
        raw = cv2.resize(frame, observer.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        small = vmasks.undo(raw, cur.vignette)
        layers = []
        for key, lay, c, a, _bg in _layers(snap, catalog, observer.painter):
            v = cur.visual.get(key)
            if v and v[0] in presets.visual:
                c, a = visual_layer(c, a, presets.visual[v[0]], v[1], observer.px_scale)
            layers.append((key, lay, c, a))
        model = np.zeros_like(small)
        for _k, _l, c, a in layers:
            model = c * a[:, :, None] + model * (1 - a[:, :, None])
        smalls.append(small)
        models.append(model)
        layer_sets.append((layers, snap))
        # 노이즈는 비네트 위에 얹힌다 — 걷어 낸 화면(가장자리가 부푼다)이 아니라 화면 그대로와 비네트를 입힌 예측을 견준다.
        raws.append((raw, vmasks.apply(model, cur.vignette)))
        stds.append(det.residual_sigma(*raws[-1]))
        observed.append(cur)
    floor = float(np.percentile(stds, 25)) if stds else 0.0
    prev = ObservedEffects()
    notes: List[List[str]] = []
    for k, (small, model, (layers, snap)) in enumerate(zip(smalls, models, layer_sets)):
        cur = observed[k]
        # 정착 프레임 n장을 평균했다 — 프레임마다 따로인 잡음은 √n만큼 약해져 있다(합성 근사 기준).
        n_avg = averaged[k] if averaged and k < len(averaged) else 1
        for layer in sorted({lay for _k, lay, _c, _a in layers}):
            hit = det.blur(small, layers, layer)
            if hit:
                cur.blur[layer] = hit[0]
        noisy_ref = raws[k][1]
        if cur.blur:
            # 층 블러를 입힌 예측과 견준다 — 흐려진 층의 잔결이 노이즈로 보이지 않게.
            noisy_ref = vmasks.apply(det.compose_blur(layers, cur.blur), cur.vignette)
        hit = det.noise(raws[k][0], noisy_ref, floor, frames[k].shape[1] / observer.size[0] * np.sqrt(n_avg))
        if hit:
            cur.noise = hit
        cur.flash = flashes[k] if k < len(flashes) else None
        cmds = effect_commands(prev, cur)
        line_notes = []
        if cmds:
            work[k] = _insert_head(work[k], cmds)
            line_notes.append("효과는 근사(런타임 셰이더 모양과 다를 수 있다): " + ", ".join(c["name"] for c in cmds))
            log(f"  {k:3d} 효과: " + ", ".join(" ".join([c["name"]] + c["args"]) for c in cmds))
        notes.append(line_notes)
        prev = ObservedEffects(dict(cur.visual), cur.vignette, cur.noise, dict(cur.blur), None)
    script.lines[0] = work[0][len(header):]
    for k in range(1, len(work)):
        script.lines[k] = work[k]
    return notes


def extract(video: str, oracle: Oracle, catalog: Catalog, tuning_dir: str, out_dir: str,
            text_xlsx: Optional[str] = None, node: str = "Extracted", preset: str = "bottom",
            log: Callable[[str], None] = print, timing: bool = True, vocab: bool = True,
            motions_path: Optional[str] = None, layout: Optional[BoxLayout] = None) -> dict:
    """layout: 대사창 자리(기본: 런타임 surface-layout 프리셋). 다른 게임은 게임텍스트추출기 프로필에서(P4)."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    if layout is None:
        layout = BoxLayout.from_tuning(tuning_dir, int(oracle.width), int(oracle.height), preset)

    track = track_ink(video, layout)
    spans = detect_lines(track)
    log(f"줄 {len(spans)}개 (지움 {sum(s.kind == 'clear' for s in spans)} · 등장 {sum(s.kind == 'onset' for s in spans)})")

    if text_xlsx:
        rows = read_xlsx(text_xlsx)
        texts = attach([s.head for s in spans], [s.end for s in spans], rows)
        log(f"대사 엑셀 {len(rows)}줄 → 붙은 줄 {sum(bool(t.rows) for t in texts)}/{len(spans)}")
    else:
        texts = [LineText(review=["대사 글자 없음(게임텍스트추출기 엑셀을 --text로 주면 붙는다)"]) for _ in spans]

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    wanted = [settled_times(s, fps) for s in spans]
    frames = grab_frames(video, wanted)

    if motions_path is None:
        guess = os.path.join(os.path.dirname(os.path.abspath(tuning_dir)), "Assets", "Dialogue", "motions.json")
        motions_path = guess if os.path.isfile(guess) else None
    presets = None
    if vocab:
        from .effects import EffectPresets
        presets = EffectPresets.load(tuning_dir)
    observer = Observer(oracle, catalog, tuning_dir, layout.stage_bottom(), motions_path=motions_path,
                        effects=presets)
    stages: List[ObservedStage] = []
    previous = None
    for i, (span, frame) in enumerate(zip(spans, frames)):
        st = observer.observe(frame, float(np.mean(wanted[i])), previous)
        stages.append(st)
        previous = st
        cast = ", ".join(f"{s.character}/{s.variant}/{s.emotion}@{s.place}-{s.depth}" for s in st.slots)
        log(f"  {i:3d} {span.head:7.2f}s  배경 {st.background or '없음'} · {cast or '인물 없음'}")

    script = build_script(stages)
    timings: List[list] = [[] for _ in spans]
    found: List[list] = [[] for _ in spans]
    flashes: List[Optional[tuple]] = [None for _ in spans]
    drafts: list = []
    if (timing or vocab) and spans:
        timings, found, flashes = measure_motion(video, oracle, catalog, tuning_dir, observer, spans, stages, script,
                                                 log, timing, vocab, frames, drafts=drafts, node=node)
    fx_notes: List[List[str]] = [[] for _ in spans]
    if vocab and spans:
        fx_notes = detect_effects(oracle, catalog, tuning_dir, observer, frames, script, stages, flashes, log,
                                  [len(w) for w in wanted])
    lines = []
    for i, (span, text, st, cmds, notes) in enumerate(zip(spans, texts, stages, script.lines, script.notes)):
        notes = notes + [f"{c.target} {c.kind}: {c.note}" for c in timings[i] if c.note]
        notes = notes + [h.note for h in found[i] if h.note] + fx_notes[i]
        lines.append({
            "index": i,
            "head": round(span.head, 4), "typed": round(span.typed, 4), "end": round(span.end, 4),
            "boundary": span.kind,
            "speaker": text.speaker, "text": text.text, "textRows": text.rows, "textNotes": text.notes,
            "observed": st.to_json(),
            "commands": cmds,
            "timing": [c.to_json() for c in timings[i]],
            "transients": [{"slot": h.slot, "kind": h.kind, "commands": h.commands, "gain": round(h.gain, 3)}
                           for h in found[i]],
            "review": text.review + notes,
        })
    settings = {"boxPreset": layout.preset, "catalog": catalog.source, "analysisScale": observer.scale,
                "oracleSize": [oracle.width, oracle.height]}
    doc = staging_document(video, node, script, lines, settings, text_xlsx)
    json_path = os.path.abspath(os.path.join(out_dir, f"{stem}.staging.json"))
    yarn_path = write_yarn(os.path.abspath(os.path.join(out_dir, f"{stem}.yarn")), node, script, lines, video, timing)
    doc["files"] = {"staging": json_path, "yarn": yarn_path}
    if drafts:
        from .motiondraft import peak, write_authoring
        motions_out = write_authoring(os.path.abspath(os.path.join(out_dir, f"{stem}.motion-authoring.json")),
                                      [d["motion"] for d in drafts])
        doc["files"]["motions"] = motions_out
        doc["motionDrafts"] = [{"line": d["line"], "slot": d["slot"], "name": d["name"],
                                "peak": [round(v, 2) for v in peak(d["motion"])]} for d in drafts]
        log(f"→ {motions_out} (모션 초안 {len(drafts)})")
    write_json(json_path, doc)
    log(f"→ {json_path}\n→ {yarn_path}")
    return doc
