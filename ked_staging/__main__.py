"""명령줄 — `python -m ked_staging <명령>`.

  catalog     런타임 저장소에서 카탈로그(catalog.json)를 만든다
  predict     yarn 경로를 오라클로 접어 줄마다 예측 화면(PNG)과 스냅샷(JSON)을 낸다
  testbed     시험장 매니페스트의 녹화마다 정렬 · 어긋남을 재서 P0 통과 여부를 낸다
  synth-check 합성 녹화를 만들어 testbed가 제대로 도는지 확인한다(설치 점검)

  scenario    yarn 경로로 P1 시험 녹화(대사창 · 타자기 · 전이)와 정답(truth.json)을 만든다
  extract     영상(+ 게임텍스트추출기 엑셀)에서 줄 · 대사 · 연출을 뽑아 staging.json · .yarn을 낸다
  evaluate    extract 결과를 정답과 줄마다 비교한다(P1 통과 기준: 상태 일치 95%)
  p1-check    scenario → extract → evaluate를 한 번에(P1 점검)
  p2-check    시간 축 시험 녹화 → extract → 상태 · 시간 · 이징 평가(P2 점검)
  gte-profile 게임텍스트추출기용 게임 프로필(대사창 자리)을 낸다

  scan          다른 게임(P4): 영상 → 카탈로그 초안(draft.json — 배경 판 · 인물 조각 · 정체 · 표정 무리)
  draft         초안 보기 · 고치기(카탈로그 툴과 같은 연산)
  catalog-build 초안 → 카탈로그(catalog.json) + 튜닝 겹본(tuning/)
  catalog-originals 그 게임의 원본 스프라이트 · 배경 + scan 초안 → 카탈로그 + 튜닝 겹본(원본 모드)
  p4-check      '모르는 게임' 시험 녹화 → scan → catalog-build → extract → 이름을 다수결로 대응시켜 평가

  sheet         P5: 엑셀 연출 열 ↔ .yarn (export · apply · check — .yarn이 주인, 엑셀은 고치는 보기)

  originals     원본 수집: MediaWiki 카테고리 → 원본 그림 목록(manifest) — 위키 주소는 설정 파일(private)에

런타임 저장소는 --runtime(기본: 이 저장소 옆의 ked-presentation-runtime)이다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Optional

from .catalog import Catalog, import_runtime
from .imageio import write_image
from .oracle import Oracle, OracleError, route_lines
from .render import render

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNTIME = os.path.join(os.path.dirname(_REPO), "ked-presentation-runtime")


def _runtime_paths(runtime: str):
    runtime = os.path.abspath(runtime)
    tuning = os.path.join(runtime, "ExportedTuning")
    motions = os.path.join(runtime, "Assets", "Dialogue", "motions.json")
    if not os.path.isdir(tuning):
        raise SystemExit(f"ExportedTuning 폴더가 없습니다: {tuning} (--runtime을 확인하세요)")
    return runtime, tuning, motions if os.path.isfile(motions) else None


def _catalog(args, runtime: str) -> Catalog:
    if getattr(args, "catalog", None):
        return Catalog.load(args.catalog)
    problems = []
    cat = import_runtime(runtime, problems)
    for p in problems:
        print(f"  주의: {p}")
    return cat


def cmd_catalog(args) -> int:
    runtime, _, _ = _runtime_paths(args.runtime)
    problems = []
    cat = import_runtime(runtime, problems)
    cat.save(args.output)
    for p in problems:
        print(f"  주의: {p}")
    print(f"카탈로그: 초상 {len(cat.portraits)} · 배경 {len(cat.backgrounds)} → {args.output}")
    return 0


def cmd_predict(args) -> int:
    runtime, tuning, motions = _runtime_paths(args.runtime)
    cat = _catalog(args, runtime)
    os.makedirs(args.out, exist_ok=True)
    with Oracle(tuning, motions, command=[args.oracle] if args.oracle else None) as oracle:
        folded = oracle.fold_yarn(args.yarn, continuous=not args.separate)
    lines = route_lines(folded)
    dump = []
    for i, line in enumerate(lines):
        snap = line["snapshot"]
        img = render(snap, cat, folded["width"], folded["height"], scale=args.scale)
        write_image(os.path.join(args.out, f"{i:03d}.png"), img)
        dump.append({"index": i, "node": line["node"], "text": line.get("text"), "commands": line["commands"],
                     "slots": snap.slots, "shot": snap.shot, "unhandled": snap.unhandled})
    with open(os.path.join(args.out, "snapshots.json"), "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    names = sorted({u["name"] for d in dump for u in d["unhandled"]})
    print(f"줄 {len(lines)} → {args.out} (코어가 접지 않은 커맨드: {', '.join(names) or '없음'})")
    return 0


def cmd_testbed(args) -> int:
    from .testbed import Manifest, run_recording, summarize
    manifest = Manifest.load(args.manifest)
    runtime, tuning, motions = _runtime_paths(manifest.runtime)
    cat = _catalog(args, runtime)
    report = {"manifest": manifest.path, "tolerancePx": args.tolerance, "recordings": []}
    ok = True
    with Oracle(tuning, motions, command=[args.oracle] if args.oracle else None) as oracle:
        for n, rec in enumerate(manifest.recordings):
            print(f"[{n + 1}/{len(manifest.recordings)}] {os.path.basename(rec.video)} {rec.note}")
            out = os.path.join(args.out, f"rec{n:02d}") if args.out else None
            results = run_recording(rec, oracle, cat, tuning, out_dir=out, log=lambda m: print("  " + m))
            summary = summarize(results, args.tolerance)
            ok &= bool(summary["pass"])
            for r in results:
                bad = [o for o in r.offsets if not (o.distance <= args.tolerance)]
                if r.start is None or bad:
                    what = "영상에서 못 찾음" if r.start is None else ", ".join(
                        f"{o.key} ({o.dx:+.2f}, {o.dy:+.2f})px" for o in bad)
                    print(f"  상태 {r.index} 줄 {r.lines[0]}–{r.lines[1]} {r.first_text[:24]!r}: {what}")
            print(f"  → 상태 {summary['statesFound']}/{summary['states']} · 슬롯 {summary['withinTolerance']}/"
                  f"{summary['slotsMeasured']} 허용 안 · 최대 {summary['worstPx']:.2f}px · 중앙 {summary['medianPx']:.3f}px")
            report["recordings"].append({
                "video": rec.video, "route": rec.route, "summary": summary,
                "states": [{"index": r.index, "lines": list(r.lines), "text": r.first_text, "start": r.start,
                            "end": r.end, "bestTime": r.best_time, "score": r.score,
                            "offsets": [vars(o) for o in r.offsets]} for r in results]})
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1, default=lambda v: None if isinstance(v, float) and math.isnan(v) else v)
    print("P0 통과" if ok else "P0 미통과")
    return 0 if ok else 1


def cmd_synth_check(args) -> int:
    from .synthetic import shifted, write_recording
    from .testbed import Recording, run_recording, summarize, visual_states
    runtime, tuning, motions = _runtime_paths(args.runtime)
    cat = _catalog(args, runtime)
    os.makedirs(args.out, exist_ok=True)
    video = os.path.join(args.out, "synthetic.mp4")
    t0 = time.time()
    with Oracle(tuning, motions, command=[args.oracle] if args.oracle else None) as oracle:
        states = visual_states(route_lines(oracle.fold_yarn(args.yarn)))
        snaps = [s.snapshot for s in states]
        # 가운데 상태의 첫 인물 슬롯을 (+3, -2)px 옮겨 녹화한다 — 측정이 그만큼을 되읽어야 한다.
        target = len(snaps) // 2
        moved = next((s["key"] for s in snaps[target].visible_slots() if not s.get("background")), None)
        if moved:
            snaps[target] = shifted(snaps[target], moved, 3.0, -2.0)
        write_recording(video, snaps, cat, oracle.width, oracle.height)
        results = run_recording(Recording(video, [os.path.abspath(p) for p in args.yarn]), oracle, cat, tuning,
                                log=lambda m: print("  " + m))
    summary = summarize(results)
    found = summary["statesFound"] == summary["states"]
    planted = [o for r in results for o in r.offsets if moved and o.key == moved and o.distance > 1.0]
    recovered = len(planted) == 1 and abs(planted[0].dx - 3.0) < 0.25 and abs(planted[0].dy + 2.0) < 0.25
    others = [o for r in results for o in r.offsets if o not in planted]
    clean = all(o.distance <= 0.25 for o in others)
    print(f"정렬 {summary['statesFound']}/{summary['states']} · 심은 어긋남 되읽기 {'성공' if recovered else '실패'}"
          f" · 나머지 최대 {max((o.distance for o in others), default=0):.3f}px · {time.time() - t0:.0f}초")
    good = found and recovered and clean
    print("synth-check: 통과" if good else "synth-check: 실패")
    return 0 if good else 1


def _catalog_for(args, runtime: str) -> Catalog:
    """--catalog(파일) · --testkit(시험용 표정 초상) · 기본(런타임 원본) 순."""
    if getattr(args, "testkit", None):
        from .testkit import build_test_catalog
        return build_test_catalog(import_runtime(runtime), args.testkit)
    return _catalog(args, runtime)


def _size(text: str):
    w, h = text.lower().split("x")
    return int(w), int(h)


def cmd_scenario(args) -> int:
    from .dialogue import BoxLayout, gte_profile
    from .gte import rows_from_truth, write_like_gte
    from .synthetic import write_scenario
    runtime, tuning, motions = _runtime_paths(args.runtime)
    cat = _catalog_for(args, runtime)
    os.makedirs(args.out, exist_ok=True)
    layout = BoxLayout.from_tuning(tuning, preset=args.preset)
    video = os.path.join(args.out, "scenario.mp4")
    t0 = time.time()
    with Oracle(tuning, motions, command=[args.oracle] if args.oracle else None) as oracle:
        lines = route_lines(oracle.fold_yarn(args.yarn))
        from .effects import EffectPresets
        truth = write_scenario(video, lines, cat, (oracle.width, oracle.height), layout, size=_size(args.size),
                               fps=args.fps, seed=args.seed, route=args.yarn,
                               oracle=oracle if args.timed else None,
                               effect_presets=EffectPresets.load(tuning) if args.timed else None)
    with open(os.path.join(args.out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "gte-profile.json"), "w", encoding="utf-8") as f:
        json.dump(gte_profile(layout), f, ensure_ascii=False, indent=2)
    write_like_gte(os.path.join(args.out, "gte-simulated.xlsx"), rows_from_truth(truth))
    print(f"줄 {len(truth['lines'])} · {truth['duration']:.1f}초 · {time.time() - t0:.0f}초 걸림 → {args.out}")
    print("  scenario.mp4 · truth.json · gte-profile.json(게임텍스트추출기 프로필) · gte-simulated.xlsx(정답에서 흉내 낸 엑셀)")
    return 0


def cmd_extract(args) -> int:
    from .dialogue import BoxLayout
    from .pipeline import extract
    runtime, tuning, motions = _runtime_paths(args.runtime)
    if args.tuning:
        tuning = os.path.abspath(args.tuning)   # 다른 게임: catalog-build가 만든 튜닝 겹본
    cat = _catalog_for(args, runtime)
    t0 = time.time()
    with Oracle(tuning, motions, command=[args.oracle] if args.oracle else None) as oracle:
        layout = BoxLayout.from_gte_profile(args.box_profile, int(oracle.width), int(oracle.height)) \
            if args.box_profile else None
        doc = extract(args.video, oracle, cat, tuning, args.out, args.text, node=args.node, preset=args.preset,
                      timing=not args.no_timing, layout=layout, motions_path=motions,
                      lines_from_text=args.lines_from_text)
    if args.sheet:
        from .sheet import export_sheet
        yarn = doc["files"]["yarn"]
        out = os.path.splitext(yarn)[0] + ".연출.xlsx"
        export_sheet(yarn, out)
        print(f"→ {out}")
    print(f"{time.time() - t0:.0f}초")
    return 0


def _print_report(report: dict) -> None:
    print(f"줄 {report['matched']}/{report['truthLines']} 짝지음(추출 {report['gotLines']}) · 머리 최대 "
          f"{report['headFramesMax']}프레임")
    print(f"상태 일치 {report['stateMatch'] * 100:.1f}% · 화자 {report['speakerMatch'] * 100:.1f}% · "
          f"대사 완전 일치 {report['textExact'] * 100:.1f}% · 글자 정확도 {report['charAccuracy'] * 100:.1f}%")
    for row in report["lines"]:
        if row["state"] or row["text"]:
            what = "; ".join(row["state"])
            if row["text"]:
                what += f" 대사: {row['text']['got']!r}"
            print(f"  정답 줄 {row['truth']}: {what}")
    if report["missed"]:
        print(f"  못 찾은 정답 줄: {report['missed']}")
    if report["extra"]:
        print(f"  더 나온 줄: {report['extra']}")
    print("P1 통과" if report["pass"] else "P1 미통과")
    tm = report.get("timing")
    if tm:
        print(f"시간: 흐름 {tm['tweens']} · 길이 2fr 안 {tm['durationWithin'] * 100:.1f}% · 시작 2fr 안 "
              f"{tm['startWithin'] * 100:.1f}% · 이징 일치 {tm['easeMatch'] * 100:.1f}%(움직임 {tm['moveTweens']})")
        for p in tm["problems"]:
            print(f"  정답 줄 {p['truth']} {p['who']} {p['kind']} {p['command']}: 정답 {p.get('want')} · 추출 {p.get('got', p.get('problem'))}")
        print("P2 통과" if tm["pass"] else "P2 미통과")
    vc = report.get("vocab")
    if vc:
        print(f"어휘 · 효과: 정답 {vc['wanted']} · 추출 {vc['found']} · 맞음 {vc['hit']} — 재현율 {vc['recall'] * 100:.1f}% · "
              f"정밀도 {vc['precision'] * 100:.1f}%")
        for p in vc["problems"]:
            print(f"  정답 줄 {p['truth']}: 빠짐 {p['missing']} · 더 나옴 {p['extra']}")
        print("P3 통과" if vc["pass"] else "P3 미통과")


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate, load_json
    runtime, tuning, motions = _runtime_paths(args.runtime)
    truth, staging = load_json(args.truth), load_json(args.staging)
    with Oracle(tuning, motions, command=[args.oracle] if args.oracle else None) as oracle:
        report = evaluate(truth, staging, oracle, args.tolerance)
        if truth.get("timed"):
            from .evaluate import evaluate_timing, evaluate_vocab
            report["timing"] = evaluate_timing(truth, staging, oracle)
            report["vocab"] = evaluate_vocab(truth, staging)
    _print_report(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 0 if all(report.get(k, {"pass": True})["pass"] for k in ("timing", "vocab")) and report["pass"] else 1


def cmd_p1_check(args) -> int:
    from .dialogue import BoxLayout
    from .evaluate import evaluate
    from .gte import rows_from_truth, write_like_gte
    from .pipeline import extract
    from .synthetic import write_scenario
    from .testkit import build_test_catalog
    runtime, tuning, motions = _runtime_paths(args.runtime)
    os.makedirs(args.out, exist_ok=True)
    cat = build_test_catalog(import_runtime(runtime), os.path.join(args.out, "testkit"))
    yarn = args.yarn or [os.path.join(_REPO, "testbed", "scenarios", "p1_basic.yarn")]
    layout = BoxLayout.from_tuning(tuning)
    video = os.path.join(args.out, "scenario.mp4")
    t0 = time.time()
    with Oracle(tuning, motions, command=[args.oracle] if args.oracle else None) as oracle:
        lines = route_lines(oracle.fold_yarn(yarn))
        truth = write_scenario(video, lines, cat, (oracle.width, oracle.height), layout, size=_size(args.size),
                               route=yarn)
        xlsx = write_like_gte(os.path.join(args.out, "gte-simulated.xlsx"), rows_from_truth(truth))
        print(f"시험 녹화: 줄 {len(lines)} · {truth['duration']:.1f}초")
        staging = extract(video, oracle, cat, tuning, args.out, xlsx, node="P1_Check", log=lambda m: None,
                          timing=False)
        report = evaluate(truth, staging, oracle)
    with open(os.path.join(args.out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    _print_report(report)
    print(f"{time.time() - t0:.0f}초 · 결과 {args.out}")
    return 0 if report["pass"] else 1


def cmd_p2_check(args) -> int:
    from .dialogue import BoxLayout
    from .evaluate import evaluate, evaluate_timing
    from .gte import rows_from_truth, write_like_gte
    from .pipeline import extract
    from .synthetic import write_scenario
    from .testkit import build_test_catalog
    runtime, tuning, motions = _runtime_paths(args.runtime)
    os.makedirs(args.out, exist_ok=True)
    cat = build_test_catalog(import_runtime(runtime), os.path.join(args.out, "testkit"))
    yarn = args.yarn or [os.path.join(_REPO, "testbed", "scenarios", "p2_timing.yarn")]
    layout = BoxLayout.from_tuning(tuning)
    video = os.path.join(args.out, "scenario.mp4")
    t0 = time.time()
    with Oracle(tuning, motions, command=[args.oracle] if args.oracle else None) as oracle:
        lines = route_lines(oracle.fold_yarn(yarn))
        truth = write_scenario(video, lines, cat, (oracle.width, oracle.height), layout, size=_size(args.size),
                               route=yarn, oracle=oracle)
        xlsx = write_like_gte(os.path.join(args.out, "gte-simulated.xlsx"), rows_from_truth(truth))
        print(f"시험 녹화(시간 축): 줄 {len(lines)} · {truth['duration']:.1f}초")
        staging = extract(video, oracle, cat, tuning, args.out, xlsx, node="P2_Check", log=lambda m: None)
        report = evaluate(truth, staging, oracle)
        report["timing"] = evaluate_timing(truth, staging, oracle)
    with open(os.path.join(args.out, "truth.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    _print_report(report)
    print(f"{time.time() - t0:.0f}초 · 결과 {args.out}")
    return 0 if report["pass"] and report["timing"]["pass"] else 1


def cmd_p3_check(args) -> int:
    """P3 점검 — 어휘(카메라 · 모션 · 슬라이드)와 화면 효과(근사) 시험 녹화를 각각 만들어 추출하고 상태 · 시간 · 어휘를 평가."""
    from .dialogue import BoxLayout
    from .effects import EffectPresets
    from .evaluate import evaluate, evaluate_timing, evaluate_vocab
    from .gte import rows_from_truth, write_like_gte
    from .pipeline import extract
    from .synthetic import write_scenario
    from .testkit import build_test_catalog
    runtime, tuning, motions = _runtime_paths(args.runtime)
    os.makedirs(args.out, exist_ok=True)
    cat = build_test_catalog(import_runtime(runtime), os.path.join(args.out, "testkit"))
    yarns = args.yarn or [os.path.join(_REPO, "testbed", "scenarios", n) for n in ("p3_vocab.yarn", "p3_effects.yarn")]
    layout = BoxLayout.from_tuning(tuning)
    presets = EffectPresets.load(tuning)
    t0 = time.time()
    good = True
    with Oracle(tuning, motions, command=[args.oracle] if args.oracle else None) as oracle:
        for yarn in yarns:
            name = os.path.splitext(os.path.basename(yarn))[0]
            out = os.path.join(args.out, name)
            os.makedirs(out, exist_ok=True)
            video = os.path.join(out, "scenario.mp4")
            lines = route_lines(oracle.fold_yarn([yarn]))
            truth = write_scenario(video, lines, cat, (oracle.width, oracle.height), layout, size=_size(args.size),
                                   route=[yarn], oracle=oracle, effect_presets=presets)
            xlsx = write_like_gte(os.path.join(out, "gte-simulated.xlsx"), rows_from_truth(truth))
            print(f"── {name}: 시험 녹화 줄 {len(lines)} · {truth['duration']:.1f}초")
            staging = extract(video, oracle, cat, tuning, out, xlsx, node=f"P3_{name}", log=lambda m: None)
            report = evaluate(truth, staging, oracle)
            report["timing"] = evaluate_timing(truth, staging, oracle)
            report["vocab"] = evaluate_vocab(truth, staging)
            with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as f:
                json.dump(truth, f, ensure_ascii=False, indent=1)
            with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=1)
            _print_report(report)
            good = good and report["pass"] and report["timing"]["pass"] and report["vocab"]["pass"]
    print(("P3 점검 통과" if good else "P3 점검 미통과") + f" · {time.time() - t0:.0f}초 · 결과 {args.out}")
    return 0 if good else 1


def cmd_scan(args) -> int:
    from .dialogue import BoxLayout
    from .scan import scan
    import cv2
    cap = cv2.VideoCapture(args.video)
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not vw:
        raise SystemExit(f"영상을 열지 못했습니다: {args.video}")
    layout = BoxLayout.from_gte_profile(args.box_profile, vw, vh)
    t0 = time.time()
    scan(args.video, layout, args.out, args.box_profile, pieces=not args.plates_only)
    print(f"{time.time() - t0:.0f}초 — 카탈로그 툴(컴포지터 src/Vn.CatalogTool) 또는 `draft`로 고친 뒤 catalog-build")
    return 0


def cmd_draft(args) -> int:
    from .catalogdraft import DraftEditor, DraftError
    ed = DraftEditor.load(args.folder)
    a = args.args
    try:
        if args.op is None:
            pass
        elif args.op == "rename":
            ed.rename_character(a[0], a[1])
        elif args.op == "merge":
            ed.merge_characters(a[0], a[1], args.variant)
        elif args.op == "move":
            if not args.to:
                raise DraftError("--to 인물[/변형[/표정]]을 주세요")
            parts = args.to.split("/")
            e = ed.move(a, parts[0], parts[1] if len(parts) > 1 else "a", parts[2] if len(parts) > 2 else None)
            print(f"표정 {e}")
        elif args.op in ("exclude", "include"):
            ed.exclude(a, args.op == "exclude")
        elif args.op == "rep":
            ed.set_representative(a[0])
        elif args.op == "merge-emotion":
            ed.merge_emotions(a[0], a[1], a[2], a[3])
        elif args.op == "bg-rename":
            ed.rename_background(a[0], a[1])
        elif args.op in ("bg-exclude", "bg-include"):
            ed.exclude_background(a[0], args.op == "bg-exclude")
        else:
            raise DraftError(f"모르는 고치기: {args.op}")
    except (DraftError, IndexError) as e:
        print(f"고치지 못했습니다: {e}", file=sys.stderr)
        return 2
    if args.op is not None:
        ed.save()
    for c, v, e, members in ed.groups():
        rep = ed.representative(c, v, e)
        ids = " ".join(("★" if rep and o["id"] == rep["id"] else "") + o["id"] for o in members)
        print(f"{c}/{v}/{e}  ({len(members)})  {ids}")
    out = [o["id"] for o in ed.observations if o.get("excluded")]
    if out:
        print(f"뺀 조각: {' '.join(out)}")
    for b in ed.backgrounds:
        print(f"배경 {b['key']}{' (뺌)' if b.get('excluded') else ''}: 줄 {len(b.get('lines', []))}개")
    return 0


def cmd_catalog_originals(args) -> int:
    from .originals import build_from_originals
    _, tuning, _ = _runtime_paths(args.runtime)
    t0 = time.time()
    path, overlay = build_from_originals(args.originals, args.draft, args.out, tuning,
                                         [args.oracle] if args.oracle else None)
    print(f"{time.time() - t0:.0f}초 — 추출: extract <영상> --catalog {path} --tuning {overlay} --box-profile <프로필>")
    return 0


def cmd_catalog_build(args) -> int:
    from .catalogdraft import build_catalog
    _, tuning, _ = _runtime_paths(args.runtime)
    path, overlay = build_catalog(args.folder, args.out, tuning, [args.oracle] if args.oracle else None)
    print(f"추출: extract <영상> --catalog {path} --tuning {overlay} --box-profile <프로필>")
    return 0


HOP = {"name": "hop", "rig": "character", "duration": 0.5, "end": "restore", "channels": [
    {"node": "CharacterPortrait_Track_Move_Y", "kind": "position", "y": [
        {"t": 0, "v": 0, "inTangent": 0, "outTangent": 0}, {"t": 0.25, "v": 30, "inTangent": 0, "outTangent": 0},
        {"t": 0.5, "v": 0, "inTangent": 0, "outTangent": 0}]}]}
HOP_PEAK = 30.0


def cmd_p4_check(args) -> int:
    """P4 점검 — 시험용 초상으로 '모르는 게임' 녹화를 만들고(런타임에 없는 모션 hop 포함), 이름을 숨긴 채
    scan → catalog-build → extract로 배워 뽑은 뒤, 보이는 모양끼리 이름을 다수결로 대응시켜 견준다."""
    from .catalogdraft import build_catalog
    from .evaluate import evaluate_unknown
    from .dialogue import BoxLayout, gte_profile
    from .gte import rows_from_truth, write_like_gte
    from .pipeline import extract
    from .scan import scan
    from .synthetic import write_scenario
    from .testkit import build_test_catalog
    runtime, tuning, motions = _runtime_paths(args.runtime)
    os.makedirs(args.out, exist_ok=True)
    cat = build_test_catalog(import_runtime(runtime), os.path.join(args.out, "testkit"))
    yarn = args.yarn or [os.path.join(_REPO, "testbed", "scenarios", "p4_unknown.yarn")]
    with open(motions, encoding="utf-8-sig") as f:
        table = json.load(f)
    table["motions"] = [m for m in table.get("motions", []) if m.get("name") != "hop"] + [HOP]
    hop_motions = os.path.join(args.out, "motions-hop.json")
    with open(hop_motions, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)
    layout = BoxLayout.from_tuning(tuning)
    video = os.path.join(args.out, "scenario.mp4")
    cmd = [args.oracle] if args.oracle else None
    quiet = (lambda m: None) if not args.verbose else print
    t0 = time.time()
    with Oracle(tuning, hop_motions, command=cmd) as oracle:
        lines = route_lines(oracle.fold_yarn(yarn))
        truth = write_scenario(video, lines, cat, (oracle.width, oracle.height), layout, size=_size(args.size),
                               route=yarn, oracle=oracle)
        truth_snaps = [l["snapshot"] for l in lines]       # 쪼개짐 세기용
    xlsx = write_like_gte(os.path.join(args.out, "gte-simulated.xlsx"), rows_from_truth(truth))
    profile = os.path.join(args.out, "game.gte-profile.json")
    with open(profile, "w", encoding="utf-8") as f:
        json.dump(gte_profile(layout, "모르는 게임(시험)"), f, ensure_ascii=False, indent=2)
    print(f"시험 녹화: 줄 {len(lines)} · {truth['duration']:.1f}초")
    vw, vh = _size(args.size)
    scan(video, BoxLayout.from_gte_profile(profile, vw, vh), os.path.join(args.out, "draft"), profile, log=print,
         pieces=not args.originals)
    if args.originals:
        # 원본 모드: 시험용 초상 · 런타임 배경을 '그 게임의 원본'처럼 <이름>_<변형>_<표정>.png · BG_<키>.png로 둔다.
        # 이름은 런타임 견본과 겹치지 않게 바꾼다(Game*) — 평가는 정답 대본을 이 겹본으로 접으므로, 같은 이름이면
        # 정답 인물의 치수가 원본 그림의 치수로 덮여 정답 모양이 틀어진다. 이름 대응은 다수결로 한다.
        from .imageio import read_image, write_image
        from .originals import build_from_originals
        odir = os.path.join(args.out, "originals")
        os.makedirs(odir, exist_ok=True)
        for p in cat.portraits:
            write_image(os.path.join(odir, f"Game{p.character.capitalize()}_{p.variant}_{p.emotion}.png"),
                        read_image(cat.abspath(p.path), alpha=True))
        for b in cat.backgrounds:
            write_image(os.path.join(odir, f"BG_game_{b.key}.png"), read_image(cat.abspath(b.path)))
        cat_path, overlay = build_from_originals(odir, os.path.join(args.out, "draft"),
                                                 os.path.join(args.out, "catalog"), tuning, cmd, log=print)
    else:
        cat_path, overlay = build_catalog(os.path.join(args.out, "draft"), os.path.join(args.out, "catalog"), tuning,
                                          cmd, log=print)
    got_cat = Catalog.load(cat_path)
    with Oracle(overlay, motions, command=cmd) as oracle:
        box = BoxLayout.from_gte_profile(profile, int(oracle.width), int(oracle.height))
        staging = extract(video, oracle, got_cat, overlay, os.path.join(args.out, "extract"), xlsx, node="P4_Check",
                          layout=box, motions_path=motions, log=quiet)
        # 정답 대본의 hop은 이 모션 표에 없다 — 제자리 모션이라 정착 화면은 같다.
        report = evaluate_unknown(truth, staging, oracle, cat, got_cat, layout.stage_bottom())
    # 모션 초안 — hop이 있는 정답 줄의 짝 줄에서 나온 초안의 y 봉우리
    hop_lines = [i for i, l in enumerate(truth["lines"]) if any(c["name"] == "motion" and c["args"][1:2] == ["hop"]
                                                               for c in l["commands"])]
    pair = {r["truth"]: r["got"] for r in report["lines"]}
    drafts = {d["line"]: d for d in staging.get("motionDrafts", [])}
    peaks = [drafts[pair[i]]["peak"][1] for i in hop_lines if i in pair and pair[i] in drafts]
    report["motionPeak"] = peaks[0] if peaks else None
    report["motionDrafts"] = len(drafts)
    motion_ok = not hop_lines or (bool(peaks) and abs(peaks[0] - HOP_PEAK) <= 3.0)
    report["pass"] = bool(report["pass"] and motion_ok)
    # 쪼개짐 — 순도는 한 사람을 둘로 나눠도 깎이지 않는다. 사람이 툴에서 합칠 일이 얼마나 남았나를 따로 센다.
    true_c = {(s.get("character")) for sn in truth_snaps for s in sn.visible_slots(0.5)
              if s.get("character") and not s.get("background")}
    true_e = {(s.get("character"), s.get("variant"), s.get("emotion")) for sn in truth_snaps
              for s in sn.visible_slots(0.5) if s.get("character") and not s.get("background")}
    report["learned"] = {"characters": len(report["characters"]), "expressions": len(report["expressions"])}
    report["truth"] = {"characters": len(true_c), "expressions": len(true_e)}
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"이름 대응: 인물 {report['characters']} · 배경 {report['backgrounds']}")
    print(f"배운 인물 {report['learned']['characters']}명(정답 {report['truth']['characters']}) · 표정 무리 "
          f"{report['learned']['expressions']}개(정답 {report['truth']['expressions']})")
    print(f"인물 순도 {report['identityPurity'] * 100:.1f}% · 표정 순도 {report['expressionPurity'] * 100:.1f}% · "
          f"줄 일치 {report['lineMatch'] * 100:.1f}% · 모양 겹침 평균 {report['meanIoU']:.3f}")
    for row in report["lines"]:
        if row["state"]:
            print(f"  정답 줄 {row['truth']}: {'; '.join(row['state'])}")
    if hop_lines:
        print(f"모션 초안 {len(drafts)}개 · hop 봉우리 {report['motionPeak']}(정답 {HOP_PEAK:g})")
    print(("P4 통과" if report["pass"] else "P4 미통과") + f" · {time.time() - t0:.0f}초 · 결과 {args.out}")
    return 0 if report["pass"] else 1


def _known_commands(args) -> Optional[set]:
    """오라클의 커맨드 표(이름) — 없으면 None(검사를 건너뛴다)."""
    try:
        _, tuning, motions = _runtime_paths(args.runtime)
        with Oracle(tuning, motions, command=[args.oracle] if args.oracle else None) as oracle:
            return set(oracle.commands())
    except (OracleError, SystemExit, OSError) as e:
        print(f"커맨드 이름 검사를 건너뜁니다({e})", file=sys.stderr)
        return None


def cmd_sheet(args) -> int:
    """P5 — 엑셀 연출 열 ↔ .yarn(.yarn이 주인, 엑셀은 고치는 보기)."""
    import tempfile
    from .sheet import SheetError, apply_sheet, export_sheet
    try:
        if args.action == "export":
            for y in args.paths:
                out = args.output if args.output and len(args.paths) == 1 else os.path.splitext(y)[0] + ".연출.xlsx"
                n = export_sheet(y, out)
                print(f"{y} → {out} (행 {n})")
            return 0
        if args.action == "apply":
            known = None if args.no_check else _known_commands(args)
            text, changed = apply_sheet(args.paths[0], args.yarn, force=args.force, known_commands=known)
            if not changed:
                print("바뀐 칸이 없습니다 — .yarn을 그대로 둡니다")
                return 0
            target = args.output
            if target is None:
                if not args.in_place:
                    print("고친 .yarn을 어디에 쓸지: -o <경로> 또는 --in-place(원래 .yarn을 고치고 .bak을 남긴다)", file=sys.stderr)
                    return 2
                from .sheet import read_sheet
                target = args.yarn or read_sheet(args.paths[0])[0]["yarn"]
                with open(target, encoding="utf-8", newline="") as f:
                    old = f.read()
                with open(target + ".bak", "w", encoding="utf-8", newline="") as f:
                    f.write(old)
            with open(target, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            print(f"바뀐 행 {len(changed)}: {', '.join(changed[:12])}{' …' if len(changed) > 12 else ''} → {target}")
            return 0
        # check: 고치지 않은 왕복이 바이트 그대로인가
        bad = 0
        with tempfile.TemporaryDirectory() as tmp:
            for y in args.paths:
                x = os.path.join(tmp, "t.xlsx")
                export_sheet(y, x)
                with open(y, encoding="utf-8", newline="") as f:
                    original = f.read()
                text, changed = apply_sheet(x, y)
                ok = text == original and not changed
                bad += not ok
                print(f"{'왕복 그대로' if ok else '왕복이 다르다'}: {y}")
        return 1 if bad else 0
    except SheetError as e:
        print(str(e), file=sys.stderr)
        return 2


def cmd_gte_profile(args) -> int:
    from .dialogue import BoxLayout, gte_profile
    _, tuning, _ = _runtime_paths(args.runtime)
    layout = BoxLayout.from_tuning(tuning, preset=args.preset)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(gte_profile(layout), f, ensure_ascii=False, indent=2)
    print(f"게임텍스트추출기 프로필 → {args.output} (프로필 폴더에 넣고 선택하세요)")
    return 0


def cmd_originals(args) -> int:
    from . import collect

    config = collect.load_json(args.config)
    data_dir = os.path.abspath(args.data)
    path = os.path.join(data_dir, "manifest.json")
    api = collect.Api(config["api"], config.get("userAgent", "ked-originals/0.1"),
                      delay=float(config.get("delay", 1.0)))
    started = time.time()
    try:
        manifest = collect.build_manifest(config, api)
    except collect.ApiError as e:
        print(f"API 오류: {e}", file=sys.stderr)
        return 2
    s = manifest["summary"]
    print(f"\n파일 {s['files']}개 · {collect.human_bytes(s['bytes'])} (요청 {api.requests}번 · {time.time() - started:.0f}초)")
    for kind, k in sorted(s["kinds"].items()):
        print(f"  {kind:<14} {k['files']:>6}개  {collect.human_bytes(k['bytes'])}")
    for label, key in (("규칙 밖 이름", "unparsed"), ("카테고리와 정체가 다름", "mismatched"),
                       ("다른 카테고리에도 있음", "duplicates"), ("문제", "problems")):
        items = manifest[key]
        print(f"{label}: {len(items)}개")
        for it in items[:args.show]:
            print(f"  {it}")
        if len(items) > args.show:
            print(f"  … 나머지 {len(items) - args.show}개는 manifest.json에")
    if os.path.isfile(path):
        d = collect.diff(collect.load_json(path), manifest)
        print(f"앞 manifest와 견주어: 새 파일 {len(d['added'])} · 바뀐 파일 {len(d['changed'])} · 사라진 파일 {len(d['removed'])}")
        for key in ("changed", "removed"):
            for t in d[key][:args.show]:
                print(f"  {key}: {t}")
    collect.write_json(path, manifest)
    print(f"→ {path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ked_staging", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oracle", help="ked-stage-oracle 경로(기본: 환경 변수 KED_STAGE_ORACLE · 옆 컴포지터 빌드)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("catalog", help="런타임 카탈로그 만들기")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("-o", "--output", default="catalog.json")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("predict", help="yarn 경로의 예측 화면")
    p.add_argument("yarn", nargs="+")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("--catalog")
    p.add_argument("--out", default="predicted")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--separate", action="store_true", help="노드마다 처음 상태에서 접는다(기본은 이어 접기)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("testbed", help="시험장 매니페스트 실행")
    p.add_argument("manifest")
    p.add_argument("--catalog")
    p.add_argument("--out", default="testbed-out")
    p.add_argument("--tolerance", type=float, default=1.0)
    p.set_defaults(func=cmd_testbed)

    p = sub.add_parser("synth-check", help="합성 녹화로 설치 점검")
    p.add_argument("yarn", nargs="+")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("--catalog")
    p.add_argument("--out", default="synth-check-out")
    p.set_defaults(func=cmd_synth_check)

    p = sub.add_parser("scenario", help="P1 시험 녹화와 정답 만들기")
    p.add_argument("yarn", nargs="+")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("--catalog")
    p.add_argument("--testkit", metavar="DIR", help="표정이 구별되는 시험용 초상을 DIR에 만들어 쓴다")
    p.add_argument("--out", default="scenario-out")
    p.add_argument("--size", default="1920x1080")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--preset", default="bottom", help="대사창 surface-layout 프리셋")
    p.add_argument("--timed", action="store_true", help="무대를 오라클 시간 축대로 움직인다(P2 — 정답에 시간표가 붙는다)")
    p.set_defaults(func=cmd_scenario)

    p = sub.add_parser("extract", help="영상에서 연출 뽑기")
    p.add_argument("video")
    p.add_argument("--text", help="게임텍스트추출기 결과 엑셀(.xlsx)")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("--catalog")
    p.add_argument("--testkit", metavar="DIR", help="시험용 초상(scenario와 같은 DIR)")
    p.add_argument("--out", default="extract-out")
    p.add_argument("--node", default="Extracted")
    p.add_argument("--preset", default="bottom")
    p.add_argument("--no-timing", action="store_true", help="시간 · 이징을 재지 않는다(P1 결과만)")
    p.add_argument("--tuning", help="튜닝 폴더(다른 게임: catalog-build가 만든 겹본)")
    p.add_argument("--box-profile", help="대사창 자리를 게임텍스트추출기 프로필에서(다른 게임)")
    p.add_argument("--sheet", action="store_true", help="추출 .yarn의 연출 엑셀(<영상>.연출.xlsx)도 낸다(P5)")
    p.add_argument("--lines-from-text", action="store_true",
                   help="줄을 --text 엑셀의 대사 줄로 세운다(대사창이 줄 사이에 글자를 지웠다 쓰는 게임 — 헛줄 · 붙은 줄을 막는다)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("evaluate", help="추출 결과를 정답과 비교")
    p.add_argument("truth")
    p.add_argument("staging")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("--tolerance", type=float, default=6.0, help="자리 허용(px)")
    p.add_argument("--report")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("p1-check", help="P1 점검(시험 녹화 → 추출 → 평가)")
    p.add_argument("yarn", nargs="*")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("--out", default="p1-check-out")
    p.add_argument("--size", default="1920x1080")
    p.set_defaults(func=cmd_p1_check)

    p = sub.add_parser("p2-check", help="P2 점검(시간 축 시험 녹화 → 추출 → 상태 · 시간 평가)")
    p.add_argument("yarn", nargs="*")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("--out", default="p2-check-out")
    p.add_argument("--size", default="1920x1080")
    p.set_defaults(func=cmd_p2_check)

    p = sub.add_parser("p3-check", help="P3 점검(어휘 · 화면 효과 시험 녹화 → 추출 → 상태 · 시간 · 어휘 평가)")
    p.add_argument("yarn", nargs="*")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("--out", default="p3-check-out")
    p.add_argument("--size", default="1280x720")
    p.set_defaults(func=cmd_p3_check)

    p = sub.add_parser("scan", help="P4: 다른 게임 영상 → 카탈로그 초안")
    p.add_argument("video")
    p.add_argument("--box-profile", required=True, help="그 게임의 게임텍스트추출기 프로필(대사창 자리)")
    p.add_argument("--out", default="draft")
    p.add_argument("--plates-only", action="store_true",
                   help="줄 · 배경 판까지만(인물을 배우지 않는다 — 원본 스프라이트로 catalog-originals를 쓸 때)")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("draft", help="P4: 초안 보기 · 고치기")
    p.add_argument("folder", help="초안 폴더(draft.json)")
    p.add_argument("op", nargs="?", choices=["rename", "merge", "move", "exclude", "include", "rep", "merge-emotion",
                                             "bg-rename", "bg-exclude", "bg-include"])
    p.add_argument("args", nargs="*")
    p.add_argument("--to", help="move: 인물[/변형[/표정]] (표정을 비우면 새 번호)")
    p.add_argument("--variant", help="merge: 옷이 다른 같은 인물로 — 이 변형으로 합친다")
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser("catalog-build", help="P4: 초안 → 카탈로그 + 튜닝 겹본")
    p.add_argument("folder", help="초안 폴더(draft.json)")
    p.add_argument("--out", default="catalog")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.set_defaults(func=cmd_catalog_build)

    p = sub.add_parser("catalog-originals", help="P4: 원본 스프라이트 · 배경으로 카탈로그 + 튜닝 겹본")
    p.add_argument("originals", help="원본 폴더(<이름>_<변형>_<표정>.png · BG_<이름>.png)")
    p.add_argument("--draft", required=True, help="scan 초안 폴더(줄 정착 화면 · 배경 판을 쓴다)")
    p.add_argument("--out", default="catalog")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.set_defaults(func=cmd_catalog_originals)

    p = sub.add_parser("p4-check", help="P4 점검('모르는 게임' 녹화 → scan → catalog-build → extract → 평가)")
    p.add_argument("yarn", nargs="*")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("--out", default="p4-check-out")
    p.add_argument("--size", default="1280x720")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--originals", action="store_true",
                   help="원본 모드 점검 — 시험용 초상 · 배경을 원본으로 주고 catalog-originals로 카탈로그를 만든다")
    p.set_defaults(func=cmd_p4_check)

    p = sub.add_parser("sheet", help="P5: 엑셀 연출 열 ↔ .yarn(export · apply · check)")
    p.add_argument("action", choices=["export", "apply", "check"])
    p.add_argument("paths", nargs="+", help="export · check: .yarn들 · apply: 연출 엑셀")
    p.add_argument("-o", "--output", help="export: 엑셀 경로 · apply: 새 .yarn 경로")
    p.add_argument("--yarn", help="apply: 원래 .yarn(기본: 엑셀 '정보' 시트의 경로)")
    p.add_argument("--in-place", action="store_true", help="apply: 원래 .yarn을 고친다(.bak을 남긴다)")
    p.add_argument("--force", action="store_true", help="apply: 내보낸 뒤 .yarn이 바뀌었어도 들여온다")
    p.add_argument("--no-check", action="store_true", help="apply: 커맨드 이름을 오라클 표로 검사하지 않는다")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.set_defaults(func=cmd_sheet)

    p = sub.add_parser("originals", help="원본 수집: MediaWiki 카테고리 → 원본 그림 목록(manifest)")
    p.add_argument("action", choices=["manifest"])
    p.add_argument("config", help="수집 설정 JSON(위키 주소 · 카테고리 — private 저장소에 둔다)")
    p.add_argument("--data", required=True, help="데이터 폴더(git 밖) — manifest.json과 원본 그림이 여기에")
    p.add_argument("--show", type=int, default=10, help="목록마다 화면에 보일 개수")
    p.set_defaults(func=cmd_originals)

    p = sub.add_parser("gte-profile", help="게임텍스트추출기 프로필 내기")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME)
    p.add_argument("--preset", default="bottom")
    p.add_argument("-o", "--output", default="ked-runtime.gte-profile.json")
    p.set_defaults(func=cmd_gte_profile)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except OracleError as e:
        print(f"오라클 오류: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
