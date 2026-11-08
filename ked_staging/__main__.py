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

런타임 저장소는 --runtime(기본: 이 저장소 옆의 ked-presentation-runtime)이다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

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
        extract(args.video, oracle, cat, tuning, args.out, args.text, node=args.node, preset=args.preset,
                timing=not args.no_timing, layout=layout)
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


def cmd_gte_profile(args) -> int:
    from .dialogue import BoxLayout, gte_profile
    _, tuning, _ = _runtime_paths(args.runtime)
    layout = BoxLayout.from_tuning(tuning, preset=args.preset)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(gte_profile(layout), f, ensure_ascii=False, indent=2)
    print(f"게임텍스트추출기 프로필 → {args.output} (프로필 폴더에 넣고 선택하세요)")
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
