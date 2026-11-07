"""명령줄 — `python -m ked_staging <명령>`.

  catalog     런타임 저장소에서 카탈로그(catalog.json)를 만든다
  predict     yarn 경로를 오라클로 접어 줄마다 예측 화면(PNG)과 스냅샷(JSON)을 낸다
  testbed     시험장 매니페스트의 녹화마다 정렬 · 어긋남을 재서 P0 통과 여부를 낸다
  synth-check 합성 녹화를 만들어 testbed가 제대로 도는지 확인한다(설치 점검)

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

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except OracleError as e:
        print(f"오라클 오류: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
