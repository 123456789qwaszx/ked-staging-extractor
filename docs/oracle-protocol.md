# ked-stage-oracle 프로토콜 (2)

오라클은 ked-story-compositor 저장소의 `src/Vn.StageOracle` 콘솔 프로그램이다. 무대 계산은 런타임 코어
(`Ked.Presentation.Core`, 컴포지터가 반입한 사본)가, 화면 투영은 컴포지터 프리뷰와 같은
`StageSceneComposer.TryProjectSlotImage`가 한다. 이 저장소는 그 결과만 읽는다(규칙 5).

```
ked-stage-oracle serve    --tuning <ExportedTuning> [--motions <motions.json>]
ked-stage-oracle groups   <file.yarn>
ked-stage-oracle fold     --tuning <dir> [--motions <file>] [--continuous] <file.yarn>...
ked-stage-oracle selftest --tuning <dir> [--motions <file>] <file.yarn>...
```

`serve`는 표준 입력으로 JSON 요청을 한 줄에 하나씩 받고, 같은 순서로 한 줄씩 답한다. 응답에는 요청의
`id`가 그대로 붙고, 실패하면 `{"ok": false, "error": "…"}`다.

| op | 요청 | 응답 |
|---|---|---|
| `ping` | — | `protocol` · `width` · `height`(기준 해상도) · `problems`(튜닝 경고) |
| `groups` | `path` 또는 `text`(+`source`) | `nodes[{name, groups[{text, lineId, commands[{name, args}]}]}]` · `warnings` |
| `foldYarn` | `paths[]`(재생 순서) 또는 `path`/`text`, `continuous`(기본 false), `node`(선택) | `width` · `height` · `nodes[{name, source, lines[{index, text, lineId, commands, snapshot}]}]` |
| `fold` | `lines[{commands[{name, args[]}]}]` | `snapshots[]` — 줄마다 하나, 처음 상태에서 접는다 |
| `timeline` | `lines[…]`(fold와 같다), `line`(기본 마지막), `times[]`(그 줄 머리부터 초) | `schedule[]` · `end` · `snapshots[]`(times마다, `time` 포함) |
| `commands` | — | `commands[{id, name, parameters[{name, type, required, default}]}]` — 컴포지터 기본 카탈로그 |

**2 (P2)**: `timeline` · `commands`가 생겼다. 1과 응답 모양은 같고 op만 늘었지만, 추출기가 버전을 확인하므로
오라클을 다시 빌드해야 한다.

`continuous`면 노드 경계에서 무대를 비우지 않는다 — 녹화한 재생 경로처럼 앞 노드의 `slot` · `cast`를
이어 쓰는 대본을 따라갈 때 쓴다.

## snapshot

```json
{
  "slots": [ /* 그리기 순서(먼 것 먼저) — StageSceneComposer.OrderForDrawing */
    {
      "key": "s1", "background": false,
      "character": "harin", "variant": "a", "emotion": "01",
      "stage": "stage00", "layer": "mid", "layerDepth": 0,
      "depthScale": 1.708, "alpha": 1,
      "spriteAspect": 0.804,
      "rect":   [x, y, w, h],               // RectTransform 경계(스트레치라 그림보다 넓다)
      "quad":   [[x,y],[x,y],[x,y],[x,y]],  // 실제로 그려지는 그림(preserveAspect): 좌하 · 우하 · 좌상 · 우상(로컬 순서)
      "bounds": [x, y, w, h],               // quad의 축 정렬 경계
      "toolMirror": false,                  // 코어 밖 mirror 축(컴포지터 규칙으로 따라간다) — 그릴 때 그림만 뒤집는다
      "mirrored": false                     // 보이는 결과(quad의 뒤집힘 XOR toolMirror)
    },
    { "key": "bg_main", "background": true, "sprite": "class_lunch", "...": "…" }
  ],
  "shot": {"zoom": 0, "pan": [0, 0], "focus": [0, 0], "cameraScale": 1, "cameraPan": [0, 0]},
  "unhandled": [{"name": "bgm", "args": [], "source": "…", "reason": "…"}]   // 직전 스냅샷 이후 새로 생긴 것만
}
```

좌표는 캔버스(기준 해상도, 좌상 원점 · y 아래)다. 스냅샷은 정착 상태(트윈이 모두 끝난 뒤)다.

## timeline — 줄 하나의 시간 축

앞 줄들은 정착까지 접고, `line`번째 줄은 `times`마다 그 순간의 스냅샷을 뜬다. 런타임 규칙 그대로다.

- 줄의 커맨드는 줄 머리에서 함께 시작한다. 대기(`pause` · `<<Nfr>>`)만 뒤 커맨드를 그만큼 늦춘다(전경은 대기뿐).
- 노드 · 카메라 트윈(보간 · 이징 · 슬라이드 튐 · 같은 노드는 뒤의 것이 이김)은 컴포지터 `StageMotionPlan`이 낸다
  (`ApplyAt`이 커맨드마다 다른 시작을 받는다).
- 페이드 투명도는 제 시간 · OutCubic(런타임 `FadeCommandCharR` 생략값)으로 흐른다.
- 즉시 커맨드(face · pose · mirror · bg_sprite · 0fr 배치)는 제 시작 시각에 바뀐다.
- 시간 · 이징 인자는 컴포지터 기본 카탈로그(`docs/game.definition.json`, 런타임 등록 표와 교차 검증)의 파라미터
  형(`duration` · `ease`)으로 찾는다 — 파서 표를 베끼지 않는다.

```json
"schedule": [
  {"index": 0, "name": "place_left", "args": ["s1","bust","20fr","InOutSine"], "start": 0, "duration": 0.8333,
   "ease": "InOutSine", "wait": false, "tween": true, "fade": false},
  {"index": 4, "name": "pause", "args": ["12fr"], "start": 0, "duration": 0.5, "ease": null, "wait": true, ...},
  {"index": 5, "name": "fade_in", "args": ["s3","20fr"], "start": 0.5, "duration": 0.8333, "ease": "OutCubic", "fade": true, ...}
]
```

## 코어가 접지 않는 것

폴드 어휘 밖의 커맨드는 `unhandled`로 온다 — 오디오(`bgm` · `sfx` · `voice`), 화면 효과(`screen_*`),
블러(`screen_blur` · `focus_on`), 캐릭터 비주얼(`char_visual_*`), 전환 마스크(`tx`), `sibling_*`,
Yarn 제어(`if` · `detour` …). 예측 화면은 이것들을 반영하지 않는다. `mirror`만은 화면에 크게 드러나서
컴포지터 프리뷰와 같은 규칙(`MiniStageFold.ResolveMirror`)으로 따라간다.
