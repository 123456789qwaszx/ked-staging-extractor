# ked-staging-extractor

플레이 영상에서 대사 줄마다 연출 커맨드를 역추정해, ked-presentation-runtime이 재생하는 `.yarn` 초안을
내는 도구입니다. 런타임 코어로 "이 커맨드면 화면이 이렇게 된다"를 예측하고, 영상과 가장 잘 맞는 커맨드를 고릅니다.

설계 문서: [VN 연출 추출기 설계](https://claude.ai/artifact/U72yVHw7kuXzoWKCvRLesm) ·
커맨드 시스템 정리: [ked-presentation-runtime 커맨드 시스템 정리](https://claude.ai/artifact/B2FqDjkZfBnxvPCnS9MXgn)

## 목표

- 이 런타임으로 만든 영상에서 원래 대본에 가까운 연출을 되살린다. 때때로 다른 게임 영상의 연출을 이 런타임의 어휘로 옮긴다.
- 구조(누가 · 어디에 · 어떤 표정 · 어떤 배경)에 더해 시간(24fps 토큰)과 이징(DOTween 35종)까지 따라잡는다.
- 출력은 초안이다. 마지막 다듬기는 사람이 한다.

## 규칙

1. **원본이 있으면 원본으로.** 런타임 모드의 카탈로그는 리소스를 그대로 가리킨다. 군집화는 원본이 없을 때만 한다.
2. **정본 어휘만 낸다.** 파서 표의 73개 이름과 컴포지터의 순서 · 패턴만 쓴다.
3. **줄 단위로 묶는다.** 모든 커맨드는 어느 대사 줄 앞에 붙는다. 줄 중간에 시작한 변화는 `pause` · `Nfr`로 나타낸다.
4. **시간과 이징은 런타임의 것만.** 시간은 24fps 프레임 토큰, 이징은 `ExportedTuning/ease-golden.json` 표본이다.
5. **화면 예측은 코어가 한다.** 오라클(아래)만 쓴다. 파이썬에 같은 수식을 다시 쓰지 않는다.
6. **폴드로 닫는다.** 낸 커맨드를 폴드한 무대가 관측과 맞아야 한다. 남은 차이가 그 줄의 신뢰도다.
7. **모르면 표시한다.** 확신이 낮은 것은 근거와 후보를 남긴다. 추측을 사실처럼 쓰지 않는다.
8. **게임에 들어가는 Yarn은 컴포지터만 낸다.** 이 도구의 `.yarn`은 시험장 · 빠른 재생용이다. 컴포지터로는 연출 층(`.vncompose.json`)으로 넘긴다(P1 이후, 컴포지터 코드로 쓴다).
9. **기밀 자료는 밖에 내지 않는다.** 다른 게임 영상과 그 카탈로그는 저장소 · 문서 · 데모에 넣지 않는다(`/private/`는 무시 목록에 있다).

## 구조

```
ked-staging-extractor/          ← 이 저장소 (파이썬)
  ked_staging/
    oracle.py      오라클 클라이언트 (ked-stage-oracle serve와 JSON 줄로 대화)
    catalog.py     카탈로그 — 무엇이 화면에 설 수 있나 (런타임 리소스 가져오기)
    render.py      예측 화면 — 스냅샷 + 카탈로그 그림 → 정착 화면
    ease.py        DOTween 이징 표 (ease-golden.json)
    tokens.py      시간 · 거리 토큰 (fr · s · u)
    testbed.py     시험장 — 녹화와 예측의 정렬, 슬롯 어긋남 측정
    synthetic.py   합성 녹화 — P0 설치 점검 · P1 시험 녹화(대사창 · 타자기 · 전이 + 정답)
    testkit.py     시험용 초상 — 표정 · 변형이 화면으로 구별되는 그림(견본 초상은 모두 같은 그림이다)
    dialogue.py    대사창 자리(surface-layout) · 합성 녹화용 그리기 · 게임텍스트추출기 프로필
    lines.py       줄 경계 — 본문 잉크가 지워지는 순간 = 줄 머리
    gte.py         게임텍스트추출기 결과 엑셀 → 줄마다 화자 · 대사
    observe.py     관측 — 배경 · 인물(정체 · 변형 · 표정) · 자리(어휘 후보 + move_by) · 반전
    inverse.py     역해석 — 앞 줄과의 차이 → 정본 커맨드
    emit.py        출력 — *.staging.json · .yarn
    pipeline.py    추출 한 번(위 순서대로)
    timing.py      시간과 이징 — 흐름마다 진행도 재기 · (시작, 길이, 이징) 맞추기 · pause 끼우기
    evaluate.py    평가 — 추출 .yarn을 다시 접어 정답과 줄마다 비교(상태 · 대사 · 시간)
  docs/oracle-protocol.md · docs/PLAN-P1.md
  testbed/manifest.example.json · testbed/scenarios/p1_basic.yarn
  RECORDING.md   실제 런타임 녹화 안내(필요할 때)

ked-story-compositor/src/Vn.StageOracle/   ← 오라클 (.NET, 컴포지터의 Core 사본 · 프리뷰 투영을 그대로 쓴다)
ked-presentation-runtime/                  ← 튜닝 덤프 · 리소스 · 대본 (읽기만)
```

네 저장소가 같은 폴더(예: `Documents\GitHub`)에 나란히 있다고 가정합니다. 다르면 `--runtime` · `--oracle`(또는
환경 변수 `KED_RUNTIME` · `KED_STAGE_ORACLE`)로 알려 주세요.

## 진행

**P0 기반 — 끝(2026-11-08)**

- [x] 오라클: 줄 묶음 · 접기 · 화면 투영 (`ked-stage-oracle`)
- [x] 이징 표 · 시간 · 거리 토큰
- [x] 런타임 카탈로그 가져오기
- [x] 예측 화면 그리기
- [x] 시험장: 녹화 ↔ 예측 정렬, 슬롯 어긋남(px) 측정, 합성 녹화 점검
- [x] 통과 기준 — 합성 녹화를 런타임 원본과 눈으로 대조해 같다고 확인했다(실측 1px 대조는 하지 않았다.
      실제 런타임 녹화로 재려면 [RECORDING.md](RECORDING.md))

**P1 구조 — 끝(2026-11-08)** · 계획 · 결과는 [docs/PLAN-P1.md](docs/PLAN-P1.md)

- [x] 시험용 표정 초상(testkit)과 P1 어휘를 고루 쓰는 시험 대본(`testbed/scenarios/p1_basic.yarn`)
- [x] 시험 녹화: 대사창 · 화자 · 타자기 · 무대 전이 + 정답(줄 머리 · 대사 · 커맨드)
- [x] 줄 경계: 정답 줄 머리와 1프레임 안
- [x] 대사: 게임텍스트추출기 결과 엑셀을 줄에 붙인다(프로필도 이 도구가 만든다)
- [x] 관측 · 역해석 · 출력(staging.json · .yarn)
- [x] 통과 기준 — 시험 대본 18줄 상태 일치 100%. 실제 게임텍스트추출기를 거쳐도 상태 100%, 대사 글자 정확도 98.6%

**P2 시간과 이징 — 끝(2026-11-08)** · 방법 · 결과는 [docs/P2.md](docs/P2.md)

- [x] 오라클 `timeline` — 줄 안의 시간 축(커맨드마다 시작 · 길이 · 이징, 대기, 페이드 투명도). 컴포지터 `StageMotionPlan.ApplyAt`
- [x] 시험 녹화를 실제 트윈으로(`scenario --timed`), 정답에 줄마다 시간표
- [x] 흐름(움직임 · 투명도)마다 진행도를 영상에서 재고 (시작 · 길이 · 이징 35종)을 맞춘다, 늦은 시작은 `pause`
- [x] 통과 기준 — `p2_timing.yarn`(10줄, 흐름 14)과 `p1_basic.yarn`(흐름 15)에서 길이 2fr 안 100%, 시작 2fr 안 100%, 이징 일치 100%

**P3 남은 어휘 — 다음**

## 설치 (Windows)

```powershell
# 1) 파이썬 3.10 이상 — 없으면 설치 후 PowerShell을 새로 연다
#    ("Python was not found"가 나오면 Microsoft Store 바로가기가 가로챈 것이다 — 아래 설치)
winget install -e --id Python.Python.3.12
py -m pip install -r requirements.txt

# 2) 오라클 빌드 — 프로젝트는 옆 저장소 ked-story-compositor에 있다(이 폴더에서 그대로 실행)
dotnet build ..\ked-story-compositor\src\Vn.StageOracle -c Release

# 3) 점검 — 단위 테스트, 합성 녹화로 시험장 도구 확인(약 2~3분)
py -m unittest discover -s tests -t .
py -m ked_staging synth-check ..\ked-presentation-runtime\Assets\Dialogue\ChapterProgression\presentation_demo_Demo_Presentation.yarn

# 4) P1 · P2 점검 — 시험 녹화 → 추출 → 평가(각 1~2분)
py -m ked_staging p1-check
py -m ked_staging p2-check
```

오라클 프로토콜이 2로 올랐습니다(P2). 컴포지터를 받은 뒤 2)의 빌드를 다시 해 주세요.

단위 테스트 일부는 오라클 빌드 · 옆 런타임 저장소 · 한글 글꼴(Windows는 맑은 고딕)이 있어야 돕니다. 없으면 그 테스트만
건너뜁니다.

## 사용

```powershell
# 카탈로그 (초상 · 배경 목록) — 다른 명령은 없으면 매번 런타임에서 새로 만든다
py -m ked_staging catalog -o catalog.json

# 대본 경로의 예측 화면 — 줄마다 PNG와 snapshots.json
py -m ked_staging predict --catalog catalog.json --out predicted `
  ..\ked-presentation-runtime\Assets\Dialogue\ChapterProgression\festival_eve_Festival_Classroom.yarn

# 시험장 — 녹화마다 정렬 · 어긋남 측정, report.json과 상태별 비교 이미지
py -m ked_staging testbed testbed\manifest.json --catalog catalog.json --out testbed-out
```

### 연출 뽑기 (P1)

```powershell
# 1) 게임텍스트추출기용 프로필 — 런타임 대사창(surface-layout 'bottom') 자리를 비율 좌표로
py -m ked_staging gte-profile -o ked-runtime.gte-profile.json
#    → 게임텍스트추출기 profiles 폴더에 넣고, 그 도구로 영상을 돌려 '…_텍스트.xlsx'를 얻는다(번역가가 고친 엑셀도 된다)

# 2) 추출 — 줄 · 대사 · 배경 · 인물 · 표정 · 자리 → extract-out\<영상>.staging.json · <영상>.yarn
py -m ked_staging extract 영상.mp4 --text 영상_텍스트.xlsx --out extract-out

# 시험 녹화를 직접 만들어 보기 — 표정이 구별되는 시험용 초상을 testkit 폴더에 만들어 쓴다(--timed: 실제 트윈)
py -m ked_staging scenario testbed\scenarios\p2_timing.yarn --timed --testkit testkit --out scenario-out
py -m ked_staging extract scenario-out\scenario.mp4 --text scenario-out\gte-simulated.xlsx --testkit testkit --out extract-out
py -m ked_staging evaluate scenario-out\truth.json extract-out\scenario.staging.json
```

`scenario`는 `gte-simulated.xlsx`(정답에서 흉내 낸 엑셀)와 `gte-profile.json`도 함께 냅니다. 실제 게임텍스트추출기를
`scenario.mp4`에 돌린 엑셀을 `--text`로 주면 OCR까지 거친 결과를 잴 수 있습니다.

`.staging.json`은 줄마다 줄 머리 · 다 찍힘 · 끝(초), 화자 · 대사 · 붙은 엑셀 줄, 관측(배경 · 인물마다 자리 후보 이름 ·
move_by · 오차), 흐름마다 잰 시간(`timing`: 시작 · 길이 프레임 · 이징 · 맞춤 오차), 낸 커맨드, 확인할 까닭을 담습니다.
시간을 재지 않으려면 `extract --no-timing`. `.yarn`은 같은 내용을 대본 모양으로 편 초안이고, 확인할 까닭은
`// 확인:` 주석으로 붙습니다.

## 알려진 한계

**P2**

- 시간은 보이는 변화만 잽니다 — 보이는 채 바뀐 자리(움직임)와 페이드. 숨은 채 놓는 자리는 `0fr`, 즉시 커맨드(face · pose ·
  mirror · bg_sprite)는 줄 머리에 둡니다(대기 뒤의 즉시 커맨드는 아직 줄 머리로 당겨집니다).
- 한 슬롯의 여러 움직임(place와 size를 다른 시간으로)은 하나의 흐름으로 잽니다.
- 다음 줄 머리에서 남은 트윈이 가속되므로(클레임 ×30) 줄을 빨리 넘긴 영상에서는 길이가 짧게 잡힐 수 있습니다 — 맞춤 오차가
  크면 `// 확인`이 붙습니다.
- OutQuart · OutQuint처럼 꼬리만 다른 이징은 짧은 흐름에서 헷갈릴 수 있습니다.

**P1**

- 슬라이드(`slide_in`)는 place + fade_in으로 냅니다(등장하며 움직이는 것은 P3).
- 자리는 `place_left/center/right × focus(bust · face) × size 프리셋`의 조합 + `move_by`(0.25u 단위)로만 설명합니다.
  `place`(임의 지점) · 회전 · 배율 · 샷 · 모션 · 캐릭터 비주얼 · 화면 효과는 P3입니다. 그런 줄은 `// 확인: 설명 못 한 화면 n%`가 붙습니다.
- 인물은 한 슬롯에 한 명(런타임 cast 규칙), 층은 언제나 `mid`로 냅니다(샷이 없으면 화면에 차이가 없습니다).
- 같은 깊이의 두 인물이 겹칠 때 그리기 순서(슬롯 키 순)가 원본과 다를 수 있습니다(`sibling_*`는 P3).
- 반투명 `fade`(0과 1 사이)는 보지 않습니다. 보이거나 안 보이거나입니다.
- 대사창 bottom 프리셋을 가정합니다(`--preset`으로 바꿀 수 있지만 줄 안에서 프리셋이 바뀌는 영상은 아직 다루지 않습니다).
- 시험장 검증은 합성 녹화(오라클 예측 그대로)로 했습니다. 실제 런타임 녹화의 압축 · 트윈 중간 프레임에서는 더 흔들릴 수 있습니다.

**P0**

- 예측 화면은 코어가 접는 어휘만 반영합니다. `mirror`는 컴포지터 규칙으로 따라가지만, 캐릭터 비주얼 · 화면 효과 · 블러 · `tx` 마스크 · `sibling_*`는 반영하지 않습니다. 그런 줄은 시험장에서 어긋나 보일 수 있습니다.
- 정착 상태(트윈이 끝난 뒤)만 예측합니다. 시간 축은 P2입니다.
- 대사창 띠(화면 아래 약 1/3)는 대조에서 가립니다.
- 화면이 같은 줄들(예: 같은 그림을 쓰는 표정 교체)은 영상으로 구별할 수 없어 하나로 묶어 잽니다. 견본 초상은 표정 그림이 모두 같습니다.
- 그리기 순서는 컴포지터 프리뷰와 같은 근사(무대 → 층 → 깊이 → 키)입니다.
