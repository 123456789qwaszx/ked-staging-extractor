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
    synthetic.py   합성 녹화 (설치 점검)
  docs/oracle-protocol.md
  testbed/manifest.example.json
  RECORDING.md   시험장 녹화 안내

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

**P1 구조 — 다음** · 계획은 [docs/PLAN-P1.md](docs/PLAN-P1.md)

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
```

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

## 알려진 한계 (P0)

- 예측 화면은 코어가 접는 어휘만 반영합니다. `mirror`는 컴포지터 규칙으로 따라가지만, 캐릭터 비주얼 · 화면 효과 · 블러 · `tx` 마스크 · `sibling_*`는 반영하지 않습니다. 그런 줄은 시험장에서 어긋나 보일 수 있습니다.
- 정착 상태(트윈이 끝난 뒤)만 예측합니다. 시간 축은 P2입니다.
- 대사창 띠(화면 아래 약 1/3)는 대조에서 가립니다.
- 화면이 같은 줄들(예: 같은 그림을 쓰는 표정 교체)은 영상으로 구별할 수 없어 하나로 묶어 잽니다. 견본 초상은 표정 그림이 모두 같습니다.
- 그리기 순서는 컴포지터 프리뷰와 같은 근사(무대 → 층 → 깊이 → 키)입니다.
