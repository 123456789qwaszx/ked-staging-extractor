# 시험장 녹화 안내

P0 통과 기준은 "오라클 예측이 런타임 화면과 1px 안에서 맞는가"입니다. 원본 `.yarn`을 아는 런타임 녹화가 필요합니다.

**빌드한 게임을 OBS의 게임 캡처로** 녹화합니다.

- 에디터 Game 뷰는 모니터 크기에 따라 축소돼 1px을 재기 어렵습니다.
- Unity Recorder는 녹화 프레임에 맞춰 게임 시간을 고정합니다. 그런데 이 런타임의 트윈은 timeScale과 무관한 시간(`SetUpdate(true)`)으로 돕니다. 그래서 녹화가 느려지면 연출 시간이 실제와 달라질 수 있습니다.

## 1. 준비 (한 번만)

- OBS 설치: `winget install -e --id OBSProject.OBSStudio`
- 모니터 해상도 확인: 설정 > 시스템 > 디스플레이 > 디스플레이 해상도

## 2. 게임 빌드 (한 번만)

Unity에서 ked-presentation-runtime을 엽니다.

1. **Edit > Project Settings > Player > Resolution and Presentation**
   - 모니터가 1920×1080이면: Fullscreen Mode = **Fullscreen Window**(지금 설정 그대로)
   - 모니터가 더 크면: Fullscreen Mode = **Windowed**, Default Screen Width **1920**, Height **1080**
   - **Run In Background** 켜기를 권합니다. 지금은 꺼져 있어서, 게임 창이 포커스를 잃는 순간 게임이 멈춥니다.
2. **File > Build Profiles > Windows**를 엽니다. 씬 목록에는 `PresentationSample`이 이미 있습니다.
3. **Build**를 누르고 저장소 밖 폴더(예: `Documents\ked-builds\sample`)를 고릅니다.

Player Settings 변경은 `ProjectSettings/ProjectSettings.asset`에 남습니다. 커밋하지 않으려면 녹화 후 되돌리세요.

## 3. OBS 설정 (한 번만)

파일 > 설정:

| 위치 | 값 |
|---|---|
| 비디오 | 기본(캔버스) 해상도 1920x1080 · 출력(스케일) 해상도 1920x1080 · 공통 FPS 60 |
| 출력 | 출력 모드 **고급** → 녹화 탭: 녹화 형식 MP4(하이브리드 MP4가 있으면 그것) |
| 출력 › 인코더 | 하드웨어(NVENC · AMD · QuickSync)가 있으면 그것, 없으면 x264 · 비율 제어 **CQP**(x264는 CRF) **16** · 키프레임 간격 1초 |
| 단축키 | 녹화 시작 **F9** · 녹화 중지 **F10**처럼 Ctrl 없는 키. 게임에서 Ctrl은 "빨리 넘기기"다 |

소스 설정:

1. 빌드한 게임을 한 번 실행해 둡니다.
2. 장면에 **게임 캡처** 소스를 추가합니다.
3. 모드는 "특정 창 캡처", 창은 `[ked-presentation-runtime.exe]`를 고릅니다.
4. **커서 캡처는 끕니다.** 커서가 화면에 찍히면 그 자리가 어긋나 보입니다.
5. 미리보기에서 소스를 우클릭 > 변환 > **변환 초기화**를 고릅니다. 화면에 맞춰 늘리지 말고 1:1로 둡니다.

## 4. 시험 녹화로 확인 (5초)

1. F9를 누르고 5초 뒤 F10을 누릅니다.
2. 파일을 `recordings\test.mp4`로 옮깁니다.
3. 아래 명령을 실행합니다. `1920 1080 60`이 나와야 합니다.

```powershell
py -c "import cv2; c=cv2.VideoCapture(r'recordings\test.mp4'); print(int(c.get(3)), int(c.get(4)), round(c.get(5)))"
```

## 5. 본 녹화 ① — 연출 데모

1. 게임을 실행하고, 타이틀에서 **[다른 예제로]**를 눌러 Presentation Demo로 바꿉니다(씬이 다시 열린다).
2. **F9**로 녹화를 시작하고, 게임 창을 클릭한 뒤 **[처음부터]**를 누릅니다.
3. 줄마다 기다렸다 넘깁니다: 글자가 다 나오고 인물 · 카메라 · 효과가 **완전히 멈춘 뒤 1초** 기다렸다가 `Space`.
   - "6-1. … 도중에 바로 넘긴다", "7. 이 줄도 도중에 바로 넘긴다"도 **기다립니다.**
   - `Ctrl` · `S` · `A` · `R` · 마우스 휠 · 백로그는 쓰지 않습니다. 녹화 중에는 다른 창을 클릭하지 않습니다.
4. 마지막 "캐릭터 퇴장" 뒤 화면이 검게 되면 1초 뒤 **F10**을 누릅니다.
5. 파일을 `recordings\demo_presentation.mp4`로 옮깁니다.

## 6. 본 녹화 ② — 축제: 교실 → 처음부터 다시

1. 타이틀에서 **[다른 예제로]**를 눌러 Festival Eve로 바꿉니다. **F9**를 누르고 **[처음부터]**로 시작합니다(이어하기가 아니라).
2. 교실 에피소드를 같은 규칙으로 진행합니다. 끝의 선택지에서 **"처음부터 다시 그린다"**를 고릅니다.
3. 다시 그리기 에피소드를 진행합니다.
4. 마지막 대사 "아직 같은 장면이라, 되감기(R)로 돌아가면…"이 다 나오면 1초 기다립니다. 그리고 **Space를 누르지 않고 F10**을 누릅니다. Space를 누르면 다음 에피소드로 자동으로 넘어갑니다.
5. 파일을 `recordings\festival_classroom_remake.mp4`로 옮깁니다.

`detour`가 있는 노드(`Festival_Wrapup` · `Festival_Staffroom`)는 아직 경로로 적을 수 없어 P0에서 뺍니다.

## 7. 돌리기

```powershell
copy testbed\manifest.example.json testbed\manifest.json
py -m ked_staging testbed testbed\manifest.json --out testbed-out
```

매니페스트 경로의 기준은 이렇습니다.

- `video`: 매니페스트 파일 위치
- `route`: 런타임 저장소

예시는 위 파일 이름 그대로 맞춰 두었습니다. `recordings\`는 저장소에 올라가지 않습니다(`.gitignore`).

결과는 콘솔과 `testbed-out\report.json`에 나옵니다. `testbed-out\rec00\state_###.jpg`에는 상태마다 예측 · 실제 · 겹침이 나란히 있습니다.
