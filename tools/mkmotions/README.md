# mkmotions — ked-motion-tool을 화면 없이

ked-motion-tool 앱에서 "motions.json 가져오기 → 프리셋에서 추가 → 내보내기"를 한 것과 같은 일을 명령줄로 한다.
같은 라이브러리(`Mt.Authoring`)의 프리셋 · 검사(`MotionValidator`) · 내보내기(`MotionExport`)를 그대로 부른다.

```
dotnet run --project tools/mkmotions -- <motions.json> <rig-schemas.json> <out motions.json> <out .motion-authoring.json> [프리셋 id ...]
# 예: 런타임 motions.json에 hop · flinch를 더해 내보내기
dotnet run --project tools/mkmotions -- ..\ked-presentation-runtime\Assets\Dialogue\motions.json ..\ked-presentation-runtime\ExportedTuning\rig-schemas.json out\motions.json out\motions.motion-authoring.json hop flinch
```

- 리그 노드는 runtime의 `ExportedTuning/rig-schemas.json`으로 검사한다. 기존 모션은 바이트 그대로 왕복된다(확인함).
- `.motion-authoring.json`은 ked-motion-tool에서 열어 다듬는 저작 원본이다.
- 저장소가 형제 폴더가 아니면 `-p:KedMotionToolDir=<경로>`.
- 다음 할 일: 영상에서 잰 모션 초안(motion-authoring/2)을 받아 같은 길로 내보내기(지금은 프리셋만).
