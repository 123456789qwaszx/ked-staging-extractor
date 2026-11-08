// 헤드리스로 ked-motion-tool(Mt.Authoring)의 프리셋·검증·내보내기를 그대로 쓴다 — 앱의 "motions.json 가져오기 → 프리셋에서 추가 → 내보내기"와 같은 길.
using Mt.Authoring.Editing;
using Mt.Authoring.Model;
using Mt.Authoring.Presets;
using Mt.Authoring.Rig;
using Mt.Authoring.Serialization;
using Mt.Authoring.Validation;

string motionsPath = args[0], rigPath = args[1], outJson = args[2], outAuthoring = args[3];
var ids = args.Skip(4).ToArray();
var rig = RigSchemaFile.Load(rigPath);
var controllers = BuiltinTuning.Controllers;
var editor = new MotionEditor(new MotionDocument());
if (!MotionsJson.TryRead(File.ReadAllText(motionsPath), out var clips, out var problems)) { Console.Error.WriteLine(string.Join("\n", problems)); return 1; }
Console.WriteLine("가져옴: " + string.Join(", ", editor.ImportClips(clips, replace: true)));
foreach (var id in ids)
{
    var preset = MotionPresets.Find(id) ?? throw new Exception("프리셋 없음: " + id);
    if (editor.Document.Find(id) is not null) { Console.WriteLine($"이미 있음: {id} — 건너뜀"); continue; }
    var table = controllers.Rig("character") ?? throw new Exception("controllers: character 없음");
    Console.WriteLine($"프리셋 추가: {MotionPresets.Add(editor, preset, "character", table)} ({preset.Label}, {preset.Duration}s, {preset.End})");
}
var issues = MotionValidator.Validate(editor.Document, rig.NodeIds, controllers);
foreach (var p in issues) Console.WriteLine("문제: " + p);
if (!MotionExport.TryExport(editor.Document, rig.NodeIds, out string json, out var exportProblems)) { Console.Error.WriteLine($"내보내기 막힘 {exportProblems.Count}건"); foreach (var p in exportProblems) Console.Error.WriteLine(p); return 2; }
File.WriteAllBytes(outJson, new System.Text.UTF8Encoding(false).GetBytes(json));
File.WriteAllText(outAuthoring, MotionDocumentJson.Write(editor.Document));
Console.WriteLine($"내보냄: {outJson} — 모션 {editor.Document.Motions.Count}개");
return 0;
