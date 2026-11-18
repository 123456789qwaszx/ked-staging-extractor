"""오라클의 프레임 기준 확인 — 추출기와 다른 fps로 세는 오라클은 띄우지 않는다(가짜 오라클로 본다)."""
import os
import sys
import tempfile
import unittest

from ked_staging.oracle import Oracle, OracleError, PROTOCOL
from ked_staging.tokens import FPS

# serve 프로토콜의 ping · timeline만 답하는 가짜 오라클. 첫 인자가 그 오라클의 fps다.
FAKE = r'''
import json, re, sys
fps = float(sys.argv[1])
for raw in sys.stdin:
    req = json.loads(raw)
    out = {"id": req["id"], "ok": True}
    if req["op"] == "ping":
        out.update(protocol=sys.argv[2], width=1920, height=1080, problems=[])
    elif req["op"] == "timeline":
        cmds = req["lines"][-1]["commands"]
        out.update(end=0, snapshots=[], schedule=[
            {"index": i, "name": c["name"], "args": c["args"], "start": 0,
             "duration": int(re.match(r"(\d+)fr", c["args"][0]).group(1)) / fps, "wait": True}
            for i, c in enumerate(cmds) if c["name"] == "pause"])
    else:
        out = {"id": req["id"], "ok": False, "error": "모르는 op"}
    print(json.dumps(out), flush=True)
'''


class FrameBasisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = os.path.join(self.tmp.name, "fake_oracle.py")
        with open(self.script, "w", encoding="utf-8") as f:
            f.write(FAKE)

    def tearDown(self):
        self.tmp.cleanup()

    def oracle(self, fps: float) -> Oracle:
        return Oracle(self.tmp.name, command=[sys.executable, self.script, str(fps), PROTOCOL], timeout=10)

    def test_same_fps_starts(self):
        with self.oracle(FPS) as o:
            self.assertEqual((o.width, o.height), (1920.0, 1080.0))

    def test_other_fps_is_refused(self):
        with self.assertRaises(OracleError) as caught:
            self.oracle(24)
        self.assertIn("프레임 기준", str(caught.exception))
        self.assertIn("24fps", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
