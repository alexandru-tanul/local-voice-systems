import json
import subprocess
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


import argparse

ROOT = Path(__file__).resolve().parent
try:
    _CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    _CONFIG = {}
_parser = argparse.ArgumentParser(description="Small status page for an RVC training run in Applio.")
_parser.add_argument("--model", required=True, help="RVC model name, the folder under Applio's logs directory.")
_parser.add_argument("--applio-root", default=_CONFIG.get("applio_root", str(ROOT.parent / "Applio")))
_parser.add_argument("--dashboard", default="http://localhost:8790")
_ARGS = _parser.parse_args()
APPLIO_MODEL_DIR = Path(_ARGS.applio_root) / "logs" / _ARGS.model
MAIN_STATE_URL = _ARGS.dashboard.rstrip("/") + "/api/state"


def active_pid(pid):
    try:
        result = subprocess.run(
            ["tasklist", "/fi", f"PID eq {pid}", "/fo", "csv", "/nh"],
            text=True,
            capture_output=True,
            timeout=2,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return str(pid) in result.stdout and "No tasks" not in result.stdout
    except Exception:
        return False


def count_files(folder, pattern):
    path = APPLIO_MODEL_DIR / folder
    return len(list(path.glob(pattern))) if path.exists() else 0


def read_main_state():
    try:
        with urllib.request.urlopen(MAIN_STATE_URL, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc), "jobs": {}, "voicechanger": {}, "log": ""}


def current_state():
    state = read_main_state()
    job = (state.get("jobs") or {}).get("train-rvc") or {}
    lines = job.get("lines") or []

    config_pids = []
    config_path = APPLIO_MODEL_DIR / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config_pids = [int(pid) for pid in config.get("process_pids", [])]
        except Exception:
            config_pids = []

    active_config_pids = [pid for pid in config_pids if active_pid(pid)]
    checkpoints = []
    if APPLIO_MODEL_DIR.exists():
        for path in APPLIO_MODEL_DIR.glob("*.pth"):
            checkpoints.append(
                {
                    "name": path.name,
                    "mb": round(path.stat().st_size / (1024 * 1024), 1),
                    "time": datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S"),
                }
            )
    checkpoints.sort(key=lambda row: row["time"], reverse=True)

    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "job_status": job.get("status", "unknown"),
        "job_returncode": job.get("returncode"),
        "active_trainer_pids": active_config_pids,
        "process_pids": (state.get("voicechanger") or {}).get("training_pids") or [],
        "sliced": count_files("sliced_audios", "*.wav"),
        "spec": count_files("sliced_audios", "*.spec.pt"),
        "features": count_files("extracted", "*.npy"),
        "pitch": count_files("f0", "*.npy"),
        "filelist_rows": len((APPLIO_MODEL_DIR / "filelist.txt").read_text(encoding="utf-8").splitlines())
        if (APPLIO_MODEL_DIR / "filelist.txt").exists()
        else 0,
        "latest_checkpoint": checkpoints[0] if checkpoints else None,
        "lines": lines[-120:],
        "main_error": state.get("error"),
    }


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RVC Training Monitor</title>
  <style>
    body { margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #10151d; color: #e8edf5; }
    main { max-width: 1100px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 24px; margin: 0 0 16px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .card { background: #1a2330; border: 1px solid #334155; border-radius: 8px; padding: 12px; }
    .label { color: #9fb0c6; font-size: 12px; margin-bottom: 6px; }
    .value { font-size: 18px; font-weight: 700; }
    .good { color: #8ff0b4; }
    .warn { color: #ffd37a; }
    pre { white-space: pre-wrap; background: #070b12; border: 1px solid #334155; border-radius: 8px; padding: 14px; min-height: 320px; }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>RVC Training Monitor</h1>
  <button id="stop" style="margin: 0 0 14px; background:#b42318; color:white; border:0; border-radius:6px; padding:10px 14px; font-weight:700; cursor:pointer;">Stop RVC Training</button>
  <div class="grid">
    <div class="card"><div class="label">Status</div><div class="value" id="status">...</div></div>
    <div class="card"><div class="label">Trainer PID</div><div class="value" id="pid">...</div></div>
    <div class="card"><div class="label">Spec cache</div><div class="value" id="spec">...</div></div>
    <div class="card"><div class="label">Checkpoint</div><div class="value" id="ckpt">...</div></div>
  </div>
  <div class="card" style="margin-top: 12px;">
    <div class="label">Prepared data</div>
    <div id="data">...</div>
  </div>
  <h1 style="margin-top: 18px;">Trainer Output</h1>
  <pre id="log">Loading...</pre>
</main>
<script>
async function refresh() {
  const res = await fetch('/state');
  const s = await res.json();
  const alive = s.active_trainer_pids.length > 0 || s.job_status === 'running';
  document.getElementById('status').textContent = alive ? 'Running' : s.job_status;
  document.getElementById('status').className = 'value ' + (alive ? 'good' : 'warn');
  document.getElementById('pid').textContent = s.active_trainer_pids.join(', ') || s.process_pids.join(', ') || 'none';
  document.getElementById('spec').textContent = `${s.spec} / ${s.sliced}`;
  document.getElementById('ckpt').textContent = s.latest_checkpoint ? `${s.latest_checkpoint.name} (${s.latest_checkpoint.mb} MB)` : 'none yet';
  document.getElementById('data').textContent = `${s.features} embeddings, ${s.pitch} pitch files, ${s.filelist_rows} filelist rows. Last checked ${s.time}.`;
  document.getElementById('log').textContent = s.lines.length ? s.lines.join('\\n') : 'No trainer text captured yet.';
}
document.getElementById('stop').onclick = async () => {
  await fetch('/stop', {method: 'POST'});
  await refresh();
};
refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/state":
            self.send_json(current_state())
            return
        encoded = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        if self.path == "/stop":
            data = json.dumps({"key": "train-rvc"}).encode("utf-8")
            request = urllib.request.Request(
                "http://localhost:8790/api/stop-job",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    response.read()
                self.send_json({"ok": True})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)})
            return
        self.send_json({"ok": False, "error": "Unknown endpoint"})

    def log_message(self, *_):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", 8791), Handler)
    print("RVC monitor running on http://localhost:8791", flush=True)
    server.serve_forever()
