import base64
import json
import os
import socket
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("RELAY_ASR_MODEL_DIR", str(ROOT / ".cache" / "faster-whisper")))
TMP_DIR = Path(os.environ.get("RELAY_ASR_TMP", str(ROOT / "outputs" / "relay_asr_tmp")))
MODEL_NAME = os.environ.get("RELAY_ASR_MODEL", "base.en")
MODEL = None
MODEL_LOCK = threading.Lock()


def load_model():
    global MODEL
    with MODEL_LOCK:
        if MODEL is None:
            from faster_whisper import WhisperModel

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            MODEL = WhisperModel(
                MODEL_NAME,
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_DIR),
            )
        return MODEL


def transcribe_wav(wav_bytes):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="relay_", suffix=".wav", dir=str(TMP_DIR))
    os.close(fd)
    wav_path = Path(path)
    try:
        wav_path.write_bytes(wav_bytes)
        started = time.time()
        model = load_model()
        segments, info = model.transcribe(
            str(wav_path),
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.55,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return {
            "ok": True,
            "text": text,
            "language": getattr(info, "language", "en"),
            "duration": round(time.time() - started, 3),
            "model": MODEL_NAME,
        }
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"ok": True, "model": MODEL_NAME, "loaded": MODEL is not None})
            return
        if self.path == "/warmup":
            started = time.time()
            try:
                load_model()
            except Exception as exc:
                # For example, the first download of the model failed. The dashboard shows this text.
                self.send_json({"ok": False, "error": f"Could not load the {MODEL_NAME} speech model: {exc}"}, 500)
                return
            self.send_json({"ok": True, "model": MODEL_NAME, "loaded": True, "duration": round(time.time() - started, 3)})
            return
        self.send_error(404)

    def do_POST(self):
        try:
            if self.path != "/transcribe":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            audio_b64 = payload.get("audio_base64", "")
            if not audio_b64:
                self.send_json({"ok": False, "error": "No audio supplied."}, 400)
                return
            self.send_json(transcribe_wav(base64.b64decode(audio_b64)))
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, 500)


class Server(ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second server bind a port that is already in use, and the
    # two then share its requests. SO_EXCLUSIVEADDRUSE makes the second bind fail instead.
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


if __name__ == "__main__":
    port = int(os.environ.get("RELAY_ASR_PORT", "8792"))
    server = Server(("127.0.0.1", port), Handler)
    print(f"Relay ASR server starting on http://127.0.0.1:{port} with {MODEL_NAME}", flush=True)
    server.serve_forever()
