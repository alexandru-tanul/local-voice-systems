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
# Mic to Voice uses MODEL_NAME; dataset transcripts ask for a larger model, loaded beside it.
MODELS = {}
MODEL_LOCK = threading.Lock()
LOAD_LOCKS = {}


def load_model(name=None):
    name = name or MODEL_NAME
    model = MODELS.get(name)
    if model is not None:
        return model
    # A lock for each model, so the first download of the dataset model, which takes minutes, does
    # not hold up Mic to Voice.
    with MODEL_LOCK:
        lock = LOAD_LOCKS.setdefault(name, threading.Lock())
    with lock:
        if name not in MODELS:
            from faster_whisper import WhisperModel

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            MODELS[name] = WhisperModel(
                name,
                device="cpu",
                compute_type="int8",
                download_root=str(MODEL_DIR),
            )
        return MODELS[name]


def transcribe_wav(wav_bytes, model_name=None, language="en", beam_size=1, vad=True):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="relay_", suffix=".wav", dir=str(TMP_DIR))
    os.close(fd)
    wav_path = Path(path)
    try:
        wav_path.write_bytes(wav_bytes)
        started = time.time()
        model = load_model(model_name)
        segments, info = model.transcribe(
            str(wav_path),
            language=language,
            beam_size=beam_size,
            # Voice detection skips the silence around speech from a microphone, but it also takes
            # some processed or game voices for noise; dataset clips are already cut at pauses.
            vad_filter=vad,
            condition_on_previous_text=False,
            no_speech_threshold=0.55,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return {
            "ok": True,
            "text": text,
            "language": getattr(info, "language", language),
            "duration": round(time.time() - started, 3),
            "model": model_name or MODEL_NAME,
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
            self.send_json({"ok": True, "model": MODEL_NAME, "loaded": MODEL_NAME in MODELS})
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
            self.send_json(transcribe_wav(
                base64.b64decode(audio_b64),
                payload.get("model") or None,
                payload.get("language") or "en",
                int(payload.get("beam_size") or 1),
                payload.get("vad", True) is not False,
            ))
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
