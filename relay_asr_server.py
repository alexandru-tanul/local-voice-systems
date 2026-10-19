import base64
import json
import os
import re
import socket
import tempfile
import threading
import time
import urllib.parse
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
# Mic to Voice translates with NLLB-200 600M, converted for CTranslate2, which faster-whisper already
# brings. Its license, CC-BY-NC 4.0, allows only non-commercial use. The revision is pinned, so a
# change in that repository cannot change what is downloaded; its tokenizer files are those of
# facebook/nllb-200-distilled-600M.
TRANSLATION_REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
TRANSLATION_REVISION = "302d78f00e6fdb50a1064059df7c392b735e9d05"
TRANSLATION_FILES = ["config.json", "model.bin", "shared_vocabulary.txt", "tokenizer.json"]
# The languages GPT-SoVITS speaks, by NLLB's names for them.
TRANSLATION_CODES = {"en": "eng_Latn", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "yue": "yue_Hant"}
TRANSLATOR = {}


def model_lock(name):
    # A lock for each model, so the first download of one, which takes minutes, does not hold up
    # the others, such as Mic to Voice's while a dataset model downloads.
    with MODEL_LOCK:
        return LOAD_LOCKS.setdefault(name, threading.Lock())


def load_model(name=None):
    name = name or MODEL_NAME
    model = MODELS.get(name)
    if model is not None:
        return model
    with model_lock(name):
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


def load_translator():
    """The translation model and its tokenizer, downloaded into MODEL_DIR on first use."""
    with model_lock("translation"):
        if not TRANSLATOR:
            import ctranslate2
            from huggingface_hub import snapshot_download
            from tokenizers import Tokenizer

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            options = {"revision": TRANSLATION_REVISION, "cache_dir": str(MODEL_DIR), "allow_patterns": TRANSLATION_FILES}
            try:
                # Once downloaded, it loads without the internet. A download that was cut short leaves
                # the folder without all its files, which huggingface_hub does not always notice.
                folder = snapshot_download(TRANSLATION_REPO, local_files_only=True, **options)
                if not all((Path(folder) / name).is_file() for name in TRANSLATION_FILES):
                    raise FileNotFoundError(folder)
            except Exception:
                folder = snapshot_download(TRANSLATION_REPO, **options)
            # Both are kept only when both load, so a failed load is tried again in full.
            model = ctranslate2.Translator(folder, device="cpu", compute_type="int8")
            tokenizer = Tokenizer.from_file(str(Path(folder) / "tokenizer.json"))
            TRANSLATOR.update(model=model, tokenizer=tokenizer)
        return TRANSLATOR["model"], TRANSLATOR["tokenizer"]


def translate_text(text, source, target):
    """text, in the language source, translated into target. NLLB-200 learned from single
    sentences, so each sentence is translated on its own."""
    if source not in TRANSLATION_CODES or target not in TRANSLATION_CODES:
        raise ValueError(f"Translation is from and to {', '.join(TRANSLATION_CODES)}.")
    # A sentence ends at . ! or ? before a space, so 3.5 stays one number, or at 。！？ alone.
    sentences = [part for part in re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", text) if part.strip()]
    if not sentences:
        return ""
    translator, tokenizer = load_translator()
    # NLLB reads the source language's code, the sentence and </s>, and writes the target
    # language's code before the translation.
    batch = [[TRANSLATION_CODES[source], *tokenizer.encode(part, add_special_tokens=False).tokens, "</s>"] for part in sentences]
    # CTranslate2 stops a translation at 256 pieces, which a long sentence can pass, such as a
    # recording that Whisper wrote without full stops; NLLB reads at most 1024.
    longest = max(len(tokens) for tokens in batch)
    results = translator.translate_batch(batch, target_prefix=[[TRANSLATION_CODES[target]]] * len(batch), beam_size=4,
                                         max_decoding_length=min(1024, max(256, 3 * longest)))
    parts = [tokenizer.decode([tokenizer.token_to_id(token) for token in result.hypotheses[0][1:]], skip_special_tokens=True).strip()
             for result in results]
    # Chinese and Japanese put no spaces between sentences.
    return ("" if target in {"zh", "ja", "yue"} else " ").join(parts)


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
        path, _, query = self.path.partition("?")
        if path == "/warmup":
            # Loads Mic to Voice's model, or the one named, and the translation model when asked.
            options = urllib.parse.parse_qs(query)
            name = options.get("model", [""])[0] or MODEL_NAME
            started = time.time()
            try:
                load_model(name)
            except Exception as exc:
                # For example, the first download of the model failed. The dashboard shows this text.
                self.send_json({"ok": False, "error": f"Could not load the {name} speech model: {exc}"}, 500)
                return
            if options.get("translation") == ["1"]:
                try:
                    load_translator()
                except Exception as exc:
                    self.send_json({"ok": False, "error": f"Could not load the translation model: {exc}"}, 500)
                    return
            self.send_json({"ok": True, "model": name, "loaded": True, "duration": round(time.time() - started, 3)})
            return
        self.send_error(404)

    def do_POST(self):
        try:
            if self.path not in {"/transcribe", "/translate"}:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/translate":
                started = time.time()
                text = translate_text(str(payload.get("text") or ""), payload.get("source"), payload.get("target"))
                self.send_json({"ok": True, "text": text, "duration": round(time.time() - started, 3)})
                return
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
