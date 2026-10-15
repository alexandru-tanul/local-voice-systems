"""The web server: the page from static/, and the /api and /v1 addresses it answers. main()
starts it."""

import hmac
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import (asr, clip_check, compare, config, cutting, datasets, engine, jobs, models, restore, speech_api,
               system, training, voices, wsl)


# The page: index.html with its app.css and app.js. They are read for each request, so an edit
# shows when the page is reloaded, without a restart.
STATIC_DIR = config.ROOT / "static"
STATIC_TYPES = {"html": "text/html", "css": "text/css", "js": "text/javascript"}
CANCELLED_GENERATIONS = set()


def read_upload(handler, limit=4 * 1024 ** 3):
    """Save the request body, a file the page sends as it is, to a temporary file."""
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0 or length > limit:
        raise RuntimeError("Files must be between 1 byte and 4 GB.")
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="upload-", suffix=".audio", dir=str(config.TMP_DIR))
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            left = length
            while left:
                chunk = handler.rfile.read(min(left, 1024 * 1024))
                if not chunk:
                    raise RuntimeError("The upload stopped before the whole file arrived.")
                handle.write(chunk)
                left -= len(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def clean_old_temp_files(max_age=3600):
    """Remove the files a crash or a closed dashboard left in TMP_DIR: uploads, cut clips and
    voice files. Only older ones go, as another dashboard may share TMP_DIR."""
    patterns = ("upload-*", "cut-*", "transcribe-*", "import-*", "voice-export-*.zip")
    for pattern in patterns:
        for path in config.TMP_DIR.glob(pattern):
            try:
                if path.is_file() and time.time() - path.stat().st_mtime > max_age:
                    path.unlink()
            except OSError:
                pass


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    if length > 150 * 1024 * 1024:
        raise RuntimeError("Request is larger than the 150 MB upload limit.")
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def latest_log_text():
    # Job output reaches the page in the "jobs" snapshots. The inference engine is not a job,
    # so its output is added here.
    chunks = []
    with jobs.STATE_LOCK:
        chunks.extend(jobs.APP_LOG[-80:])
        if engine.API_PROCESS:
            chunks.append("")
            chunks.append(f"--- {engine.API_PROCESS.name}: {engine.API_PROCESS.status} ---")
            chunks.extend(engine.API_PROCESS.lines[-180:])
    return "\n".join(chunks)


class DashboardServer(ThreadingHTTPServer):
    # On Windows, SO_REUSEADDR lets a second server bind a port that is already in use, and the
    # two then share its requests. SO_EXCLUSIVEADDRUSE makes the second bind fail instead.
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # The browser closed the connection, as an audio element does when it seeks.
            pass

    def request_allowed(self):
        """Accept requests addressed to this dashboard, sent by its own page or by a local tool.

        The Host check stops DNS rebinding, where another site's name resolves to 127.0.0.1.
        The Origin check stops other sites' pages; local tools such as ptt_helper.py send no Origin."""
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        # Programs in Docker reach this PC as host.docker.internal, a name no other site can own.
        if urllib.parse.urlparse(self.path).path.startswith("/v1/"):
            hosts.add(f"host.docker.internal:{port}")
        if self.headers.get("Host", "").lower() not in hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin.lower() in {f"http://{host}" for host in hosts}

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def speech_api_allowed(self):
        """With api_key in config.json, the speech API speaks only for requests that carry it."""
        if not speech_api.SPEECH_API_KEY or hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {speech_api.SPEECH_API_KEY}".encode()):
            return True
        self.send_json(speech_api.SpeechApiError(401, "Incorrect API key. Use the api_key from config.json.").body(), 401)
        return False

    def send_audio(self, data, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text, content_type="text/html"):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, name):
        """Sends a file of the page from static/. Only a plain file name is served, so no path leads
        out of the folder, and send_text's no-store keeps a browser from using an old app.js after
        an update."""
        match = re.fullmatch(r"[A-Za-z0-9_-]+\.(html|css|js)", name)
        file = STATIC_DIR / name
        if not match or not file.is_file():
            self.send_error(404)
            return
        self.send_text(file.read_text(encoding="utf-8"), STATIC_TYPES[match.group(1)])

    def do_GET(self):
        if not self.request_allowed():
            self.send_error(403)
            return
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if path == "/":
            self.send_static("index.html")
            return
        if path.startswith("/static/"):
            self.send_static(path[len("/static/") :])
            return
        # The lists answer without api_key: Open WebUI sends the key only with speech requests.
        if path in {"/v1/audio/voices", "/v1/audio/models", "/v1/models"}:
            if path == "/v1/audio/voices":
                self.send_json({"voices": speech_api.speech_voices()})
            elif path == "/v1/audio/models":
                self.send_json({"models": [{"id": speech_api.SPEECH_MODEL}]})
            else:
                self.send_json({"object": "list", "data": [{"id": speech_api.SPEECH_MODEL, "object": "model", "created": 0, "owned_by": "local-voice-systems"}]})
            return
        if path == "/api/ptt-state":
            with system.PTT_CONDITION:
                self.send_json(dict(system.PTT_STATE))
            return
        if path == "/api/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_sequence = -1
            try:
                while True:
                    with system.PTT_CONDITION:
                        system.PTT_CONDITION.wait_for(
                            lambda: system.PTT_STATE["sequence"] != last_sequence,
                            timeout=15,
                        )
                        event = dict(system.PTT_STATE)
                    if event["sequence"] != last_sequence:
                        last_sequence = event["sequence"]
                        payload = json.dumps(event)
                        self.wfile.write(f"event: ptt\ndata: {payload}\n\n".encode("utf-8"))
                    else:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                pass
            return
        if path == "/api/setup-status":
            self.send_json(system.setup_status())
            return
        if path == "/api/system-state":
            self.send_json({"system": system.system_snapshot()})
            return
        if path == "/api/state":
            query = urllib.parse.parse_qs(parsed_url.query)
            dataset_rows = datasets.list_datasets()
            dataset_ids = {row["id"] for row in dataset_rows}
            dataset_id = query.get("dataset", [""])[0]
            if dataset_id not in dataset_ids:
                dataset_id = dataset_rows[0]["id"] if dataset_rows else ""
            all_references = datasets.parse_references(dataset_id, valid_only=False)
            data = {
                "config_error": config.CONFIG_ERROR,
                "datasets": dataset_rows,
                "reference_dataset": dataset_id,
                "models": models.polled_model_files(),
                "models_listed": models.MODEL_LIST["listed"],
                "references": [ref for ref in all_references if ref["valid_reference"]],
                "all_references": all_references,
                "api_ready": engine.api_ready(),
                "gpu_busy": jobs.gpu_busy(),
                "restore": {"installed": restore.RESTORE_PYTHON.exists(), "items": restore.restore_items()},
                "relay_asr_ready": asr.relay_asr_ready(),
                "jobs": {key: proc.snapshot() for key, proc in jobs.jobs_now()},
                "system": system.system_snapshot(),
                "log": latest_log_text(),
            }
            self.send_json(data)
            return
        if path.startswith("/api/voice-download/"):
            name = path.rsplit("/", 1)[1]
            file = config.TMP_DIR / name
            if not voices.VOICE_EXPORT.match(name) or not file.is_file():
                self.send_error(404)
                return
            download = urllib.parse.parse_qs(parsed_url.query).get("name", ["voice.zip"])[0]
            download = re.sub(r"[^A-Za-z0-9._-]+", "-", download) or "voice.zip"
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
                self.send_header("Content-Length", str(file.stat().st_size))
                self.end_headers()
                with file.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile, 1024 * 1024)
            finally:
                file.unlink(missing_ok=True)
            return
        if path == "/api/dataset-clips":
            query = urllib.parse.parse_qs(parsed_url.query)
            try:
                self.send_json({"ok": True, **datasets.dataset_clips(query.get("dataset", [""])[0])})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 404)
            return
        if path.startswith("/datasets/"):
            self.serve_file(config.DATASETS_DIR / path[len("/datasets/") :])
            return
        if path.startswith("/outputs/"):
            self.serve_file(config.OUTPUT_DIR / path[len("/outputs/") :])
            return
        self.send_error(404)

    def do_HEAD(self):
        if not self.request_allowed():
            self.send_error(403)
            return
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/datasets/"):
            self.serve_file(config.DATASETS_DIR / path[len("/datasets/") :], send_body=False)
            return
        if path.startswith("/outputs/"):
            self.serve_file(config.OUTPUT_DIR / path[len("/outputs/") :], send_body=False)
            return
        self.send_error(404)

    def serve_file(self, path, send_body=True):
        try:
            full = path.resolve()
            allowed = [config.DATASETS_DIR.resolve(), config.OUTPUT_DIR.resolve()]
            if not any(full.is_relative_to(root) for root in allowed):
                self.send_error(403)
                return
            if not full.exists() or not full.is_file():
                self.send_error(404)
                return
            file_size = full.stat().st_size
            ctype = "audio/wav" if full.suffix.lower() == ".wav" else "application/octet-stream"
            start = 0
            end = file_size - 1
            status = 200
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                status = 206
                start_text, _, end_text = range_header[6:].partition("-")
                if start_text:
                    start = int(start_text)
                    if end_text:
                        end = int(end_text)
                elif end_text:
                    # "bytes=-N" asks for the last N bytes.
                    start = max(file_size - int(end_text), 0)
                end = min(end, file_size - 1)
                if start > end:
                    self.send_error(416)
                    return
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()
            if not send_body:
                return
            with full.open("rb") as file:
                file.seek(start)
                remaining = length
                while remaining:
                    chunk = file.read(min(1024 * 256, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # No answer can reach a closed connection. handle() ends the request quietly.
            raise
        except Exception as exc:
            self.send_error(500, str(exc))

    def do_POST(self):
        if not self.request_allowed():
            self.send_error(403)
            return
        path = urllib.parse.urlparse(self.path).path
        if path == "/v1/audio/speech":
            if not self.speech_api_allowed():
                return
            try:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    if not 0 < length <= 1024 * 1024:
                        raise ValueError
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(body, dict):
                        raise ValueError
                except ValueError as exc:
                    raise speech_api.SpeechApiError(400, "The body must be a JSON object of up to 1 MB.") from exc
                audio, content_type = speech_api.speak_for_api(body)
                self.send_audio(audio, content_type)
            except speech_api.SpeechApiError as exc:
                self.send_json(exc.body(), exc.status)
            except Exception as exc:
                jobs.log(f"Speech API error: {exc}")
                self.send_json(speech_api.SpeechApiError(500, str(exc), "server_error").body(), 500)
            return
        if path == "/api/voice-import":
            source = None
            try:
                source = read_upload(self)
                self.send_json({"ok": True, **voices.import_voice(source)})
            except Exception as exc:
                jobs.log(f"Error: {exc}")
                self.send_json({"ok": False, "error": str(exc)}, 500)
            finally:
                if source:
                    source.unlink(missing_ok=True)
            return
        if path == "/api/dataset-cut":
            # The file itself is the body, as a long recording is too large for JSON.
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                source = read_upload(self)
                cutting.start_dataset_cut(query.get("dataset", [""])[0], [(source, query.get("filename", ["recording"])[0])])
                self.send_json({"ok": True})
            except Exception as exc:
                jobs.log(f"Error: {exc}")
                self.send_json({"ok": False, "error": str(exc)}, 500)
            return
        try:
            body = read_json(self)
            if path == "/api/start-system":
                if body.get("auto") and jobs.gpu_busy():
                    self.send_json({"ok": True, "started": False})
                    return
                # The startup worker checks the models once WSL is running; see model_pair.
                system.start_system(body.get("gpt") or "", body.get("sovits") or "", body.get("binding") or "ShiftLeft")
                self.send_json({"ok": True, "started": True})
                return
            if path == "/api/stop-system":
                system.stop_system()
                self.send_json({"ok": True})
                return
            if path == "/api/ptt-binding":
                with system.PTT_CONDITION:
                    system.PTT_STATE["binding"] = body.get("binding") or "ShiftLeft"
                    system.PTT_CONDITION.notify_all()
                self.send_json({"ok": True, "binding": system.PTT_STATE["binding"]})
                return
            if path == "/api/ptt-event":
                with system.PTT_CONDITION:
                    system.PTT_STATE["held"] = bool(body.get("held"))
                    system.PTT_STATE["sequence"] += 1
                    system.PTT_CONDITION.notify_all()
                self.send_json({"ok": True})
                return
            if path == "/api/cancel-generation":
                generation_id = str(body.get("generation_id") or "")
                if generation_id:
                    with jobs.STATE_LOCK:
                        CANCELLED_GENERATIONS.add(generation_id)
                    jobs.log(f"Cancelled Mic to Voice line {generation_id}.")
                self.send_json({"ok": True})
                return
            if path == "/api/create-dataset":
                result = datasets.create_dataset(body.get("name"), body.get("speaker"), body.get("language"))
                self.send_json({"ok": True, "dataset": result})
                return
            if path == "/api/upload-dataset-audio":
                result = datasets.add_dataset_audio(
                    body.get("dataset_id"),
                    body.get("filename"),
                    body.get("audio_base64"),
                    body.get("text"),
                    body.get("speaker"),
                    body.get("language"),
                )
                self.send_json({"ok": True, "dataset": result})
                return
            if path == "/api/voice-export":
                file, name = voices.export_voice(body.get("gpt"), body.get("sovits"), body.get("dataset_id"), body.get("reference"),
                                          body.get("prompt_text"), body.get("extra_references"))
                self.send_json({"ok": True, "url": f"/api/voice-download/{file.name}?name={urllib.parse.quote(name)}", "name": name})
                return
            if path == "/api/compare-epochs":
                compare.start_compare(body.get("model"), body.get("kind"), body.get("dataset_id"), body.get("reference"), body.get("text"),
                              body.get("version") or "v2Pro")
                self.send_json({"ok": True})
                return
            if path == "/api/dataset-check":
                clip_check.start_dataset_check(body.get("dataset_id"))
                self.send_json({"ok": True})
                return
            if path == "/api/dataset-cut-restored":
                cutting.cut_restored(body.get("dataset_id"), body.get("item_ids"))
                self.send_json({"ok": True})
                return
            if path == "/api/dataset-clip-text":
                self.send_json({"ok": True, "text": datasets.update_dataset_clip(body.get("dataset_id"), body.get("wav"), body.get("text"))})
                return
            if path == "/api/dataset-clip-delete":
                datasets.delete_dataset_clip(body.get("dataset_id"), body.get("wav"))
                self.send_json({"ok": True})
                return
            if path == "/api/dataset-transcribe":
                text = asr.transcribe_dataset_upload(body.get("dataset_id"), body.get("filename"), body.get("audio_base64"))
                self.send_json({"ok": True, "text": text})
                return
            if path == "/api/restore-upload":
                self.send_json({"ok": True, "item": restore.add_restore_audio(body.get("filename"), body.get("audio_base64"))})
                return
            if path == "/api/restore-run":
                self.send_json({"ok": True, "results": restore.start_restore(body.get("item_ids"), body.get("settings"))})
                return
            if path == "/api/restore-delete":
                restore.delete_restore_item(str(body.get("item_id") or ""))
                self.send_json({"ok": True})
                return
            if path == "/api/sync-dataset":
                training.sync_dataset_job(body.get("dataset_id") or "")
                self.send_json({"ok": True})
                return
            if path == "/api/prepare-dataset":
                with jobs.gpu_reserved():
                    training.prepare_tts_dataset_job(
                        body.get("dataset_id") or "",
                        body.get("model_name") or "voice-model",
                    )
                self.send_json({"ok": True})
                return
            if path == "/api/start-api":
                gpt, sovits = models.model_pair(body.get("gpt"), body.get("sovits"))
                if not gpt or not sovits:
                    self.send_json({"ok": False, "error": "No GPT/SoVITS model pair found."}, 400)
                    return
                engine.start_api(gpt, sovits)
                self.send_json({"ok": True})
                return
            if path == "/api/stop-api":
                engine.stop_api()
                self.send_json({"ok": True})
                return
            if path == "/api/start-relay-asr":
                asr.start_relay_asr()
                self.send_json({"ok": True})
                return
            if path == "/api/transcribe":
                asr.ensure_relay_asr()
                req = urllib.request.Request(
                    config.RELAY_ASR_URL + "/transcribe",
                    data=json.dumps({"audio_base64": body.get("audio_base64", "")}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = resp.read().decode("utf-8", errors="replace")
                result = json.loads(data)
                self.send_json(result, 200 if result.get("ok") else 500)
                return
            if path == "/api/start-train":
                with jobs.gpu_reserved():
                    training.start_train(
                        body.get("kind"),
                        int(body.get("epochs", 15)),
                        int(body.get("batch_size", 1)),
                        int(body.get("save_every", 5)),
                        body.get("dataset_id") or "",
                        body.get("model_name") or "voice-model",
                        body.get("version") or "v2Pro",
                    )
                self.send_json({"ok": True})
                return
            if path == "/api/stop-job":
                key = body.get("key")
                proc = jobs.JOBS.get(key)
                if proc:
                    proc.stop()
                elif key in wsl.WSL_STOP_SCRIPTS or key == "restore":
                    # Started before the dashboard restarted, so only WSL still knows the job.
                    wsl.stop_in_wsl(wsl.WSL_STOP_SCRIPTS["separate" if key == "restore" else key])
                self.send_json({"ok": True})
                return
            if path == "/api/generate":
                if not engine.api_ready():
                    self.send_json({"ok": False, "error": "Inference engine is not ready yet."}, 409)
                    return
                payload = {
                    "text": body["text"],
                    "text_lang": datasets.language_code(body.get("text_lang")),
                    "ref_audio_path": body["ref_audio_path"],
                    "aux_ref_audio_paths": body.get("aux_ref_audio_paths") or [],
                    "prompt_text": body["prompt_text"],
                    "prompt_lang": datasets.language_code(body.get("prompt_lang")),
                    "top_k": int(body.get("top_k", 15)),
                    "top_p": float(body.get("top_p", 0.6)),
                    "temperature": float(body.get("temperature", 0.6)),
                    "text_split_method": body.get("text_split_method", "cut5"),
                    "batch_size": 1,
                    "speed_factor": float(body.get("speed_factor", 1.0)),
                    "fragment_interval": float(body.get("fragment_interval", 0.3)),
                    "seed": int(body.get("seed", -1)),
                    "media_type": "wav",
                    "parallel_infer": True,
                    "repetition_penalty": float(body.get("repetition_penalty", 1.35)),
                    "streaming_mode": int(body.get("streaming_mode", 0)),
                    "overlap_length": int(body.get("overlap_length", 2)),
                    "min_chunk_length": int(body.get("min_chunk_length", 16)),
                }
                generation_id = str(body.get("generation_id") or "")
                if generation_id:
                    with jobs.STATE_LOCK:
                        CANCELLED_GENERATIONS.discard(generation_id)
                req = urllib.request.Request(
                    config.API_URL + "/tts",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                # The Generate page and Relay can ask at the same time. One at a time keeps one
                # request from switching the models while the other is generating.
                with engine.GENERATION_LOCK:
                    started = time.time()
                    engine.set_api_weights(body.get("gpt"), body.get("sovits"))
                    with urllib.request.urlopen(req, timeout=240) as resp:
                        data = resp.read()
                        if "application/json" in resp.headers.get("Content-Type", ""):
                            self.send_json({"ok": False, "error": data.decode("utf-8", errors="replace")}, 400)
                            return
                    if generation_id:
                        with jobs.STATE_LOCK:
                            cancelled = generation_id in CANCELLED_GENERATIONS
                            CANCELLED_GENERATIONS.discard(generation_id)
                        if cancelled:
                            jobs.log(f"Discarded cancelled Mic to Voice line {generation_id}.")
                            self.send_json({"ok": False, "cancelled": True, "error": "Generation cancelled."}, 409)
                            return
                    if int(payload.get("streaming_mode", 0)):
                        data = engine.repair_streamed_wav(data)
                    output = engine.new_output_path(config.RELAY_OUTPUT_DIR if body.get("relay") else config.OUTPUT_DIR, body.get("voice"))
                    output.write_bytes(data)
                if body.get("relay"):
                    engine.prune_relay_outputs()
                else:
                    speech_api.save_speech_settings(body.get("voice"), speech_api.speech_settings_from_generation(body))
                jobs.log(f"Generated {output.name} in {time.time() - started:.1f}s.")
                url = "/outputs/" + output.relative_to(config.OUTPUT_DIR).as_posix()
                self.send_json({"ok": True, "url": url, "name": output.name})
                return
            self.send_error(404)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.send_json({"ok": False, "error": detail or str(exc)}, exc.code)
        except Exception as exc:
            jobs.log(f"Error: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, 500)


def main():
    port = config.DASHBOARD_PORT
    # Launch-Voice-Dashboard.ps1 writes the output to a file, which Python would otherwise write
    # in the Windows code page, failing on names in other alphabets.
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    jobs.log(f"Local Voice UI starting on http://localhost:{port}")
    threading.Thread(target=restore.kill_orphan_restore_workers, daemon=True).start()
    threading.Thread(target=clean_old_temp_files, daemon=True).start()
    if config.CONFIG_ERROR:
        jobs.log(config.CONFIG_ERROR)
    server = DashboardServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for key in ("ptt-helper", "relay-asr", "restore", "dataset-cut", "compare"):
            proc = jobs.JOBS.get(key)
            if proc and proc.running():
                proc.stop()
        if engine.API_PROCESS and engine.API_PROCESS.process and engine.API_PROCESS.process.poll() is None:
            engine.stop_api()
