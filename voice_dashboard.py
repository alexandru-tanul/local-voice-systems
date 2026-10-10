import json
import base64
import contextlib
import ctypes
import hashlib
import io
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def load_config(path):
    """Read config.json. Returns the settings and an error message for the page, or "" when there is none."""
    if not path.exists():
        return {}, ""
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return {}, (
            f"config.json could not be read ({exc}), so the dashboard is using default paths. "
            "In Windows paths, write \\\\ or / instead of \\."
        )
    if not isinstance(config, dict):
        return {}, "config.json must contain one JSON object, so the dashboard is using default paths."
    return config, ""


def config_port(config, key, default):
    """A port from config.json, and an error message for the page when the value is not a port."""
    value = config.get(key, default)
    try:
        port = int(value)
    except (TypeError, ValueError):
        port = 0
    if isinstance(value, bool) or not 1 <= port <= 65535:
        return default, f"config.json: {key} must be a port number from 1 to 65535, so {default} is used."
    return port, ""


APP_CONFIG, CONFIG_ERROR = load_config(CONFIG_PATH)
DASHBOARD_PORT, _port_error = config_port(APP_CONFIG, "dashboard_port", 8790)
RELAY_ASR_PORT, _asr_port_error = config_port(APP_CONFIG, "asr_port", 8792)
TTS_PORT, _tts_port_error = config_port(APP_CONFIG, "tts_port", 9880)
CONFIG_ERROR = " ".join(filter(None, [CONFIG_ERROR, _port_error, _asr_port_error, _tts_port_error]))
# For running a second copy beside the installed one, for example while testing.
if os.environ.get("VOICE_DASHBOARD_PORT"):
    DASHBOARD_PORT = int(os.environ["VOICE_DASHBOARD_PORT"])
DATASETS_DIR = ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(APP_CONFIG.get("cache_dir") or ROOT / ".cache")
TMP_DIR = Path(APP_CONFIG.get("tmp_dir") or CACHE_DIR / "tmp")

WSL_DISTRO = APP_CONFIG.get("wsl_distro") or "Ubuntu-24.04"
WSL_GSV_ROOT = APP_CONFIG.get("wsl_gpt_sovits_root") or "/root/GPT-SoVITS"
WSL_PYTHON = APP_CONFIG.get("wsl_python") or "/root/gsv-venv/bin/python"
WSL_DATASETS_ROOT = APP_CONFIG.get("wsl_datasets_root") or "/root/datasets"
WSL_GSV_WIN = Path(f"\\\\wsl.localhost\\{WSL_DISTRO}{WSL_GSV_ROOT.replace('/', os.sep)}")
# 127.0.0.1, not localhost: Windows tries localhost as ::1 first, where the forwarded WSL port is
# not listening, and each request then waited about 2 seconds before it fell back to IPv4.
API_URL = f"http://127.0.0.1:{TTS_PORT}"
RELAY_ASR_URL = f"http://127.0.0.1:{RELAY_ASR_PORT}"
RELAY_ASR_PYTHON = ROOT / "relay_env" / "Scripts" / "python.exe"
RELAY_ASR_MODEL = APP_CONFIG.get("asr_model") or "base.en"
RELAY_ASR_MODEL_DIR = Path(APP_CONFIG.get("asr_model_dir") or CACHE_DIR / "faster-whisper")
RELAY_ASR_TMP = OUTPUT_DIR / "relay_asr_tmp"
# Relay lines are played once, so only the newest are kept. Generate page output is kept.
RELAY_OUTPUT_DIR = OUTPUT_DIR / "relay"
RELAY_OUTPUT_LIMIT = 200
# pkill patterns for jobs that run inside WSL. They also work after a dashboard restart, when the
# job is no longer known. The brackets keep a pattern from matching the stop command's own shell.
WSL_STOP_SCRIPTS = {
    "dataset-sync": "pkill -f '[d]ataset-sync; ' || true",
    "dataset-prepare": "pkill -f '[G]PT_SoVITS/prepare_datasets/' || true",
    "train-gpt": "pkill -f '[G]PT_SoVITS/s1_train.py' || true",
    "train-sovits": "pkill -f '[G]PT_SoVITS/s2_train.py' || true",
    "api": f"pkill -f '[a]pi_v2.py.*-p {TTS_PORT} ' || true",
}
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 10.0
LANGUAGES = {"en", "zh", "ja", "ko", "yue"}
MODEL_LIST = {"checked_at": 0.0, "listed": False, "models": {"gpt": [], "sovits": []}}

APP_LOG = []
JOBS = {}
API_PROCESS = None
API_GPT_PATH = None
API_SOVITS_PATH = None
STATE_LOCK = threading.Lock()
# Held while a startup step checks that its run is still current and starts a service, so
# Stop Everything cannot fall between the check and the start.
SYSTEM_LOCK = threading.Lock()
# The voice engine handles one generation at a time, including its model switch.
GENERATION_LOCK = threading.Lock()
ENGINE_LOCK = threading.Lock()
PTT_CONDITION = threading.Condition()
PTT_STATE = {
    "binding": "ShiftLeft",
    "held": False,
    "sequence": 0,
}
SYSTEM_STATE = {
    "status": "stopped",
    "phase": "stopped",
    "message": "System is stopped.",
    "error": "",
}
SYSTEM_THREAD = None
SYSTEM_RUN_ID = 0
CANCELLED_GENERATIONS = set()


def log(message):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    with STATE_LOCK:
        APP_LOG.append(line)
        del APP_LOG[:-500]
    print(line, flush=True)


def set_system_state(status, phase, message, error=""):
    with STATE_LOCK:
        SYSTEM_STATE.update(
            {
                "status": status,
                "phase": phase,
                "message": message,
                "error": error,
            }
        )
    log(message)


def shquote(value):
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def wsl_command(script):
    python_dir = WSL_PYTHON.rsplit("/", 1)[0]
    lib_dir = python_dir.rsplit("/", 1)[0] + "/lib"
    script = (
        f'export PATH={shquote(python_dir)}:"$PATH":/usr/lib/wsl/lib; '
        f'export LD_LIBRARY_PATH={shquote(lib_dir)}"${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"; '
        + script
    )
    # --exec hands the script to bash unchanged. With "--", wsl.exe first runs the command line
    # through the default shell, which expands the script's own variables before bash sets them.
    return ["wsl", "-d", WSL_DISTRO, "--exec", "bash", "-lc", script]


def run_wsl(script, **kwargs):
    return subprocess.run(
        wsl_command(script),
        text=True,
        capture_output=True,
        **kwargs,
    )


def stop_in_wsl(script):
    """Run a pkill stop script in WSL. A stopped distro runs nothing to stop, and starting it
    only for this took longer than the timeout, so it is skipped."""
    if not wsl_distro_running():
        return
    try:
        run_wsl(script, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"Could not run the stop command in WSL: {exc}")


def voice_runtime():
    device = APP_CONFIG.get("tts_device", "cuda")
    precision = APP_CONFIG.get("tts_precision", "fp32")
    if device not in {"cuda", "cpu"} or precision not in {"fp16", "fp32"}:
        raise ValueError("Set tts_device to cuda or cpu, and tts_precision to fp16 or fp32.")
    return {
        "device": device,
        "is_half": device == "cuda" and precision == "fp16",
        "gpu_index": "0" if device == "cuda" else "",
    }


def voice_environment():
    runtime = voice_runtime()
    return (
        f"export CUDA_VISIBLE_DEVICES={shquote(runtime['gpu_index'])}; "
        f"export _CUDA_VISIBLE_DEVICES={shquote(runtime['gpu_index'])}; "
        f"export is_half={runtime['is_half']}; "
    )


class ManagedProcess:
    def __init__(self, name, command, stop_script=None, cwd=ROOT, env=None):
        self.name = name
        self.command = command
        self.stop_script = stop_script
        self.cwd = cwd
        self.env = env
        self.process = None
        self.returncode = None
        self.lines = []
        self.status = "idle"
        self.reader = None

    def start(self):
        if self.process and self.process.poll() is None:
            return False
        self.returncode = None
        self.status = "running"
        self.lines = []
        self.process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=self.env,
        )
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()
        threading.Thread(target=self._wait, daemon=True).start()
        return True

    def last_lines(self, count):
        # Output can still be in the pipe when the process ends, so read it first.
        if self.reader and not self.running():
            self.reader.join(timeout=1)
        return self.lines[-count:]

    def _read_output(self):
        if not self.process or not self.process.stdout:
            return
        for line in self.process.stdout:
            line = line.rstrip()
            with STATE_LOCK:
                self.lines.append(line)
                del self.lines[:-1200]

    def _wait(self):
        if not self.process:
            return
        self.returncode = self.process.wait()
        if self.status == "stopping":
            self.status = "stopped"
            return
        self.status = "done" if self.returncode == 0 else "failed"
        log(f"{self.name} {self.status} with exit code {self.returncode}.")

    def running(self):
        return self.process is not None and self.process.poll() is None

    def stop(self):
        if self.stop_script:
            stop_in_wsl(self.stop_script)
        if not self.running():
            # Already ended: keep "done" or "failed" instead of reporting "stopped".
            return
        self.status = "stopping"
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self.status = "stopped"
        log(f"{self.name} stopped.")

    def snapshot(self):
        running = self.process is not None and self.process.poll() is None
        return {
            "name": self.name,
            "status": "running" if running else self.status,
            "returncode": self.returncode,
            "lines": self.lines[-250:],
        }


def parse_references(dataset_id, valid_only=True):
    refs = []
    if not dataset_id:
        return refs
    try:
        descriptor = dataset_descriptor(dataset_id)
    except Exception:
        return refs
    list_path = descriptor["list_path"]
    if not list_path.exists():
        return refs
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("|", 3)
        if len(parts) != 4:
            continue
        rel_path, speaker, lang, text = parts
        stem = Path(rel_path).stem
        wav_path = descriptor["root"] / rel_path.replace("/", os.sep)
        duration = None
        try:
            with wave.open(str(wav_path), "rb") as wav:
                duration = round(wav.getnframes() / float(wav.getframerate()), 1)
        except Exception:
            pass
        valid_reference = duration is not None and MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS
        valid_aux_reference = duration is not None and duration <= MAX_REFERENCE_SECONDS
        if valid_only and not valid_reference:
            continue
        refs.append(
            {
                "id": stem,
                "label": stem.replace("_", " "),
                "text": text,
                "duration": duration,
                "valid_reference": valid_reference,
                "valid_aux_reference": valid_aux_reference,
                "lang": lang,
                "speaker": speaker,
                "local_url": f"/datasets/{descriptor['id']}/" + rel_path.replace("\\", "/"),
                "wsl_path": f"{descriptor['wsl_root']}/{rel_path.replace('\\', '/')}",
            }
        )
    return refs


def model_files():
    def collect(folder, pattern):
        base = WSL_GSV_WIN / folder
        rows = []
        try:
            for path in base.glob(pattern):
                if path.is_file():
                    rows.append(
                        {
                            "name": path.name,
                            "path": f"{folder}/{path.name}",
                            "mtime": path.stat().st_mtime,
                            "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                        }
                    )
        except OSError:
            pass
        rows.sort(key=lambda row: row["mtime"], reverse=True)
        return rows

    models = {
        "gpt": collect("GPT_weights_v2Pro", "*.ckpt") + collect("GPT_SoVITS/pretrained_models", "s1v3.ckpt"),
        "sovits": collect("SoVITS_weights_v2Pro", "*.pth") + collect("GPT_SoVITS/pretrained_models/v2Pro", "s2Gv2Pro.pth"),
    }
    MODEL_LIST.update(models=models, checked_at=time.time(), listed=True)
    return models


def wsl_distro_running():
    try:
        result = subprocess.run(
            ["wsl", "--list", "--running", "--quiet"],
            capture_output=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    # wsl.exe prints UTF-16 unless WSL_UTF8 is set.
    text = result.stdout.decode("utf-16-le" if b"\x00" in result.stdout else "utf-8", errors="ignore")
    return WSL_DISTRO.lower() in {line.strip().lower() for line in text.splitlines()}


def polled_model_files(max_age=15):
    """Model list for the page's 5-second poll.

    It rescans at most every max_age seconds, and only while the distro runs, because reading
    \\\\wsl.localhost starts a stopped distro. Starting an engine or a check rescans at once."""
    if time.time() - MODEL_LIST["checked_at"] >= max_age:
        MODEL_LIST["checked_at"] = time.time()
        if wsl_distro_running():
            model_files()
    return MODEL_LIST["models"]


def slugify(value, fallback="dataset"):
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return (slug or fallback)[:64]


def language_code(value):
    return value if value in LANGUAGES else "en"


def dataset_id_for_name(name):
    """Folder id for a dataset name. A name without Latin letters or digits gets a stable id from its hash."""
    clean = str(name or "").strip()
    return slugify(clean, "") or "dataset-" + hashlib.sha1(clean.encode("utf-8")).hexdigest()[:8]


def dataset_descriptor(dataset_id):
    dataset_id = slugify(dataset_id)
    root = (DATASETS_DIR / dataset_id).resolve()
    if root.parent != DATASETS_DIR.resolve() or not root.exists():
        raise RuntimeError("The selected dataset does not exist.")
    manifest_path = root / "dataset.json"
    if not manifest_path.exists():
        raise RuntimeError("The selected dataset is missing dataset.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "id": dataset_id,
        "name": manifest.get("name") or dataset_id,
        "root": root,
        "list_path": root / f"{dataset_id}.list",
        "wsl_root": f"{WSL_DATASETS_ROOT}/{dataset_id}",
        "entries": manifest.get("entries") or [],
    }


def dataset_summary(descriptor):
    # dataset.json lists the clips. Other WAV files in the folder are not part of the dataset.
    root = descriptor["root"]
    entries = descriptor["entries"]
    list_path = descriptor["list_path"]
    rows = []
    if list_path.exists():
        rows = [line for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    wav_count = sum((root / "wavs" / str(entry.get("wav", ""))).is_file() for entry in entries)
    return {
        "id": descriptor["id"],
        "name": descriptor["name"],
        "wav_count": wav_count,
        "list_rows": len(rows),
        "ready": bool(entries) and wav_count == len(entries) == len(rows),
        "path": str(root),
        "wsl_root": descriptor["wsl_root"],
    }


def list_datasets():
    rows = []
    for manifest_path in sorted(DATASETS_DIR.glob("*/dataset.json")):
        try:
            rows.append(dataset_summary(dataset_descriptor(manifest_path.parent.name)))
        except Exception as exc:
            log(f"Could not read dataset {manifest_path.parent.name}: {exc}")
    return rows


def create_dataset(name, speaker="speaker", language="en"):
    clean_name = str(name or "").strip()
    if not clean_name:
        raise RuntimeError("Enter a dataset name.")
    dataset_id = dataset_id_for_name(clean_name)
    root = DATASETS_DIR / dataset_id
    if root.exists():
        raise RuntimeError(f"A dataset with the id {dataset_id} already exists. Choose another name.")
    (root / "wavs").mkdir(parents=True)
    manifest = {
        "id": dataset_id,
        "name": clean_name,
        "default_speaker": slugify(speaker, "speaker"),
        "default_language": language_code(language),
        "created_at": time.time(),
        "entries": [],
    }
    write_text_atomic(root / "dataset.json", json.dumps(manifest, indent=2))
    write_text_atomic(root / f"{dataset_id}.list", "")
    write_text_atomic(root / "metadata.csv", "")
    log(f"Created dataset {clean_name} ({dataset_id}).")
    return dataset_summary(dataset_descriptor(dataset_id))


def write_text_atomic(path, text):
    # Readers see the old file or the new one, never a half-written one. LF line endings keep
    # carriage returns out of the transcripts that GPT-SoVITS reads in WSL.
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)


@contextlib.contextmanager
def dataset_lock(root):
    """Let one change to a dataset run at a time, also across processes, such as an
    import_dataset.py run during an upload from the page."""
    with open(root / ".lock", "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def clean_transcript(text):
    # One line per clip: "|" separates the .list columns, and a line break would start a new row.
    return " ".join(str(text or "").replace("|", " ").split())


def write_dataset_manifest(root, manifest):
    dataset_id = slugify(manifest.get("id") or root.name)
    entries = manifest.get("entries") or []
    list_rows = []
    metadata_rows = []
    for entry in entries:
        wav_name = entry["wav"]
        text = clean_transcript(entry["text"])
        speaker = slugify(entry.get("speaker"), "speaker")
        language = language_code(entry.get("language"))
        list_rows.append(f"wavs/{wav_name}|{speaker}|{language}|{text}")
        metadata_rows.append(f"{Path(wav_name).stem}|{text}|{text}")
    ending = "\n" if list_rows else ""
    write_text_atomic(root / f"{dataset_id}.list", "\n".join(list_rows) + ending)
    write_text_atomic(root / "metadata.csv", "\n".join(metadata_rows) + ending)
    write_text_atomic(root / "dataset.json", json.dumps(manifest, indent=2))


def convert_audio(source, target):
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-ar", "44100", "-ac", "1", "-sample_fmt", "s16", str(target)],
        text=True,
        capture_output=True,
        timeout=180,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg could not convert this audio file.")


def add_dataset_audio(dataset_id, filename, audio_base64, text, speaker="", language="en"):
    descriptor = dataset_descriptor(dataset_id)
    transcript = clean_transcript(text)
    if not transcript:
        raise RuntimeError("Every audio file needs a transcript.")
    try:
        audio = base64.b64decode(audio_base64, validate=True)
    except Exception as exc:
        raise RuntimeError("The uploaded audio data is invalid.") from exc
    if not audio or len(audio) > 100 * 1024 * 1024:
        raise RuntimeError("Audio files must be between 1 byte and 100 MB.")
    root = descriptor["root"]
    filename = str(filename or "clip.audio")
    stem = slugify(Path(filename).stem, "clip")
    suffix = Path(filename).suffix.lower() or ".audio"
    fd, source_name = tempfile.mkstemp(prefix="upload-", suffix=suffix, dir=str(OUTPUT_DIR))
    os.close(fd)
    source = Path(source_name)
    converted = source.with_name(source.stem + "-converted.wav")
    try:
        # The slow conversion runs before the lock, and a failed one leaves nothing in wavs/.
        source.write_bytes(audio)
        convert_audio(source, converted)
        with dataset_lock(root):
            manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
            taken = {entry.get("wav") for entry in manifest.get("entries", [])}
            wav_name = f"{stem}.wav"
            counter = 2
            while wav_name in taken or (root / "wavs" / wav_name).exists():
                wav_name = f"{stem}-{counter}.wav"
                counter += 1
            os.replace(converted, root / "wavs" / wav_name)
            manifest.setdefault("entries", []).append(
                {
                    "wav": wav_name,
                    "source_name": Path(filename).name,
                    "text": transcript,
                    "speaker": speaker or manifest.get("default_speaker") or "speaker",
                    "language": language or manifest.get("default_language") or "en",
                }
            )
            write_dataset_manifest(root, manifest)
    finally:
        for path in (source, converted):
            try:
                path.unlink()
            except OSError:
                pass
    log(f"Added {wav_name} to dataset {descriptor['name']}.")
    return dataset_summary(dataset_descriptor(dataset_id))


class TcpInitialRtoParameters(ctypes.Structure):
    _fields_ = [("Rtt", ctypes.c_ushort), ("MaxSynRetransmissions", ctypes.c_ubyte)]


def port_open(port, timeout=0.2):
    """True when something accepts connections on 127.0.0.1:port.

    Windows retries a refused connection until the timeout, so each check of a stopped service
    took the full timeout. With SYN retransmissions turned off, a refusal returns at once."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name == "nt":
            # SIO_TCP_INITIAL_RTO with TCP_INITIAL_RTO_UNSPECIFIED_RTT and
            # TCP_INITIAL_RTO_NO_SYN_RETRANSMISSIONS, from mstcpip.h. If the call fails, the
            # connect still works and only waits longer.
            params = TcpInitialRtoParameters(0xFFFF, 0xFE)
            ctypes.windll.ws2_32.WSAIoctl(
                ctypes.c_size_t(sock.fileno()), ctypes.c_ulong(0x98000011),
                ctypes.byref(params), ctypes.c_ulong(ctypes.sizeof(params)),
                None, ctypes.c_ulong(0), ctypes.byref(ctypes.c_ulong()), None, None,
            )
        sock.settimeout(timeout)
        try:
            sock.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def api_ready():
    return port_open(TTS_PORT)


def relay_asr_ready():
    return port_open(RELAY_ASR_PORT)


def start_relay_asr():
    if relay_asr_ready():
        log("Relay ASR is already running.")
        return True
    if not RELAY_ASR_PYTHON.exists():
        raise RuntimeError("Relay ASR environment is missing.")
    RELAY_ASR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RELAY_ASR_TMP.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    relay_env = os.environ.copy()
    relay_env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": str(CACHE_DIR / "huggingface"),
            "HF_HUB_CACHE": str(CACHE_DIR / "huggingface" / "hub"),
            "PIP_CACHE_DIR": str(CACHE_DIR / "pip"),
            "TEMP": str(TMP_DIR),
            "TMP": str(TMP_DIR),
            "RELAY_ASR_MODEL_DIR": str(RELAY_ASR_MODEL_DIR),
            "RELAY_ASR_TMP": str(RELAY_ASR_TMP),
            "RELAY_ASR_MODEL": RELAY_ASR_MODEL,
            "RELAY_ASR_PORT": str(RELAY_ASR_PORT),
        }
    )
    proc = ManagedProcess(
        "Relay ASR",
        [str(RELAY_ASR_PYTHON), str(ROOT / "relay_asr_server.py")],
        cwd=ROOT,
        env=relay_env,
    )
    with STATE_LOCK:
        current = JOBS.get("relay-asr")
        if current and current.running():
            # Started, but not listening yet. The caller waits for the port.
            return True
        JOBS["relay-asr"] = proc
    proc.start()
    log(f"Relay ASR starting on {RELAY_ASR_URL}.")
    return True


def warm_up_relay_asr():
    # The server answers a failed model load with JSON that names the cause.
    try:
        with urllib.request.urlopen(RELAY_ASR_URL + "/warmup", timeout=180) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
    try:
        warmup = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        warmup = {}
    if not warmup.get("ok"):
        raise RuntimeError(warmup.get("error") or "Speech recognition model failed to load.")


def ptt_helper_ready():
    proc = JOBS.get("ptt-helper")
    return bool(proc and proc.process and proc.process.poll() is None)


def start_ptt_helper():
    if ptt_helper_ready():
        log("Global PTT helper is already running.")
        return True
    helper_path = ROOT / "ptt_helper.py"
    if not helper_path.exists():
        raise RuntimeError("Global PTT helper is missing.")
    proc = ManagedProcess(
        "Global PTT helper",
        [sys.executable, str(helper_path), "--dashboard", f"http://127.0.0.1:{DASHBOARD_PORT}"],
        cwd=ROOT,
    )
    with STATE_LOCK:
        JOBS["ptt-helper"] = proc
    proc.start()
    log("Global PTT helper started.")
    return True


def wait_until(check, timeout, run_id, description, proc=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if run_id != SYSTEM_RUN_ID:
            raise RuntimeError("Startup was cancelled.")
        if check():
            return
        if proc and proc.process and proc.process.poll() is not None:
            # The process ended before it became ready. Its last lines usually name the cause.
            output = "\n".join(proc.last_lines(6))
            raise RuntimeError(
                f"{description} exited with code {proc.process.poll()} before it became ready."
                + (f"\n{output}" if output else "")
            )
        time.sleep(0.5)
    raise RuntimeError(f"{description} did not become ready within {int(timeout)} seconds.")


def startup_step(run_id, phase, message):
    # Call with SYSTEM_LOCK held, so Stop Everything cannot cancel the run between this
    # check and the service the step starts.
    if run_id != SYSTEM_RUN_ID:
        raise RuntimeError("Startup was cancelled.")
    set_system_state("starting", phase, message)


def _start_system_worker(gpt_path, sovits_path, run_id):
    try:
        result = run_wsl("true", timeout=30)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "WSL failed to start.").replace("\x00", "").strip()
            raise RuntimeError(detail)
        models = model_files()
        gpt_path = gpt_path or (models["gpt"][0]["path"] if models["gpt"] else "")
        sovits_path = sovits_path or (models["sovits"][0]["path"] if models["sovits"] else "")
        if not gpt_path or not sovits_path:
            raise RuntimeError("No compatible GPT and SoVITS model pair was found.")

        with SYSTEM_LOCK:
            startup_step(run_id, "asr", "Starting local speech recognition...")
            start_relay_asr()
        wait_until(relay_asr_ready, 120, run_id, "Speech recognition", JOBS.get("relay-asr"))
        with SYSTEM_LOCK:
            startup_step(run_id, "asr", "Loading the speech recognition model...")
        warm_up_relay_asr()

        with SYSTEM_LOCK:
            startup_step(run_id, "tts", "Starting the selected voice engine...")
            engine_running = api_ready()
            if not engine_running:
                start_api(gpt_path, sovits_path)
        if engine_running:
            # Only applies the selected models, which can take minutes, so it runs outside the lock.
            start_api(gpt_path, sovits_path)
        wait_until(api_ready, 240, run_id, "Voice engine", API_PROCESS)

        with SYSTEM_LOCK:
            startup_step(run_id, "ptt", "Starting global push-to-talk...")
            start_ptt_helper()
        wait_until(ptt_helper_ready, 15, run_id, "Global push-to-talk")

        with SYSTEM_LOCK:
            if run_id != SYSTEM_RUN_ID:
                raise RuntimeError("Startup was cancelled.")
            set_system_state("ready", "ready", "System ready. Press your push-to-talk key.")
    except Exception as exc:
        with SYSTEM_LOCK:
            if run_id == SYSTEM_RUN_ID:
                set_system_state("failed", "failed", "System startup failed.", str(exc))


def start_system(gpt_path, sovits_path, binding="ShiftLeft"):
    global SYSTEM_THREAD, SYSTEM_RUN_ID
    with SYSTEM_LOCK:
        # A startup that Stop Everything cancelled can still be waiting, but it can no longer
        # change the status or start a service, so only the current run's state matters here.
        if SYSTEM_STATE["status"] == "starting":
            raise RuntimeError("The system is already starting.")
        if SYSTEM_STATE["status"] == "stopping":
            raise RuntimeError("The system is still stopping. Try again in a moment.")
        with PTT_CONDITION:
            PTT_STATE["binding"] = binding or "ShiftLeft"
            PTT_CONDITION.notify_all()
        SYSTEM_RUN_ID += 1
        run_id = SYSTEM_RUN_ID
        set_system_state("starting", "wsl", "Starting the local model environment...")
        SYSTEM_THREAD = threading.Thread(
            target=_start_system_worker,
            args=(gpt_path, sovits_path, run_id),
            daemon=True,
        )
        SYSTEM_THREAD.start()


def run_powershell(script, timeout):
    # -EncodedCommand takes the script as base64 UTF-16, so its quotes and backslashes need no escaping.
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        text=True,
        capture_output=True,
        timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def kill_port_listener(port, label, marker):
    # JOBS only knows processes this dashboard instance launched; a service started
    # by an earlier instance survives a dashboard restart, so stop it by port owner.
    # Only an owner whose command line contains the marker is stopped, so another
    # program that uses the same port is left alone.
    try:
        result = run_powershell(
            f"Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | "
            "ForEach-Object { Get-CimInstance Win32_Process -Filter \"ProcessId=$($_.OwningProcess)\" } | "
            f"Where-Object {{ $_.CommandLine -like '*{marker}*' }} | Select-Object -ExpandProperty ProcessId",
            timeout=10,
        )
        pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}
        for pid in pids:
            if pid <= 0 or pid == os.getpid():
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=6,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            log(f"Stopped orphaned {label} process {pid}.")
    except Exception as exc:
        log(f"Could not stop orphaned {label}: {exc}")


def kill_orphan_ptt_helpers():
    try:
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'ptt_helper\\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }",
            ],
            capture_output=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        log(f"Could not stop orphaned PTT helpers: {exc}")


def stop_system():
    global SYSTEM_RUN_ID
    with SYSTEM_LOCK:
        SYSTEM_RUN_ID += 1
        stop_id = SYSTEM_RUN_ID
        set_system_state("stopping", "stopping", "Stopping managed services...")
    try:
        asr = JOBS.get("relay-asr")
        if asr:
            asr.stop()
        if relay_asr_ready():
            kill_port_listener(RELAY_ASR_PORT, "speech recognition", "relay_asr_server.py")
            if relay_asr_ready():
                log(f"Another program listens on port {RELAY_ASR_PORT}. It was left running.")
        helper = JOBS.get("ptt-helper")
        if helper:
            helper.stop()
        kill_orphan_ptt_helpers()
        stop_api()
    finally:
        with PTT_CONDITION:
            PTT_STATE["held"] = False
            PTT_STATE["sequence"] += 1
            PTT_CONDITION.notify_all()
        with SYSTEM_LOCK:
            # A second Stop that started meanwhile reports the result itself.
            if SYSTEM_RUN_ID == stop_id:
                set_system_state("stopped", "stopped", "System is stopped.")


def system_snapshot():
    snapshot = dict(SYSTEM_STATE)
    snapshot["components"] = {
        "asr": "ready" if relay_asr_ready() else "off",
        "tts": "ready" if api_ready() else "off",
        "ptt": "ready" if ptt_helper_ready() else "off",
    }
    if snapshot["status"] == "ready" and "off" in snapshot["components"].values():
        snapshot["status"] = "degraded"
        snapshot["message"] = "A required service stopped unexpectedly."
    return snapshot


def setup_status():
    models = model_files()
    def safe_exists(path):
        try:
            return path.exists()
        except OSError:
            return False

    gpu = {"status": "missing", "detail": "Voice engine is not installed."}
    try:
        runtime = voice_runtime()
        probe = (
            "import torch; "
            f"x = torch.ones((8, 8), device={runtime['device']!r}, "
            f"dtype=torch.{'float16' if runtime['is_half'] else 'float32'}); "
            "assert (x @ x).isfinite().all().item(), 'Try tts_precision: fp32'; "
            f"print(torch.cuda.get_device_name(0) if {runtime['device'] == 'cuda'} else 'CPU')"
        )
        result = run_wsl(f"{shquote(WSL_PYTHON)} -c {shquote(probe)}", timeout=60)
        if result.returncode:
            # wsl.exe reports its own failures on stdout in UTF-16, so fall back to that.
            message = result.stderr.strip() or result.stdout.replace("\x00", "").strip().split("\n")[0]
            raise RuntimeError(message or "The voice engine check failed.")
        precision = "FP16" if runtime["is_half"] else "FP32"
        gpu = {"status": "ready", "detail": f"{result.stdout.strip()} ({precision})"}
    except Exception as exc:
        gpu["detail"] = str(exc)

    return {
        "config": {
            "status": "needs_configuration" if CONFIG_ERROR else ("ready" if CONFIG_PATH.exists() else "missing"),
            "detail": CONFIG_ERROR or str(CONFIG_PATH),
        },
        "dashboard_python": {
            "status": "ready" if Path(sys.executable).exists() else "missing",
            "detail": sys.executable,
        },
        "wsl_files": {
            "status": "ready" if safe_exists(WSL_GSV_WIN) else "missing",
            "detail": str(WSL_GSV_WIN),
        },
        "gpt_models": {
            "status": "ready" if models["gpt"] else "missing",
            "detail": f"{len(models['gpt'])} model(s) found",
        },
        "sovits_models": {
            "status": "ready" if models["sovits"] else "missing",
            "detail": f"{len(models['sovits'])} model(s) found",
        },
        "asr_environment": {
            "status": "ready" if RELAY_ASR_PYTHON.exists() else "missing",
            "detail": str(RELAY_ASR_PYTHON),
        },
        "asr_model_cache": {
            "status": "ready" if RELAY_ASR_MODEL_DIR.exists() else "needs_configuration",
            "detail": str(RELAY_ASR_MODEL_DIR),
        },
        "ptt_helper": {
            "status": "ready" if (ROOT / "ptt_helper.py").exists() else "missing",
            "detail": str(ROOT / "ptt_helper.py"),
        },
        "gpu": gpu,
        "vb_cable": {
            "status": "needs_browser_check",
            "detail": "Audio devices are checked in the browser after microphone permission is granted.",
        },
    }


def write_tts_config(gpt_path, sovits_path):
    runtime = voice_runtime()
    config = f"""custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: {runtime['device']}
  is_half: {str(runtime['is_half']).lower()}
  t2s_weights_path: {json.dumps(gpt_path)}
  version: v2Pro
  vits_weights_path: {json.dumps(sovits_path)}
"""
    path = WSL_GSV_WIN / "TEMP" / "dashboard_tts_infer.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config, encoding="utf-8")
    return "TEMP/dashboard_tts_infer.yaml"


def write_s1_config(epochs, batch_size, save_every, model_name="voice-model"):
    runtime = voice_runtime()
    model_name = slugify(model_name, "voice-model")
    text = f"""data:
  max_eval_sample: 8
  max_sec: 54
  num_workers: 4
  pad_val: 1024
inference:
  top_k: 15
model:
  EOS: 1024
  dropout: 0
  embedding_dim: 512
  head: 16
  hidden_dim: 512
  linear_units: 2048
  n_layer: 24
  phoneme_vocab_size: 732
  random_bert: 0
  vocab_size: 1025
optimizer:
  decay_steps: 40000
  lr: 0.01
  lr_end: 0.0001
  lr_init: 1.0e-05
  warmup_steps: 2000
output_dir: logs/{model_name}/logs_s1_v2Pro
pretrained_s1: GPT_SoVITS/pretrained_models/s1v3.ckpt
train:
  batch_size: {int(batch_size)}
  epochs: {int(epochs)}
  exp_name: {model_name}
  gradient_clip: 1.0
  half_weights_save_dir: GPT_weights_v2Pro
  if_dpo: false
  if_save_every_weights: true
  if_save_latest: true
  precision: {"16-mixed" if runtime['is_half'] else '"32"'}
  save_every_n_epoch: {int(save_every)}
  seed: 1234
train_phoneme_path: logs/{model_name}/2-name2text.txt
train_semantic_path: logs/{model_name}/6-name2semantic.tsv
"""
    path = WSL_GSV_WIN / "TEMP" / f"{model_name}_s1.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"{WSL_GSV_ROOT}/TEMP/{model_name}_s1.yaml"


def write_s2_config(epochs, batch_size, save_every, model_name="voice-model", include_model_version=True):
    runtime = voice_runtime()
    model_name = slugify(model_name, "voice-model")
    config = {
        "train": {
            "log_interval": 100,
            "eval_interval": 500,
            "seed": 1234,
            "epochs": int(epochs),
            "learning_rate": 0.0001,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": int(batch_size),
            "fp16_run": runtime["is_half"],
            "lr_decay": 0.999875,
            "segment_size": 20480,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
            "text_low_lr_rate": 0.4,
            "grad_ckpt": False,
            "pretrained_s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
            "pretrained_s2D": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth",
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": int(save_every),
            "gpu_numbers": runtime["gpu_index"],
            "lora_rank": "32",
        },
        "data": {
            "max_wav_value": 32768.0,
            "sampling_rate": 32000,
            "filter_length": 2048,
            "hop_length": 640,
            "win_length": 2048,
            "n_mel_channels": 128,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "n_speakers": 300,
            "cleaned_text": True,
            "exp_dir": f"logs/{model_name}",
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 8, 2, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 8, 2, 2],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "gin_channels": 1024,
            "semantic_frame_rate": "25hz",
            "freeze_quantizer": True,
            "version": "v2Pro",
        },
        "s2_ckpt_dir": f"logs/{model_name}",
        "content_module": "cnhubert",
        "save_weight_dir": "SoVITS_weights_v2Pro",
        "name": model_name,
        "version": "v2Pro",
    }
    if not include_model_version:
        config["model"].pop("version", None)
    path = WSL_GSV_WIN / "TEMP" / f"{model_name}_s2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return f"{WSL_GSV_ROOT}/TEMP/{model_name}_s2.json"


def start_api(gpt_path, sovits_path):
    global API_PROCESS, API_GPT_PATH, API_SOVITS_PATH
    # The page and Start System can both start the engine; one check-and-start runs at a time.
    with ENGINE_LOCK:
        if API_PROCESS and API_PROCESS.running():
            log("Inference engine is already running.")
            return True
        if not api_ready():
            config_path = write_tts_config(gpt_path, sovits_path)
            script = (
                voice_environment() + "ulimit -l 2097152; "
                f"cd {shquote(WSL_GSV_ROOT)} && "
                f"{shquote(WSL_PYTHON)} api_v2.py -a 127.0.0.1 -p {TTS_PORT} -c {shquote(config_path)}"
            )
            API_PROCESS = ManagedProcess("inference engine", wsl_command(script), stop_script=WSL_STOP_SCRIPTS["api"])
            API_PROCESS.start()
            API_GPT_PATH = gpt_path
            API_SOVITS_PATH = sovits_path
            log(f"Starting inference engine on {API_URL}.")
            return True
    log("A voice engine is already accepting connections; applying the selected models.")
    set_api_weights(gpt_path, sovits_path)
    return True


def stop_api():
    global API_PROCESS, API_GPT_PATH, API_SOVITS_PATH
    try:
        urllib.request.urlopen(API_URL + "/control?command=exit", timeout=1)
    except Exception:
        pass
    if API_PROCESS:
        API_PROCESS.stop()
    else:
        # An engine left by an earlier dashboard run.
        stop_in_wsl(WSL_STOP_SCRIPTS["api"])
    API_PROCESS = None
    API_GPT_PATH = None
    API_SOVITS_PATH = None


def set_api_weights(gpt_path, sovits_path):
    global API_GPT_PATH, API_SOVITS_PATH
    for endpoint, path, loaded in [
        ("set_gpt_weights", gpt_path, API_GPT_PATH),
        ("set_sovits_weights", sovits_path, API_SOVITS_PATH),
    ]:
        # No path, as when the page has not listed models yet, keeps the loaded weights.
        if not path or path == loaded:
            continue
        started = time.time()
        url = API_URL + "/" + endpoint + "?" + urllib.parse.urlencode({"weights_path": path})
        with urllib.request.urlopen(url, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status >= 400:
                raise RuntimeError(body)
        if endpoint == "set_gpt_weights":
            API_GPT_PATH = path
        else:
            API_SOVITS_PATH = path
        log(f"Loaded {endpoint.replace('set_', '').replace('_weights', '')} weights in {time.time() - started:.1f}s.")


def repair_streamed_wav(data):
    if not data.startswith(b"RIFF"):
        return data
    try:
        with wave.open(io.BytesIO(data), "rb") as src:
            if src.getnframes() > 0:
                return data
            channels = src.getnchannels()
            sample_width = src.getsampwidth()
            sample_rate = src.getframerate()
    except wave.Error:
        return data
    data_marker = data.find(b"data", 12)
    if data_marker < 0:
        return data
    raw_audio = data[data_marker + 8:]
    if not raw_audio:
        return data
    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(sample_width)
        dst.setframerate(sample_rate)
        dst.writeframes(raw_audio)
    return out.getvalue()


def dataset_sync_script(descriptor):
    target = descriptor["wsl_root"]
    list_name = descriptor["list_path"].name
    # "&", "|" and "\" have a meaning in a sed replacement, so they are escaped.
    sed_target = target.replace("\\", "\\\\").replace("&", "\\&").replace("|", "\\|")
    return (
        "set -e; "
        # wslpath maps the Windows folder the way this distro mounts drives.
        f"source_dir=$(wslpath -u {shquote(str(descriptor['root']))}); "
        f"mkdir -p {shquote(target)}/wavs; "
        # -u copies only clips that are new or changed since the last sync.
        f'cp -ru "$source_dir"/wavs/. {shquote(target)}/wavs/; '
        f'cp "$source_dir"/{shquote(list_name)} {shquote(target)}/source.list; '
        f"sed {shquote(f's|^wavs/|{sed_target}/wavs/|')} {shquote(target)}/source.list > {shquote(target)}/training.list; "
        f"echo Dataset synced; wc -l {shquote(target + '/training.list')}"
    )


def sync_dataset_job(dataset_id):
    descriptor = dataset_descriptor(dataset_id)
    summary = dataset_summary(descriptor)
    if not summary["ready"]:
        raise RuntimeError("The selected dataset needs matching audio files and transcript rows before it can be synced.")
    # ": dataset-sync" is a no-op that puts a name for pkill into this shell's command line.
    proc = ManagedProcess(
        f"Sync {descriptor['name']}",
        wsl_command(": dataset-sync; " + dataset_sync_script(descriptor)),
        stop_script=WSL_STOP_SCRIPTS["dataset-sync"],
    )
    with STATE_LOCK:
        current = JOBS.get("dataset-sync")
        if current and current.process and current.process.poll() is None:
            raise RuntimeError("A dataset sync is already running.")
        JOBS["dataset-sync"] = proc
    proc.start()
    log(f"Syncing dataset {descriptor['name']} to WSL.")


def prepare_tts_dataset_job(dataset_id, model_name):
    descriptor = dataset_descriptor(dataset_id)
    summary = dataset_summary(descriptor)
    if not summary["ready"]:
        raise RuntimeError("The selected dataset is incomplete. Add matching audio and transcripts first.")
    model_name = slugify(model_name, "voice-model")
    s2_config = write_s2_config(1, 1, 1, model_name, include_model_version=False)
    target = descriptor["wsl_root"]
    opt_dir = f"{WSL_GSV_ROOT}/logs/{model_name}"
    common = (
        f"export inp_text={shquote(target + '/training.list')}; "
        f"export inp_wav_dir={shquote(target + '/wavs')}; "
        f"export exp_name={shquote(model_name)}; export i_part=0; export all_parts=1; "
        + voice_environment() + "export version=v2Pro; "
        f"export opt_dir={shquote(opt_dir)}; "
        f"export bert_pretrained_dir={shquote(WSL_GSV_ROOT + '/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large')}; "
        f"export cnhubert_base_dir={shquote(WSL_GSV_ROOT + '/GPT_SoVITS/pretrained_models/chinese-hubert-base')}; "
        f"export pretrained_s2G={shquote(WSL_GSV_ROOT + '/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth')}; "
        f"export sv_path={shquote(WSL_GSV_ROOT + '/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt')}; "
        f"export s2config_path={shquote(s2_config)}; "
    )
    script = (
        "set -e; "
        + dataset_sync_script(descriptor)
        + f"; mkdir -p {shquote(opt_dir)}; "
        + f"rm -f {shquote(opt_dir)}/2-name2text-0.txt {shquote(opt_dir)}/6-name2semantic-0.tsv; "
        + common
        + f"cd {shquote(WSL_GSV_ROOT)}; "
        + f"{shquote(WSL_PYTHON)} -s GPT_SoVITS/prepare_datasets/1-get-text.py; "
        + f"cp {shquote(opt_dir)}/2-name2text-0.txt {shquote(opt_dir)}/2-name2text.txt; "
        + f"{shquote(WSL_PYTHON)} -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py; "
        + f"{shquote(WSL_PYTHON)} -s GPT_SoVITS/prepare_datasets/2-get-sv.py; "
        + f"{shquote(WSL_PYTHON)} -s GPT_SoVITS/prepare_datasets/3-get-semantic.py; "
        + f"printf 'item_name\\tsemantic_audio\\n' > {shquote(opt_dir)}/6-name2semantic.tsv; "
        + f"cat {shquote(opt_dir)}/6-name2semantic-0.tsv >> {shquote(opt_dir)}/6-name2semantic.tsv; "
        + f"echo Training features ready for {shquote(model_name)}."
    )
    proc = ManagedProcess(
        f"Prepare {descriptor['name']} for {model_name}",
        wsl_command(script),
        # Matches this job's shell, whose command line names the scripts, and each running script.
        stop_script=WSL_STOP_SCRIPTS["dataset-prepare"],
    )
    with STATE_LOCK:
        current = JOBS.get("dataset-prepare")
        if current and current.process and current.process.poll() is None:
            raise RuntimeError("Dataset preparation is already running.")
        JOBS["dataset-prepare"] = proc
    proc.start()
    log(f"Preparing dataset {descriptor['name']} for model {model_name}.")


def start_train(kind, epochs, batch_size, save_every, dataset_id="", model_name="voice-model"):
    descriptor = dataset_descriptor(dataset_id)
    model_name = slugify(model_name, "voice-model")
    feature_root = f"{WSL_GSV_ROOT}/logs/{model_name}"
    check = run_wsl(
        f"test -s {shquote(feature_root + '/2-name2text.txt')} -a -s {shquote(feature_root + '/6-name2semantic.tsv')}",
        timeout=15,
    )
    if check.returncode != 0:
        raise RuntimeError("Training features are missing. Run Sync & Prepare before starting training.")
    if kind == "gpt":
        config_path = write_s1_config(epochs, batch_size, save_every, model_name)
        script = (
            voice_environment() + "ulimit -l 2097152; "
            f"cd {shquote(WSL_GSV_ROOT)} && "
            f"{shquote(WSL_PYTHON)} -s GPT_SoVITS/s1_train.py --config_file {shquote(config_path)}"
        )
        name = f"GPT training: {model_name}"
    elif kind == "sovits":
        config_path = write_s2_config(epochs, batch_size, save_every, model_name)
        script = (
            voice_environment() + "ulimit -l 2097152; "
            f"cd {shquote(WSL_GSV_ROOT)} && "
            f"{shquote(WSL_PYTHON)} -s GPT_SoVITS/s2_train.py --config {shquote(config_path)}"
        )
        name = f"SoVITS training: {model_name}"
    else:
        raise ValueError("Unknown training kind.")
    key = f"train-{kind}"
    proc = ManagedProcess(name, wsl_command(script), stop_script=WSL_STOP_SCRIPTS[key])
    with STATE_LOCK:
        current = JOBS.get(key)
        if current and current.process and current.process.poll() is None:
            raise RuntimeError(f"{name} is already running.")
        JOBS[key] = proc
    proc.start()
    log(f"{name} started from {descriptor['name']}: epochs={epochs}, batch={batch_size}, save_every={save_every}.")


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
    with STATE_LOCK:
        chunks.extend(APP_LOG[-80:])
        if API_PROCESS:
            chunks.append("")
            chunks.append(f"--- {API_PROCESS.name}: {API_PROCESS.status} ---")
            chunks.extend(API_PROCESS.lines[-180:])
    return "\n".join(chunks)


def new_output_path(folder, voice):
    # Call with GENERATION_LOCK held, so no other generation takes the same name meanwhile.
    folder.mkdir(parents=True, exist_ok=True)
    stem = slugify(voice, "voice") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = folder / f"{stem}.wav"
    counter = 2
    while path.exists():
        path = folder / f"{stem}-{counter}.wav"
        counter += 1
    return path


def prune_relay_outputs(keep=RELAY_OUTPUT_LIMIT):
    files = sorted(RELAY_OUTPUT_DIR.glob("*.wav"), key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    for path in files[keep:]:
        try:
            path.unlink()
        except OSError:
            pass


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

    def send_text(self, text, content_type="text/html"):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.request_allowed():
            self.send_error(403)
            return
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if path == "/":
            self.send_text(HTML)
            return
        if path == "/api/ptt-state":
            with PTT_CONDITION:
                self.send_json(dict(PTT_STATE))
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
                    with PTT_CONDITION:
                        PTT_CONDITION.wait_for(
                            lambda: PTT_STATE["sequence"] != last_sequence,
                            timeout=15,
                        )
                        event = dict(PTT_STATE)
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
            self.send_json(setup_status())
            return
        if path == "/api/system-state":
            self.send_json({"system": system_snapshot()})
            return
        if path == "/api/state":
            query = urllib.parse.parse_qs(parsed_url.query)
            datasets = list_datasets()
            dataset_ids = {row["id"] for row in datasets}
            dataset_id = query.get("dataset", [""])[0]
            if dataset_id not in dataset_ids:
                dataset_id = datasets[0]["id"] if datasets else ""
            all_references = parse_references(dataset_id, valid_only=False)
            data = {
                "config_error": CONFIG_ERROR,
                "datasets": datasets,
                "reference_dataset": dataset_id,
                "models": polled_model_files(),
                "models_listed": MODEL_LIST["listed"],
                "references": [ref for ref in all_references if ref["valid_reference"]],
                "all_references": all_references,
                "api_ready": api_ready(),
                "relay_asr_ready": relay_asr_ready(),
                "jobs": {key: proc.snapshot() for key, proc in JOBS.items()},
                "system": system_snapshot(),
                "log": latest_log_text(),
            }
            self.send_json(data)
            return
        if path.startswith("/datasets/"):
            self.serve_file(DATASETS_DIR / path[len("/datasets/") :])
            return
        if path.startswith("/outputs/"):
            self.serve_file(OUTPUT_DIR / path[len("/outputs/") :])
            return
        self.send_error(404)

    def do_HEAD(self):
        if not self.request_allowed():
            self.send_error(403)
            return
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/datasets/"):
            self.serve_file(DATASETS_DIR / path[len("/datasets/") :], send_body=False)
            return
        if path.startswith("/outputs/"):
            self.serve_file(OUTPUT_DIR / path[len("/outputs/") :], send_body=False)
            return
        self.send_error(404)

    def serve_file(self, path, send_body=True):
        try:
            full = path.resolve()
            allowed = [DATASETS_DIR.resolve(), OUTPUT_DIR.resolve()]
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
        try:
            body = read_json(self)
            if path == "/api/start-system":
                # The startup worker picks the newest models when none are given, after WSL is running.
                start_system(body.get("gpt") or "", body.get("sovits") or "", body.get("binding") or "ShiftLeft")
                self.send_json({"ok": True})
                return
            if path == "/api/stop-system":
                stop_system()
                self.send_json({"ok": True})
                return
            if path == "/api/ptt-binding":
                with PTT_CONDITION:
                    PTT_STATE["binding"] = body.get("binding") or "ShiftLeft"
                    PTT_CONDITION.notify_all()
                self.send_json({"ok": True, "binding": PTT_STATE["binding"]})
                return
            if path == "/api/ptt-event":
                with PTT_CONDITION:
                    PTT_STATE["held"] = bool(body.get("held"))
                    PTT_STATE["sequence"] += 1
                    PTT_CONDITION.notify_all()
                self.send_json({"ok": True})
                return
            if path == "/api/cancel-generation":
                generation_id = str(body.get("generation_id") or "")
                if generation_id:
                    with STATE_LOCK:
                        CANCELLED_GENERATIONS.add(generation_id)
                    log(f"Cancelled relay generation {generation_id}.")
                self.send_json({"ok": True})
                return
            if path == "/api/create-dataset":
                result = create_dataset(body.get("name"), body.get("speaker"), body.get("language"))
                self.send_json({"ok": True, "dataset": result})
                return
            if path == "/api/upload-dataset-audio":
                result = add_dataset_audio(
                    body.get("dataset_id"),
                    body.get("filename"),
                    body.get("audio_base64"),
                    body.get("text"),
                    body.get("speaker"),
                    body.get("language"),
                )
                self.send_json({"ok": True, "dataset": result})
                return
            if path == "/api/sync-dataset":
                sync_dataset_job(body.get("dataset_id") or "")
                self.send_json({"ok": True})
                return
            if path == "/api/prepare-dataset":
                prepare_tts_dataset_job(
                    body.get("dataset_id") or "",
                    body.get("model_name") or "voice-model",
                )
                self.send_json({"ok": True})
                return
            if path == "/api/start-api":
                models = model_files()
                gpt = body.get("gpt") or (models["gpt"][0]["path"] if models["gpt"] else "")
                sovits = body.get("sovits") or (models["sovits"][0]["path"] if models["sovits"] else "")
                if not gpt or not sovits:
                    self.send_json({"ok": False, "error": "No GPT/SoVITS model pair found."}, 400)
                    return
                start_api(gpt, sovits)
                self.send_json({"ok": True})
                return
            if path == "/api/stop-api":
                stop_api()
                self.send_json({"ok": True})
                return
            if path == "/api/start-relay-asr":
                start_relay_asr()
                self.send_json({"ok": True})
                return
            if path == "/api/transcribe":
                if not relay_asr_ready():
                    start_relay_asr()
                    for _ in range(80):
                        if relay_asr_ready():
                            break
                        time.sleep(0.25)
                req = urllib.request.Request(
                    RELAY_ASR_URL + "/transcribe",
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
                start_train(
                    body.get("kind"),
                    int(body.get("epochs", 15)),
                    int(body.get("batch_size", 1)),
                    int(body.get("save_every", 5)),
                    body.get("dataset_id") or "",
                    body.get("model_name") or "voice-model",
                )
                self.send_json({"ok": True})
                return
            if path == "/api/stop-job":
                key = body.get("key")
                proc = JOBS.get(key)
                if proc:
                    proc.stop()
                elif key in WSL_STOP_SCRIPTS:
                    # Started before the dashboard restarted, so only WSL still knows the job.
                    stop_in_wsl(WSL_STOP_SCRIPTS[key])
                self.send_json({"ok": True})
                return
            if path == "/api/generate":
                if not api_ready():
                    self.send_json({"ok": False, "error": "Inference engine is not ready yet."}, 409)
                    return
                payload = {
                    "text": body["text"],
                    "text_lang": language_code(body.get("text_lang")),
                    "ref_audio_path": body["ref_audio_path"],
                    "aux_ref_audio_paths": body.get("aux_ref_audio_paths") or [],
                    "prompt_text": body["prompt_text"],
                    "prompt_lang": language_code(body.get("prompt_lang")),
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
                    with STATE_LOCK:
                        CANCELLED_GENERATIONS.discard(generation_id)
                req = urllib.request.Request(
                    API_URL + "/tts",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                # The Generate page and Relay can ask at the same time. One at a time keeps one
                # request from switching the models while the other is generating.
                with GENERATION_LOCK:
                    started = time.time()
                    set_api_weights(body.get("gpt"), body.get("sovits"))
                    with urllib.request.urlopen(req, timeout=240) as resp:
                        data = resp.read()
                        if "application/json" in resp.headers.get("Content-Type", ""):
                            self.send_json({"ok": False, "error": data.decode("utf-8", errors="replace")}, 400)
                            return
                    if generation_id:
                        with STATE_LOCK:
                            cancelled = generation_id in CANCELLED_GENERATIONS
                            CANCELLED_GENERATIONS.discard(generation_id)
                        if cancelled:
                            log(f"Discarded cancelled relay generation {generation_id}.")
                            self.send_json({"ok": False, "cancelled": True, "error": "Generation cancelled."}, 409)
                            return
                    if int(payload.get("streaming_mode", 0)):
                        data = repair_streamed_wav(data)
                    output = new_output_path(RELAY_OUTPUT_DIR if body.get("relay") else OUTPUT_DIR, body.get("voice"))
                    output.write_bytes(data)
                if body.get("relay"):
                    prune_relay_outputs()
                log(f"Generated {output.name} in {time.time() - started:.1f}s.")
                url = "/outputs/" + output.relative_to(OUTPUT_DIR).as_posix()
                self.send_json({"ok": True, "url": url, "name": output.name})
                return
            self.send_error(404)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.send_json({"ok": False, "error": detail or str(exc)}, exc.code)
        except Exception as exc:
            log(f"Error: {exc}")
            self.send_json({"ok": False, "error": str(exc)}, 500)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Local Voice Systems</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111315;
      --panel: #181b1e;
      --panel-2: #1d2024;
      --panel-3: #23272b;
      --field: #121416;
      --line: #30343a;
      --line-subtle: #272b30;
      --text: #e4e7eb;
      --muted: #9aa1a9;
      --accent: #3cab61;
      --accent-dark: #2b7745;
      --accent-soft: rgba(60, 171, 97, .12);
      --good: #48b96c;
      --bad: #b96b63;
      --warn: #c0a562;
      --focus: #55bd76;
      --radius: 4px;
      --control-height: 40px;
      --space-1: 4px;
      --space-2: 8px;
      --space-3: 12px;
      --space-4: 16px;
      --space-5: 24px;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      padding-bottom: 112px;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 "Segoe UI", system-ui, sans-serif;
      min-height: 100vh;
      background-image: linear-gradient(rgba(255,255,255,.009) 1px, transparent 1px);
      background-size: 100% 5px;
    }
    header {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: var(--space-4);
      padding: 10px var(--space-5);
      border-bottom: 1px solid var(--line-subtle);
      background: #16191c;
      position: sticky;
      top: 0;
      z-index: 20;
    }
    h1 { margin: 0; font-size: 16px; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
    h2 { margin: 0 0 var(--space-3); font-size: 15px; letter-spacing: 0; }
    .chip {
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 6px 10px;
      border-radius: var(--radius);
      color: var(--muted);
      white-space: nowrap;
    }
    .chip.good { color: var(--good); border-color: var(--accent-dark); }
    .chip.bad { color: var(--bad); border-color: #743333; }
    main {
      max-width: 1680px;
      margin: 0 auto;
      padding: var(--space-5);
    }
    .tablist {
      display: flex;
      gap: var(--space-1);
      justify-content: center;
      flex-wrap: wrap;
    }
    .tab-button {
      background: transparent;
      border: 1px solid transparent;
      border-radius: var(--radius);
      color: var(--muted);
      min-height: 34px;
      padding: 6px 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      font-size: 12px;
      font-weight: 600;
    }
    .tab-button:hover:not(:disabled) { background: var(--panel-3); border-color: transparent; color: var(--text); }
    .tab-button.active {
      background: var(--panel-3);
      color: var(--text);
      box-shadow: inset 0 -2px 0 var(--accent);
    }
    .tab-button:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    section.relay-page.active { display: grid; }
    section {
      border: 1px solid var(--line-subtle);
      background: var(--panel);
      border-radius: var(--radius);
      padding: var(--space-5);
      box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-4); }
    .grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--space-4); }
    label { display: block; color: var(--muted); font-size: 12px; margin-bottom: var(--space-2); }
    input:not([type="checkbox"]):not([type="range"]), select, textarea {
      width: 100%;
      background: var(--field);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      min-height: var(--control-height);
      padding: 9px 12px;
      font: inherit;
    }
    textarea { min-height: 92px; resize: vertical; }
    input:not([type="checkbox"]):not([type="range"]):focus, select:focus, textarea:focus {
      outline: 2px solid var(--accent-soft);
      outline-offset: 1px;
      border-color: var(--focus);
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      min-height: 0;
      margin: 0;
      accent-color: var(--accent);
      flex: 0 0 16px;
    }
    input[type="range"] {
      width: 100%;
      height: var(--control-height);
      min-height: var(--control-height);
      margin: 0;
      padding: 0;
      accent-color: var(--accent);
    }
    label:has(> input[type="checkbox"]) {
      display: inline-flex;
      align-items: center;
      gap: var(--space-2);
      min-height: var(--control-height);
      margin: 0;
    }
    button {
      border: 1px solid var(--accent-dark);
      background: #26362c;
      color: #e4f4e8;
      border-radius: var(--radius);
      min-height: var(--control-height);
      padding: 9px 14px;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover:not(:disabled) { background: #2d4034; border-color: var(--accent); }
    button.secondary {
      background: var(--panel-3);
      border-color: var(--line);
      color: var(--text);
    }
    button.secondary:hover:not(:disabled) { background: #2a2f34; border-color: #444a51; }
    button.danger {
      background: #382526;
      border-color: #704142;
      color: #efcdca;
    }
    button.danger:hover:not(:disabled) { background: #472b2d; border-color: #8a4c4e; }
    button.active { background: #2b5839; border-color: var(--accent); color: #eef8f1; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .row { display: flex; gap: var(--space-3); align-items: center; flex-wrap: wrap; }
    .row > * { flex: 1; }
    .row > button { flex: 0 0 auto; }
    .small { color: var(--muted); font-size: 12px; }
    .mini-button {
      min-height: 32px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 600;
    }
    .check-list {
      height: 230px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--field);
      margin-top: var(--space-2);
    }
    .check-row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      align-items: center;
      gap: var(--space-2);
      padding: var(--space-2) 10px;
      border-bottom: 1px solid var(--line-subtle);
      color: var(--text);
      font-size: 12px;
    }
    .check-row.selected { background: var(--accent-soft); }
    .check-row.invalid { opacity: .58; }
    .clip-title { color: var(--text); font-weight: 600; }
    .clip-time { color: #858c94; font-size: 11px; font-weight: 400; margin-left: 8px; }
    .check-row:last-child { border-bottom: 0; }
    .check-row.disabled { color: #667078; }
    .selected-list {
      min-height: 35px;
      display: flex;
      gap: var(--space-2);
      flex-wrap: wrap;
      margin-top: var(--space-2);
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: var(--space-2);
      max-width: 100%;
      padding: 5px 8px;
      background: var(--panel-3);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      color: var(--text);
      font-size: 12px;
    }
    .pill button {
      padding: 0 4px;
      border: 0;
      background: transparent;
      color: var(--muted);
      font-weight: 700;
      min-height: 0;
    }
    .pill button:hover:not(:disabled) { background: transparent; border-color: transparent; color: var(--text); }
    .pill .pill-play {
      color: var(--text);
      font-weight: 600;
      text-align: left;
    }
    .preset-row {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: var(--space-2);
      margin-top: var(--space-2);
    }
    .preset-row button { min-width: 0; }
    .log {
      height: 430px;
      overflow: auto;
      white-space: pre-wrap;
      background: #0d0f11;
      border: 1px solid var(--line-subtle);
      border-radius: var(--radius);
      padding: var(--space-3);
      color: #cbd0d5;
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 12px;
    }
    audio { width: 100%; margin-top: 8px; }
    .relay-block-head { display: flex; align-items: center; justify-content: space-between; gap: var(--space-2); min-height: 32px; }
    .relay-block-head label { margin: 0; }
    .relay-volume-row { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: var(--space-3); }
    .relay-volume-row label { margin: 0; }
    #tab-generate audio { display: block; width: 100%; min-width: 0; height: 40px; margin: 0; }
    #tab-generate textarea { margin: 0; }
    #targetText { min-height: 88px; }
    #tab-generate .check-list { height: 190px; margin-top: 0; }
    #tab-generate .selected-list { margin-top: 0; min-height: var(--control-height); }
    .cancel-generation { display: none; }
    .cancel-generation.visible { display: inline-block; }
    .global-error { display: none; max-width: 1680px; margin: 12px auto 0; border: 1px solid #7e4944; background: #2c1d1b; color: #e5b8b3; padding: 12px 16px; }
    .global-error.visible { display: flex; align-items: center; justify-content: space-between; gap: 15px; }
    .setup-list { display: grid; gap: var(--space-2); }
    .setup-scope-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: var(--space-3);
      margin-top: var(--space-4);
    }
    .setup-scope-item {
      min-height: 112px;
      padding: var(--space-3);
      border: 1px solid var(--line-subtle);
      border-radius: var(--radius);
      background: var(--panel);
    }
    .setup-scope-item strong { display: block; margin-bottom: var(--space-2); }
    .setup-legend {
      display: flex;
      gap: var(--space-3);
      flex-wrap: wrap;
      margin-bottom: var(--space-3);
      color: var(--muted);
      font-size: 12px;
    }
    .setup-legend strong { color: var(--text); }
    .setup-row { display: grid; grid-template-columns: 170px 130px 1fr; gap: var(--space-3); align-items: center; border: 1px solid var(--line-subtle); background: var(--field); padding: var(--space-3); }
    .setup-state { text-transform: uppercase; font-size: 11px; letter-spacing: .07em; color: var(--muted); }
    .setup-state.ready { color: var(--good); }
    .setup-state.missing { color: var(--bad); }
    .setup-state.needs_configuration, .setup-state.needs_browser_check { color: var(--warn); }
    .training-project { border: 1px solid var(--line-subtle); background: var(--field); padding: var(--space-4); }
    .upload-list { display: grid; gap: var(--space-2); max-height: 310px; overflow: auto; margin-top: var(--space-3); }
    .upload-row { display: grid; grid-template-columns: minmax(150px,.7fr) minmax(260px,1.5fr); gap: var(--space-2); align-items: center; border: 1px solid var(--line-subtle); padding: var(--space-2); background: #101214; }
    .upload-row .file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); font-size: 12px; }
    .models {
      display: grid;
      gap: 8px;
      max-height: 180px;
      overflow: auto;
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }
    .model-line { color: var(--muted); font-size: 12px; display: flex; justify-content: space-between; gap: 12px; }
    .output-name { color: var(--good); font-weight: 700; }
    /* ---------- Relay page: history sidebar, central workspace ---------- */
    :root { --rec: #e0524a; --rec-soft: rgba(224, 82, 74, .18); --header-height: 56px; --island-height: 112px; }
    section.relay-page {
      grid-template-columns: 272px minmax(0, 1fr);
      gap: var(--space-5);
      align-items: start;
      background: transparent;
      border: 0;
      box-shadow: none;
      padding: 0;
      min-height: calc(100vh - var(--header-height) - var(--island-height) - 48px);
    }
    .history {
      display: flex;
      flex-direction: column;
      gap: var(--space-2);
      border: 1px solid var(--line-subtle);
      background: var(--panel);
      border-radius: 8px;
      padding: var(--space-3);
      position: sticky;
      top: calc(var(--header-height) + 20px);
      max-height: calc(100vh - var(--header-height) - var(--island-height) - 40px);
    }
    .history-new { width: 100%; }
    .history-list {
      flex: 1 1 auto;
      min-height: 80px;
      overflow: auto;
      display: grid;
      gap: 4px;
      align-content: start;
    }
    .history-item {
      display: grid;
      gap: 3px;
      width: 100%;
      text-align: left;
      background: transparent;
      border: 1px solid transparent;
      border-radius: 6px;
      padding: 8px 10px;
      min-height: 0;
      font-weight: 400;
      color: var(--text);
    }
    .history-item:hover:not(:disabled) { background: var(--panel-3); border-color: var(--line); }
    .history-item:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }
    .history-item[aria-selected="true"] { background: var(--accent-soft); border-color: var(--accent-dark); }
    .history-text {
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      font-size: 13px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .history-meta { display: flex; justify-content: space-between; gap: 8px; font-size: 11px; color: var(--muted); }
    .history-state { font-weight: 700; text-transform: uppercase; letter-spacing: .06em; font-size: 10px; }
    .history-state:empty { display: none; }
    .history-state.queued { color: var(--warn); }
    .history-state.generating, .history-state.ready, .history-state.playing { color: var(--accent); }
    .history-state.failed { color: var(--bad); }
    .history-empty { color: var(--muted); font-size: 12px; padding: 14px 8px; text-align: center; }
    .history-foot {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      border-top: 1px solid var(--line-subtle);
      padding-top: 8px;
      min-height: 30px;
    }
    .history-toggle { display: none; }
    .workspace {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: var(--space-4);
      padding: var(--space-5) var(--space-4);
      min-height: calc(100vh - var(--header-height) - var(--island-height) - 48px);
    }
    .workspace[data-view="result"] { justify-content: flex-start; }
    .capture {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: var(--space-3);
      width: 100%;
      max-width: 760px;
    }
    .workspace[data-view="result"] .capture {
      flex-direction: row;
      flex-wrap: wrap;
      align-items: center;
      justify-content: flex-start;
      gap: var(--space-2) var(--space-4);
      padding-bottom: var(--space-4);
      border-bottom: 1px solid var(--line-subtle);
    }
    .talk-button {
      --level: 0;
      position: relative;
      width: 176px;
      height: 176px;
      border-radius: 50%;
      border: 2px solid var(--line);
      background: radial-gradient(circle at 50% 35%, #272c31, #1a1e22 72%);
      color: var(--text);
      display: grid;
      place-items: center;
      padding: 0;
      min-height: 0;
      cursor: pointer;
      touch-action: none;
      user-select: none;
      -webkit-user-select: none;
      transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease, background .15s ease, width .25s ease, height .25s ease;
    }
    .talk-button svg { width: 36%; height: 36%; pointer-events: none; }
    .talk-button:hover:not(:disabled) {
      border-color: var(--accent);
      background: radial-gradient(circle at 50% 35%, #2f3a33, #1d2420 72%);
      box-shadow: 0 0 0 8px var(--accent-soft);
      transform: translateY(-1px);
    }
    .talk-button:focus-visible { outline: 3px solid var(--focus); outline-offset: 5px; }
    .talk-button.active, .talk-button.active:hover:not(:disabled) {
      border-color: var(--rec);
      background: radial-gradient(circle at 50% 35%, #4d2521, #2d1614 72%);
      color: #ffd9d6;
      transform: scale(.96);
      box-shadow: 0 0 0 4px rgba(224, 82, 74, .55), 0 0 0 calc(6px + 18px * var(--level)) var(--rec-soft);
    }
    .talk-button.active::after {
      content: "";
      position: absolute;
      inset: -10px;
      border-radius: 50%;
      border: 2px solid rgba(224, 82, 74, .55);
      animation: talk-pulse 1.4s ease-out infinite;
      pointer-events: none;
    }
    @keyframes talk-pulse { from { transform: scale(1); opacity: .9; } to { transform: scale(1.2); opacity: 0; } }
    .workspace[data-view="result"] .talk-button { width: 64px; height: 64px; }
    .workspace[data-view="result"] .talk-button::after { inset: -6px; }
    .capture-status {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 36px;
      font-size: 22px;
      font-weight: 600;
      line-height: 1.2;
      text-align: center;
    }
    .workspace[data-view="result"] .capture-status { font-size: 17px; text-align: left; }
    .capture-status.recording { color: var(--rec); }
    .capture-status.transcribing, .capture-status.generating, .capture-status.queued, .capture-status.playing { color: var(--accent); }
    .capture-status.failed { color: var(--bad); }
    .status-icon { display: inline-flex; width: 24px; height: 24px; align-items: center; justify-content: center; flex: 0 0 24px; }
    .status-icon svg { width: 100%; height: 100%; }
    .status-icon:empty { display: none; }
    .rec-dot { width: 13px; height: 13px; border-radius: 50%; background: var(--rec); animation: rec-blink 1.1s steps(2, start) infinite; }
    @keyframes rec-blink { to { visibility: hidden; } }
    .status-icon.transcribing svg { animation: pen-write 1s ease-in-out infinite; }
    @keyframes pen-write { 0%, 100% { transform: translate(0, 0) rotate(0deg); } 50% { transform: translate(3px, -2px) rotate(-8deg); } }
    .status-icon.generating svg rect, .status-icon.queued svg path.hand { transform-origin: center; }
    .status-icon.generating svg rect { animation: bar-bounce 1s ease-in-out infinite; transform-box: fill-box; transform-origin: bottom; }
    .status-icon.generating svg rect:nth-child(2) { animation-delay: .15s; }
    .status-icon.generating svg rect:nth-child(3) { animation-delay: .3s; }
    .status-icon.generating svg rect:nth-child(4) { animation-delay: .45s; }
    @keyframes bar-bounce { 0%, 100% { transform: scaleY(.45); } 50% { transform: scaleY(1); } }
    .status-icon.playing svg path.wave { animation: wave-fade 1s ease-in-out infinite; }
    @keyframes wave-fade { 0%, 100% { opacity: .3; } 50% { opacity: 1; } }
    .cancel-generation { margin-left: 4px; }
    .capture-hint {
      color: var(--muted);
      font-size: 13px;
      max-width: 520px;
      text-align: center;
      line-height: 1.45;
    }
    .capture-hint:empty { display: none; }
    .workspace[data-view="result"] .capture-hint { flex-basis: 100%; text-align: left; max-width: none; }
    .shortcut-control { display: flex; flex-direction: column; align-items: center; gap: 4px; }
    .workspace[data-view="result"] .shortcut-control { margin-left: auto; align-items: flex-end; }
    .shortcut {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: transparent;
      border: 1px solid transparent;
      border-radius: 6px;
      color: var(--muted);
      padding: 6px 10px;
      min-height: 34px;
      font-weight: 500;
      font-size: 13px;
      letter-spacing: 0;
    }
    .shortcut svg { width: 18px; height: 18px; flex: 0 0 18px; }
    .shortcut:hover:not(:disabled), .shortcut:focus-visible { border-color: var(--line); background: var(--panel-2); color: var(--text); }
    .shortcut:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
    .shortcut.capturing, .shortcut.capturing:hover:not(:disabled) { border-color: var(--accent); background: var(--accent-soft); color: var(--text); }
    .shortcut .keycaps { display: inline-flex; align-items: center; gap: 6px; }
    .keycap {
      display: inline-block;
      padding: 1px 8px;
      border: 1px solid var(--line);
      border-bottom-width: 3px;
      border-radius: 5px;
      background: var(--panel-3);
      color: var(--text);
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 12px;
      line-height: 1.6;
      white-space: nowrap;
    }
    .shortcut.capturing .keycaps { color: var(--accent); font-weight: 600; }
    .shortcut-note { font-size: 11px; color: var(--muted); }
    .shortcut-note:empty { display: none; }
    .manual-line { width: 100%; max-width: 560px; }
    .manual-line > summary { cursor: pointer; color: var(--muted); font-size: 13px; text-align: center; list-style: none; padding: 4px; border-radius: 4px; }
    .manual-line > summary::-webkit-details-marker { display: none; }
    .manual-line > summary:hover, .manual-line[open] > summary { color: var(--text); }
    .manual-line > summary:focus-visible { outline: 2px solid var(--focus); }
    .manual-body { display: grid; gap: 8px; margin-top: 8px; }
    .manual-body textarea { min-height: 72px; }
    .manual-body button { justify-self: end; }
    .workspace[data-view="result"] .manual-line { flex-basis: 100%; max-width: none; }
    .workspace[data-view="result"] .manual-line > summary { text-align: left; padding-left: 0; }
    .result { width: 100%; max-width: 760px; display: grid; gap: var(--space-4); }
    .result-label { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 6px; }
    .result-transcript { font-size: 21px; line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere; }
    .result-player { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; align-items: center; }
    #relayAudio { display: block; width: 100%; min-width: 0; height: 40px; margin: 0; }
    .result-meta { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .result-meta:empty { display: none; }
    .result-actions { display: flex; gap: 8px; flex-wrap: wrap; }
    .result-actions:empty { display: none; }
    /* ---------- Fixed control island ---------- */
    .island {
      position: fixed;
      left: 50%;
      bottom: 14px;
      transform: translateX(-50%);
      z-index: 40;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      gap: 4px 6px;
      width: max-content;
      max-width: calc(100vw - 24px);
      padding: 7px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(24, 27, 30, .95);
      backdrop-filter: blur(10px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, .4);
    }
    .island-group { display: flex; align-items: center; gap: 6px; }
    .island-group + .island-group { border-left: 1px solid var(--line-subtle); padding-left: 10px; margin-left: 4px; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }
    .status-pill .dot { width: 8px; height: 8px; border-radius: 50%; background: #565d65; flex: 0 0 8px; }
    .status-pill strong { font-weight: 600; color: var(--text); }
    .status-pill.ready .dot { background: var(--good); }
    .status-pill.ready strong { color: var(--good); }
    .status-pill.starting .dot { background: var(--warn); animation: rec-blink 1s steps(2, start) infinite; }
    .status-pill.starting strong { color: var(--warn); }
    .status-pill.failed .dot { background: var(--bad); }
    .status-pill.failed strong { color: var(--bad); }
    .island-system-state {
      font-size: 12px;
      color: var(--muted);
      max-width: 220px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      padding: 0 4px;
    }
    .island-system-state.good { color: var(--good); }
    .island-system-state.bad { color: var(--bad); }
    .island .mini-button { min-height: 30px; border-radius: 999px; white-space: nowrap; }
    .island label.switch {
      position: relative;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 30px;
      margin: 0;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      color: var(--text);
      cursor: pointer;
      white-space: nowrap;
    }
    .island label.switch:hover { background: var(--panel-3); }
    .switch input { position: absolute; opacity: 0; width: 1px; height: 1px; margin: 0; }
    .switch-track {
      position: relative;
      width: 30px;
      height: 16px;
      border-radius: 999px;
      background: #3a4046;
      border: 1px solid var(--line);
      flex: 0 0 30px;
      transition: background .15s ease;
    }
    .switch-track::after {
      content: "";
      position: absolute;
      top: 1px;
      left: 1px;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: #c6ccd2;
      transition: transform .15s ease;
    }
    .switch input:checked + .switch-track { background: var(--accent-dark); border-color: var(--accent); }
    .switch input:checked + .switch-track::after { transform: translateX(14px); background: #fff; }
    .switch input:focus-visible + .switch-track { outline: 2px solid var(--focus); outline-offset: 2px; }
    .switch-state { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; min-width: 22px; }
    .switch-state::after { content: "off"; }
    .switch input:checked ~ .switch-state::after { content: "on"; }
    .switch input:checked ~ .switch-state { color: var(--good); }
    /* ---------- Settings dialog ---------- */
    .settings-dialog {
      width: min(760px, calc(100vw - 32px));
      max-height: calc(100vh - 48px);
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      padding: 0;
      box-shadow: 0 20px 60px rgba(0, 0, 0, .5);
    }
    .settings-dialog::backdrop { background: rgba(0, 0, 0, .55); }
    .settings-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line-subtle);
      margin: 0;
    }
    .settings-head h2 { margin: 0; }
    .settings-body { padding: 18px; display: grid; gap: 20px; overflow: auto; max-height: calc(100vh - 130px); }
    .settings-body h3 { margin: 0 0 10px; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
    .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
    /* ---------- Secondary pages: Generate, Training, Logs, Setup ---------- */
    section.page {
      background: transparent;
      border: 0;
      box-shadow: none;
      padding: 0;
      max-width: 920px;
      margin: 0 auto;
    }
    section.page.active { display: grid; gap: var(--space-4); align-content: start; }
    section.page.page-wide { max-width: 1280px; }
    .page-head { display: flex; align-items: baseline; justify-content: space-between; gap: var(--space-3); flex-wrap: wrap; padding: 0 2px; }
    .page-head h2 { margin: 0; font-size: 18px; }
    .page-head .small { flex: 1 1 320px; }
    .panel { border: 1px solid var(--line-subtle); background: var(--panel); border-radius: 8px; padding: var(--space-4) var(--space-5); }
    .panel-head { display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); flex-wrap: wrap; margin-bottom: var(--space-3); }
    .panel-head h3 { margin: 0; font-size: 14px; display: flex; align-items: center; }
    .panel h4 { margin: 0 0 var(--space-3); font-size: 13px; }
    .step-number {
      display: inline-grid;
      place-items: center;
      width: 22px;
      height: 22px;
      margin-right: 8px;
      border-radius: 50%;
      border: 1px solid var(--line);
      background: var(--panel-3);
      font-size: 11px;
      color: var(--muted);
    }
    .status-line { font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }
    .status-line.good { color: var(--good); }
    .status-line.bad { color: var(--bad); }
    .status-line.warn { color: var(--warn); }
    .status-line:empty { display: none; }
    details.sub { border-top: 1px solid var(--line-subtle); margin-top: 0; }
    :not(details.sub) + details.sub { margin-top: var(--space-3); }
    details.sub > summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
      padding: 13px 0;
      list-style: none;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    details.sub > summary::-webkit-details-marker { display: none; }
    details.sub > summary::before { content: ""; width: 6px; height: 6px; border-right: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor; transform: rotate(-45deg); transition: transform .15s ease; flex: 0 0 6px; }
    details.sub[open] > summary::before { transform: rotate(45deg); }
    details.sub > summary:hover, details.sub[open] > summary { color: var(--text); }
    details.sub > summary:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 4px; }
    details.sub > .sub-body { display: grid; gap: var(--space-3); padding: 0 0 var(--space-3); }
    .field-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-3); }
    .field-row > * { min-width: 0; }
    .field-row audio { display: block; width: 100%; height: 40px; margin: 0; }
    .primary-big { min-height: 44px; padding: 10px 22px; font-size: 14px; }
    .generate-actions { display: flex; align-items: center; gap: var(--space-3); flex-wrap: wrap; margin-top: var(--space-3); }
    .generate-result { margin-top: var(--space-4); padding-top: var(--space-4); border-top: 1px solid var(--line-subtle); }
    #outputAudio { display: block; width: 100%; height: 40px; margin: 6px 0 8px; }
    .generate-history { margin-top: var(--space-3); display: grid; gap: 4px; }
    .history-line { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; font-size: 13px; color: var(--muted); padding: 4px 0; }
    .history-line span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .log-tools { display: flex; align-items: center; gap: var(--space-2); flex-wrap: wrap; }
    .log-tools input:not([type="checkbox"]) { width: 220px; min-height: 32px; padding: 5px 10px; }
    .generate-history::before { content: "Earlier this session"; display: block; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 6px 0 2px; }
    .log-tools label { margin: 0; min-height: 32px; }
    section.page .log { height: calc(100vh - var(--header-height) - var(--island-height) - 120px); min-height: 280px; }
    .setup-list { margin-top: var(--space-3); }
    .setup-list:empty { display: none; }
    body:not([data-tab="relay"]) .island-relay { display: none; }
    @media (max-width: 1020px) {
      .grid2 { grid-template-columns: 1fr; }
      .grid3 { grid-template-columns: 1fr; }
      .setup-scope-grid { grid-template-columns: 1fr; }
      header { grid-template-columns: 1fr; }
      header .tablist { justify-content: flex-start; }
      section.relay-page { grid-template-columns: 1fr; gap: var(--space-3); }
      .history-toggle { display: inline-flex; gap: 4px; justify-self: start; }
      .history {
        display: none;
        position: fixed;
        left: 12px;
        top: 12px;
        bottom: calc(var(--island-height) + 8px);
        width: min(320px, calc(100vw - 24px));
        z-index: 45;
        max-height: none;
        box-shadow: 0 20px 60px rgba(0, 0, 0, .5);
      }
      .history.open { display: flex; }
      .workspace { padding: var(--space-3); min-height: calc(100vh - 200px); }
      section.relay-page { min-height: 0; }
    }
    @media (max-width: 760px) {
      :root { --island-height: 150px; }
      body { padding-bottom: 160px; }
      .island { border-radius: 18px; gap: 6px; }
      .island-group + .island-group { border-left: 0; padding-left: 0; margin-left: 0; }
      .island-system-state { max-width: 160px; }
      header { padding: 8px var(--space-3); gap: var(--space-2); }
      .tab-button { min-height: 28px; padding: 4px 8px; font-size: 11px; letter-spacing: .04em; }
      .workspace[data-view="result"] .capture { align-items: flex-start; }
      .workspace[data-view="result"] .shortcut-control { margin-left: 0; align-items: flex-start; }
      .talk-button { width: 150px; height: 150px; }
      .capture-status { font-size: 19px; }
      .result-transcript { font-size: 18px; }
    }
    @media (max-width: 600px) {
      main { padding: var(--space-3); }
      section { padding: var(--space-3); }
      section.relay-page { padding: 0; }
      .relay-volume-row { grid-template-columns: 1fr; gap: 8px; }
      .result-player { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation: none !important; transition: none !important; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Local Voice Systems</h1>
    <nav class="tablist" role="tablist" aria-label="Pages">
      <button class="tab-button active" data-tab="relay" role="tab" aria-selected="true">Relay</button>
      <button class="tab-button" data-tab="generate" role="tab" aria-selected="false">Generate</button>
      <button class="tab-button" data-tab="training" role="tab" aria-selected="false">TTS Training</button>
      <button class="tab-button" data-tab="logs" role="tab" aria-selected="false">Logs</button>
      <button class="tab-button" data-tab="setup" role="tab" aria-selected="false">Setup</button>
    </nav>
    <div class="sr-only" id="recentActivity" aria-live="polite">Dashboard loaded. System is stopped.</div>
  </header>
  <div class="global-error" id="globalError"><span id="globalErrorText"></span><div class="row"><button class="secondary mini-button" id="openErrorLogs">Open Logs</button><button class="danger mini-button" id="dismissError">Dismiss</button></div></div>

  <main>
      <section id="tab-generate" class="tab-panel page" role="tabpanel">
        <div class="page-head"><h2>Generate a line</h2><span class="small">Type text, choose the reference clip, and generate a WAV in the selected voice.</span></div>
        <div class="panel">
          <label for="targetText">Text to speak</label>
          <textarea id="targetText">Please remain calm. Your message is being processed.</textarea>
          <div class="generate-actions">
            <button class="primary-big" id="generate">Generate WAV</button>
            <span class="status-line" id="generateStatus"></span>
          </div>
          <div class="generate-result" id="generateResult" hidden>
            <div class="result-label">Generated voice</div>
            <audio id="outputAudio" controls></audio>
            <div class="small output-name" id="outputName"></div>
            <div class="generate-history" id="generateHistory" hidden></div>
          </div>
        </div>
        <div class="panel">
          <div class="panel-head"><h3>Voice</h3><span class="status-line" id="engineStatus"></span></div>
          <div class="field-row">
            <div><label for="voiceDataset">Voice dataset</label><select id="voiceDataset"></select></div>
            <div><label for="reference">Reference clip (3-10 s)</label><select id="reference"></select></div>
            <div><label for="refAudio">Preview</label><audio id="refAudio" controls></audio></div>
          </div>
          <details class="sub">
            <summary>Text spoken in the reference clip</summary>
            <div class="sub-body">
              <textarea id="promptText" aria-label="Text spoken in the reference clip"></textarea>
              <div class="small">Filled from the clip. Edit it only if the clip says something else.</div>
            </div>
          </details>
          <details class="sub">
            <summary>Extra reference clips <span class="small" id="auxCount">0 selected</span></summary>
            <div class="sub-body">
              <div class="small">Optional additional clips for style. More clips can improve delivery but slow generation.</div>
              <div class="row"><input id="auxSearch" placeholder="Search clips" aria-label="Search clips" /><button class="secondary mini-button" id="clearAuxRefs">Clear</button></div>
              <div id="auxRefsList" class="check-list"></div>
              <div id="auxSelectedList" class="selected-list"></div>
              <audio id="auxPreview" controls></audio>
              <div class="small" id="auxPreviewStatus"></div>
            </div>
          </details>
          <details class="sub">
            <summary>Engine and models</summary>
            <div class="sub-body">
              <div class="field-row">
                <div><label for="gptModel">GPT model</label><select id="gptModel"></select></div>
                <div><label for="sovitsModel">SoVITS model</label><select id="sovitsModel"></select></div>
              </div>
              <div class="row"><button class="secondary" id="startApi">Start Engine</button><button class="secondary" id="stopApi">Stop Engine</button><span class="small">Generate starts the engine on its own when it is off.</span></div>
            </div>
          </details>
          <details class="sub">
            <summary>Generation settings</summary>
            <div class="sub-body">
              <div class="grid3">
                <div><label for="textLang" title="Language of the text to speak. It follows the voice dataset's language when you choose a dataset.">Text language</label><select id="textLang"><option value="en">English</option><option value="zh">Chinese</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="yue">Cantonese</option></select></div>
                <div><label>Slice method</label><select id="splitMethod">
                  <option value="cut5">English punctuation</option>
                  <option value="cut4">Periods only</option>
                  <option value="cut0">Do not split</option>
                </select></div>
                <div><label>Seed</label><input id="seed" type="number" value="-1" /></div>
                <div><label>Top K</label><input id="topK" type="number" min="1" max="100" value="15" /></div>
                <div><label>Repetition penalty</label><input id="repPenalty" type="number" step="0.05" min="1" max="2" value="1.35" /></div>
                <div><label>Top P</label><input id="topP" type="number" step="0.05" min="0" max="1" value="0.6" /></div>
                <div><label>Temperature</label><input id="temperature" type="number" step="0.05" min="0" max="1" value="0.6" /></div>
                <div><label>Speed</label><input id="speed" type="number" step="0.05" min="0.5" max="1.5" value="1" /></div>
                <div><label>Pause</label><input id="pause" type="number" step="0.05" min="0" max="2" value="0.3" /></div>
              </div>
            </div>
          </details>
        </div>
      </section>

      <section id="tab-relay" class="tab-panel active relay-page" role="tabpanel">
        <button class="secondary mini-button history-toggle" id="historyToggle" aria-expanded="false" aria-controls="relayHistory">Recordings <span id="historyCount"></span></button>
        <aside class="history" id="relayHistory" aria-label="Recordings">
          <button class="secondary history-new" id="relayNewRecording">+ New recording</button>
          <div class="history-list" id="relayQueue" role="listbox" aria-label="Recordings from this session"></div>
          <div class="history-foot">
            <span class="small" id="relayQueueCount">0 waiting</span>
            <button class="secondary mini-button" id="relayClearQueue" hidden>Clear waiting</button>
          </div>
        </aside>

        <div class="workspace" id="relayWorkspace" data-view="capture">
          <div class="capture" id="relayCapture">
            <button class="talk-button" id="relayPttButton" aria-label="Hold to talk" aria-describedby="relayStatusText relayStatusHint">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/><path d="M8 21h8"/></svg>
            </button>
            <div class="capture-status" id="relayStatusBox" role="status" aria-live="polite">
              <span class="status-icon" id="relayStatusIcon" aria-hidden="true"></span>
              <span class="status-text" id="relayStatusText">Hold to talk</span>
              <button class="danger mini-button cancel-generation" id="relayCancelCurrent">Cancel</button>
            </div>
            <div class="capture-hint" id="relayStatusHint"></div>
            <div class="shortcut-control">
              <button class="shortcut" id="relaySetPttKey" type="button" aria-label="Push-to-talk shortcut. Activate to change.">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M6 14h.01M18 14h.01M9 14h6"/></svg>
                <span class="keycaps" id="relayPttKeyLabel"></span>
              </button>
              <input id="relayPttKey" type="hidden" value="ShiftLeft" />
              <span class="shortcut-note" id="relayShortcutNote"></span>
            </div>
            <details class="manual-line" id="relayManual">
              <summary>Type a line instead</summary>
              <div class="manual-body">
                <label class="sr-only" for="relayManualText">Line to speak</label>
                <textarea id="relayManualText" placeholder="Type a line for the voice to speak"></textarea>
                <button class="mini-button" id="relaySpeakManual">Speak this line</button>
              </div>
            </details>
          </div>

          <div class="result" id="relayResult" hidden>
            <div>
              <div class="result-label">Transcript</div>
              <div class="result-transcript" id="relayTranscript"></div>
            </div>
            <div id="relayPlayerBlock" hidden>
              <div class="result-label">Generated voice</div>
              <div class="result-player" id="relayPlayer">
                <audio id="relayAudio" controls></audio>
                <button class="secondary mini-button" id="relayPlayThis" hidden>Play</button>
                <button class="secondary mini-button" id="relayReplayLast">Replay</button>
              </div>
            </div>
            <div class="result-actions" id="relayResultActions"></div>
            <div class="result-meta" id="relayNowPlaying"></div>
          </div>
        </div>
      </section>

      <section id="tab-training" class="tab-panel page" role="tabpanel">
        <div class="page-head"><h2>TTS Training</h2><span class="small">Build a dataset, prepare its features, and train a named GPT-SoVITS voice.</span></div>

        <div class="panel">
          <div class="panel-head"><h3><span class="step-number">1</span>Dataset</h3><span class="status-line" id="trainingDatasetStatus">Loading datasets...</span></div>
          <label for="trainingDataset">Training dataset</label>
          <select id="trainingDataset"></select>
          <details class="sub">
            <summary>Add audio and transcripts</summary>
            <div class="sub-body">
              <input id="datasetAudioFiles" type="file" accept="audio/*,.wav,.ogg,.mp3,.flac,.m4a" multiple aria-label="Audio files to add" />
              <div class="small">Each selected file receives an editable transcript before upload. Filenames are converted into a starting suggestion.</div>
              <div class="upload-list" id="datasetUploadRows"></div>
              <div class="row"><button id="uploadDatasetAudio" disabled>Add Files to Dataset</button><span class="status-line" id="datasetUploadProgress"></span></div>
            </div>
          </details>
          <details class="sub">
            <summary>Create a new dataset</summary>
            <div class="sub-body">
              <div class="field-row">
                <div><label for="newDatasetName">Dataset name</label><input id="newDatasetName" placeholder="Example: Station Announcer" /></div>
                <div><label for="newDatasetSpeaker">Default speaker name</label><input id="newDatasetSpeaker" value="speaker" /></div>
                <div><label for="newDatasetLanguage">Language</label><select id="newDatasetLanguage"><option value="en">English</option><option value="zh">Chinese</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="yue">Cantonese</option></select></div>
              </div>
              <div><button class="secondary" id="createDataset">Create Dataset</button></div>
            </div>
          </details>
        </div>

        <div class="panel">
          <div class="panel-head"><h3><span class="step-number">2</span>Prepare</h3><span class="status-line" id="datasetPrepareStatus">Not prepared in this session.</span></div>
          <div class="field-row">
            <div>
              <label for="trainingModelName">Model name</label>
              <input id="trainingModelName" value="voice-model" placeholder="Example: station-announcer" />
              <div class="small" style="margin-top: 6px;">Saved checkpoints use: <strong id="trainingModelSlug">voice-model</strong></div>
            </div>
          </div>
          <div class="small" style="margin: 12px 0;">Syncs audio to WSL, normalizes transcripts, extracts voice features on the GPU, and creates the files required by both trainers.</div>
          <div class="row">
            <button id="prepareDataset">Sync &amp; Prepare</button>
            <button class="secondary" id="syncDataset">Sync Only</button>
            <button class="danger" data-stop="dataset-prepare">Stop Preparation</button>
          </div>
        </div>

        <div class="panel">
          <div class="panel-head"><h3><span class="step-number">3</span>Train</h3><span class="status-line" id="trainingJobsStatus"></span></div>
          <div class="grid2">
            <div class="training-project">
              <h4>SoVITS</h4>
              <label for="sovitsEpochs">Total epochs</label><input id="sovitsEpochs" type="number" value="16" min="1" />
              <label for="sovitsBatch">Batch size</label><input id="sovitsBatch" type="number" value="1" min="1" />
              <label for="sovitsSave">Save every N epochs</label><input id="sovitsSave" type="number" value="4" min="1" />
              <div class="row" style="margin-top: 10px;">
                <button id="trainSovits">Train SoVITS</button>
                <button class="danger" data-stop="train-sovits">Stop</button>
              </div>
              <div class="status-line" id="trainSovitsStatus" style="margin-top: 8px;"></div>
            </div>
            <div class="training-project">
              <h4>GPT</h4>
              <label for="gptEpochs">Total epochs</label><input id="gptEpochs" type="number" value="32" min="1" />
              <label for="gptBatch">Batch size</label><input id="gptBatch" type="number" value="1" min="1" />
              <label for="gptSave">Save every N epochs</label><input id="gptSave" type="number" value="5" min="1" />
              <div class="row" style="margin-top: 10px;">
                <button id="trainGpt">Train GPT</button>
                <button class="danger" data-stop="train-gpt">Stop</button>
              </div>
              <div class="status-line" id="trainGptStatus" style="margin-top: 8px;"></div>
            </div>
          </div>
        </div>

        <div class="panel">
          <details class="sub" style="border-top: 0; margin-top: 0;">
            <summary>Saved models <span class="small" id="modelCount"></span></summary>
            <div class="sub-body"><div class="models" id="modelList" style="border-top: 0; padding-top: 0; margin-top: 0;"></div></div>
          </details>
        </div>
      </section>

      <section id="tab-logs" class="tab-panel page page-wide" role="tabpanel">
        <div class="page-head">
          <h2>Logs</h2>
          <div class="log-tools">
            <input id="logFilter" placeholder="Filter lines" aria-label="Filter log lines" />
            <label><input id="logFollow" type="checkbox" checked /> Follow new lines</label>
            <button class="secondary mini-button" id="copyLog">Copy all</button>
            <span class="status-line" id="logStatus"></span>
          </div>
        </div>
        <pre class="log" id="log" aria-live="off"></pre>
      </section>

      <section id="tab-setup" class="tab-panel page" role="tabpanel">
        <div class="page-head"><h2>System Setup</h2><span class="small">A read-only check of the software, models, GPU access, and audio devices the voice tools need. It starts nothing and changes nothing.</span></div>
        <div class="panel">
          <div class="row">
            <button id="runSetupCheck">Run System Check</button>
            <span class="status-line" id="setupSummary">No check has been run.</span>
          </div>
          <div class="setup-list" id="setupList"></div>
          <details class="sub">
            <summary>What this check inspects</summary>
            <div class="sub-body">
              <div class="setup-scope-grid" style="margin-top: 0;">
                <div class="setup-scope-item">
                  <strong>Software environment</strong>
                  <div class="small">The dashboard Python runtime, GPT-SoVITS files in WSL, the speech-recognition environment and model cache, and the global push-to-talk helper.</div>
                </div>
                <div class="setup-scope-item">
                  <strong>Models and GPU</strong>
                  <div class="small">Counts GPT and SoVITS checkpoints, then asks PyTorch inside WSL whether GPU acceleration is available and which GPU it sees.</div>
                </div>
                <div class="setup-scope-item">
                  <strong>Audio devices</strong>
                  <div class="small">Briefly requests microphone permission, closes the stream at once, and lists microphone, output, and VB-CABLE devices. No audio is recorded or saved.</div>
                </div>
              </div>
              <div class="setup-legend" style="margin: 0;"><span><strong>Ready:</strong> available now</span><span><strong>Needs configuration:</strong> installed but requires attention</span><span><strong>Missing:</strong> not detected</span></div>
            </div>
          </details>
        </div>
      </section>
  </main>

  <div class="island" id="systemIsland" role="region" aria-label="System status and global controls">
    <div class="island-group" aria-label="Service status">
      <span class="status-pill" id="relayAsrCard" title="Speech recognition service (local ASR)"><i class="dot" aria-hidden="true"></i>Speech <strong id="relayAsrStatus">Off</strong></span>
      <span class="status-pill" id="relayTtsCard" title="Voice engine (GPT-SoVITS)"><i class="dot" aria-hidden="true"></i>Voice <strong id="relayTtsStatus">Off</strong></span>
      <span class="status-pill" id="relayPttCard" title="Global push-to-talk helper (works when this page is not focused)"><i class="dot" aria-hidden="true"></i>Global key <strong id="relayPttStatus">Off</strong></span>
    </div>
    <div class="island-group">
      <span class="island-system-state" id="systemChip" role="status">system stopped</span>
      <button class="mini-button system-start" id="startSystem" title="Start speech recognition, the voice engine, and the global push-to-talk helper">Start System</button>
      <button class="danger mini-button system-stop" id="stopSystem" title="Stop all managed services and the microphone. Recordings stay listed.">Stop Everything</button>
    </div>
    <div class="island-group island-relay">
      <label class="switch" title="On: record while the key or button is held. Off: voice activation records whenever speech is heard."><input id="relayPushToTalk" type="checkbox" checked /><span class="switch-track" aria-hidden="true"></span>Hold to talk<span class="switch-state" aria-hidden="true"></span></label>
      <label class="switch" title="On: each transcript is spoken in the selected voice automatically. Off: transcribe only."><input id="relayAutoSpeak" type="checkbox" checked /><span class="switch-track" aria-hidden="true"></span>Speak automatically<span class="switch-state" aria-hidden="true"></span></label>
      <button class="secondary mini-button" id="openRelaySettings" aria-haspopup="dialog" title="Output device, volume, voice models, and generation settings">Settings</button>
    </div>
  </div>

  <dialog class="settings-dialog" id="relaySettingsDialog" aria-labelledby="relaySettingsTitle">
    <form method="dialog" class="settings-head">
      <h2 id="relaySettingsTitle">Audio and voice settings</h2>
      <button class="secondary mini-button" value="close">Close</button>
    </form>
    <div class="settings-body">
      <div>
        <h3>Playback</h3>
        <div class="relay-block-head">
          <label for="relayOutputDevice">Output device</label>
          <button class="secondary mini-button" id="refreshRelayDevices">Refresh</button>
        </div>
        <select id="relayOutputDevice"></select>
        <div class="relay-volume-row" style="margin-top: 10px;">
          <label for="relayVolume">Volume <span id="relayVolumeLabel">25%</span></label>
          <input id="relayVolume" type="range" value="25" min="0" max="100" step="1" />
          <label><input id="relayMonitorLocal" type="checkbox" checked /> Monitor locally</label>
        </div>
      </div>
      <div>
        <h3>Voice</h3>
        <div class="grid2">
          <div><label for="relayDataset" title="The dataset whose clips serve as voice references.">Voice dataset</label><select id="relayDataset"></select></div>
          <div><label title="The semantic language model checkpoint used to compose speech.">GPT model</label><select id="relayGptModel"></select></div>
          <div><label title="The acoustic voice checkpoint that produces the selected voice timbre.">SoVITS model</label><select id="relaySovitsModel"></select></div>
          <div>
            <label title="The main example clip used for voice character and delivery.">Reference style</label>
            <select id="relayReference"></select>
            <audio id="relayRefAudio" controls></audio>
          </div>
        </div>
        <details class="sub">
          <summary>Extra reference clips <span class="small" id="relayAuxCount">0 selected</span></summary>
          <div class="sub-body">
            <div class="small">Optional additional clips for style. More clips can improve delivery but slow generation.</div>
            <div class="row"><input id="relayAuxSearch" placeholder="Search clips" aria-label="Search clips" /><button class="secondary mini-button" id="relayClearAuxRefs">Clear</button></div>
            <div id="relayAuxRefsList" class="check-list" style="height: 190px; margin-top: 0;"></div>
            <div id="relayAuxSelectedList" class="selected-list" style="margin-top: 0;"></div>
            <audio id="relayAuxPreview" controls style="display: block; width: 100%; height: 40px; margin: 0;"></audio>
            <div class="small" id="relayAuxPreviewStatus"></div>
          </div>
        </details>
      </div>
      <div>
        <h3>Generation</h3>
        <label>Voice preset</label>
        <div class="preset-row">
          <button class="secondary mini-button" type="button" data-relay-profile="fast">Fast Relay</button>
          <button class="secondary mini-button" type="button" data-relay-profile="balanced">Balanced</button>
          <button class="secondary mini-button" type="button" data-relay-profile="quality">Best Quality</button>
        </div>
        <div class="grid3" style="margin-top: 12px;">
          <div><label title="Limits token choices. Lower values are more consistent; higher values add variation.">Top K</label><input id="relayTopK" type="number" min="1" max="100" value="15" /></div>
          <div><label title="Controls how much of the probability distribution can be sampled.">Top P</label><input id="relayTopP" type="number" step="0.05" min="0" max="1" value="0.6" /></div>
          <div><label title="Controls randomness. Lower is steadier; higher is more expressive.">Temperature</label><input id="relayTemperature" type="number" step="0.05" min="0" max="1" value="0.55" /></div>
          <div><label title="Speech playback speed. 1.0 is normal speed.">Speed</label><input id="relaySpeed" type="number" step="0.05" min="0.5" max="1.5" value="0.95" /></div>
          <div><label title="Pause inserted between generated fragments.">Pause</label><input id="relayPause" type="number" step="0.05" min="0" max="2" value="0.25" /></div>
          <div><label title="Discourages repeated words or phrases.">Repetition penalty</label><input id="relayRepPenalty" type="number" step="0.05" min="1" max="2" value="1.35" /></div>
        </div>
      </div>
    </div>
  </dialog>

  <script>
    // Settings saved by earlier versions used a different key prefix.
    ["RelaySettings", "RelayAuxSelected", "RelayPttKey", "RelayOutputDevice", "TrainingDataset", "TrainingModelName"].forEach(key => {
      try {
        const old = localStorage.getItem("workingJoe" + key);
        if (old !== null && localStorage.getItem("voiceDashboard" + key) === null) localStorage.setItem("voiceDashboard" + key, old);
      } catch (_) {}
    });
    // Settings of the removed voice changer page.
    try { ["voiceDashboardRvcDataset", "voiceDashboardRvcModelName"].forEach(key => localStorage.removeItem(key)); }
    catch (_) {}
    let state = null;
    let voiceDataset = "";
    try { voiceDataset = localStorage.getItem("voiceDashboardVoiceDataset") || ""; }
    catch (_) {}
    let auxSelected = new Set();
    let relayAuxSelected = new Set();
    try { relayAuxSelected = new Set(JSON.parse(localStorage.getItem("voiceDashboardRelayAuxSelected") || "[]")); }
    catch (_) { relayAuxSelected = new Set(); }
    let refsSignature = "";
    let modelsSignature = "";
    let refreshBusy = false;
    const $ = id => document.getElementById(id);
    let relayListening = false;
    let relayQueue = [];
    let relayBusy = false;
    let relayLastUrl = "";
    let relayNextId = 1;
    let relayStream = null;
    let relayAudioCtx = null;
    let relaySource = null;
    let relayProcessor = null;
    let relayKeepAlive = null;
    let relayPushHeld = false;
    let relayBindingPtt = false;
    let relayCapturing = false;
    let relayCurrentChunks = [];
    let relayCurrentFrames = 0;
    let relayLastVoiceAt = 0;
    let relayLastLevelDb = -120;
    let relayCurrentItem = null;
    let relayPlaybackResolve = null;
    let armRelayWhenReady = false;
    let lastSystemMessage = "";
    let datasetUploadFiles = [];
    // Relay workspace view: "capture" shows the recording control alone; "result" shows one queue item.
    let relayView = {mode: "capture", itemId: null};
    let relayTranscribing = false;
    let relayStatusText = "";
    let relayLoadedUrl = "";
    let relayLevelTick = 0;
    let lastSystemSnapshot = null;
    let lastSystemStatus = "";
    let pttBindingCancelledAt = 0;
    let engineBusy = false;
    let logRaw = "";
    let generateOutputs = [];
    let textLangDataset = "";
    let configErrorShown = false;
    let relaySavedSettings = {};
    try { relaySavedSettings = JSON.parse(localStorage.getItem("voiceDashboardRelaySettings") || "{}"); }
    catch (_) { relaySavedSettings = {}; }

    const RELAY_SETTING_IDS = [
      "relayGptModel", "relaySovitsModel", "relayReference",
      "relayOutputDevice", "relayPttKey", "relayMonitorLocal", "relayPushToTalk",
      "relayAutoSpeak", "relayVolume", "relayTopK", "relayTopP",
      "relayTemperature", "relaySpeed", "relayPause", "relayRepPenalty"
    ];

    function recentActivity(message) {
      if (message) $("recentActivity").textContent = message;
    }

    function showError(message) {
      $("globalErrorText").textContent = message || "An unknown error occurred.";
      $("globalError").classList.add("visible");
      recentActivity(`Failed: ${message}`);
    }

    window.alert = showError;
    $("dismissError").onclick = () => $("globalError").classList.remove("visible");
    $("openErrorLogs").onclick = () => activateTab("logs");

    function saveRelaySettings() {
      const saved = {};
      RELAY_SETTING_IDS.forEach(id => {
        const element = $(id);
        if (!element) return;
        if (element.tagName === "SELECT" && !element.options.length && relaySavedSettings[id] !== undefined) {
          saved[id] = relaySavedSettings[id];
        } else {
          saved[id] = element.type === "checkbox" ? element.checked : element.value;
        }
      });
      relaySavedSettings = saved;
      localStorage.setItem("voiceDashboardRelaySettings", JSON.stringify(saved));
    }

    function loadRelaySettings() {
      RELAY_SETTING_IDS.forEach(id => {
        const element = $(id);
        if (!element || relaySavedSettings[id] === undefined) return;
        if (element.type === "checkbox") element.checked = Boolean(relaySavedSettings[id]);
        else if (element.tagName !== "SELECT" || Array.from(element.options).some(option => option.value === relaySavedSettings[id])) element.value = relaySavedSettings[id];
      });
    }

    function componentCard(cardId, statusId, value) {
      const card = $(cardId);
      const label = $(statusId);
      card.classList.remove("ready", "starting", "failed");
      if (value === "ready") {
        card.classList.add("ready");
        label.textContent = "Ready";
      } else if (value === "starting") {
        card.classList.add("starting");
        label.textContent = "Starting";
      } else if (value === "failed") {
        card.classList.add("failed");
        label.textContent = "Failed";
      } else {
        label.textContent = "Off";
      }
    }

    function modelSlug(value) {
      return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64) || "voice-model";
    }

    function suggestedTranscript(filename) {
      const stem = String(filename || "").replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
      if (!stem) return "";
      const text = stem[0].toUpperCase() + stem.slice(1);
      return /[.!?]$/.test(text) ? text : text + ".";
    }

    function fillDatasetSelects(datasets, activeId) {
      ["voiceDataset", "relayDataset"].forEach(id => {
        const select = $(id);
        select.innerHTML = "";
        datasets.forEach(dataset => option(select, dataset.id, `${dataset.name} — ${dataset.wav_count} clips`));
        if (datasets.some(dataset => dataset.id === activeId)) select.value = activeId;
      });
      if (activeId !== voiceDataset) {
        voiceDataset = activeId;
        try { localStorage.setItem("voiceDashboardVoiceDataset", voiceDataset); } catch (_) {}
      }
    }

    function fillTrainingDatasets(datasets) {
      const select = $("trainingDataset");
      const old = select.value || localStorage.getItem("voiceDashboardTrainingDataset") || "";
      select.innerHTML = "";
      (datasets || []).forEach(dataset => option(select, dataset.id, `${dataset.name} — ${dataset.wav_count} clips`));
      if ((datasets || []).some(dataset => dataset.id === old)) select.value = old;
      const selected = (datasets || []).find(dataset => dataset.id === select.value);
      if (selected) {
        $("trainingDatasetStatus").textContent = `${selected.wav_count} WAV files / ${selected.list_rows} transcripts · ${selected.ready ? "ready to prepare" : "needs audio or transcripts"}`;
        $("trainingDatasetStatus").className = selected.ready ? "status-line good" : "status-line warn";
        $("uploadDatasetAudio").disabled = !datasetUploadFiles.length;
      }
    }

    function renderDatasetUploads() {
      const list = $("datasetUploadRows");
      list.innerHTML = "";
      datasetUploadFiles.forEach((file, index) => {
        const row = document.createElement("div");
        row.className = "upload-row";
        const name = document.createElement("div");
        name.className = "file-name";
        name.title = file.name;
        name.textContent = file.name;
        const transcript = document.createElement("input");
        transcript.value = suggestedTranscript(file.name);
        transcript.placeholder = "Exact spoken transcript";
        transcript.dataset.uploadTranscript = String(index);
        row.append(name, transcript);
        list.appendChild(row);
      });
      const selected = state && (state.datasets || []).find(dataset => dataset.id === $("trainingDataset").value);
      $("uploadDatasetAudio").disabled = !datasetUploadFiles.length || !selected;
    }

    function fileToBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    }

    loadRelaySettings();
    // Migrate a saved volume from the old 0-4x scale to 0-100%; new-scale values are always > 4.
    if (relaySavedSettings.relayVolume !== undefined && Number(relaySavedSettings.relayVolume) <= 4) {
      $("relayVolume").value = Math.round(Number(relaySavedSettings.relayVolume) / 4 * 100);
    }
    function activateTab(name) {
      if (!$(`tab-${name}`)) name = "relay";
      document.body.dataset.tab = name;
      document.querySelectorAll(".tab-button").forEach(button => {
        const active = button.dataset.tab === name;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", active ? "true" : "false");
      });
      document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `tab-${name}`);
      });
      if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    }
    document.querySelectorAll(".tab-button").forEach(button => {
      button.onclick = () => {
        activateTab(button.dataset.tab);
        refresh();
      };
    });
    activateTab((location.hash || "#relay").slice(1));
    function option(select, value, label) {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      select.appendChild(opt);
    }

    function dbFromChunk(chunk) {
      if (!chunk || !chunk.length) return null;
      let sum = 0;
      for (let i = 0; i < chunk.length; i++) sum += chunk[i] * chunk[i];
      const rms = Math.sqrt(sum / chunk.length);
      return 20 * Math.log10(Math.max(rms, 0.000001));
    }

    function fillSelect(select, rows, oldValue) {
      select.innerHTML = "";
      rows.forEach(row => option(select, row.path, `${row.name} (${row.size_mb} MB)`));
      if (oldValue && rows.some(row => row.path === oldValue)) select.value = oldValue;
    }

    function fillReferences(refs) {
      const old = $("reference").value;
      $("reference").innerHTML = "";
      refs.forEach(ref => {
        option($("reference"), ref.id, ref.label);
      });
      if (old && refs.some(ref => ref.id === old)) $("reference").value = old;
      const selected = refs.find(ref => ref.id === $("reference").value) || refs[0];
      if (selected) setReference(selected, false);
      // Choosing another voice dataset sets the text language to that dataset's language.
      if (selected && state && state.reference_dataset !== textLangDataset) {
        textLangDataset = state.reference_dataset;
        $("textLang").value = selected.lang;
      }
      renderAuxRefs(state ? state.all_references : refs);
    }

    // One clip picker implementation serves both the Generate page and the Relay settings dialog.
    const generateAux = {
      ids: {search: "auxSearch", list: "auxRefsList", selected: "auxSelectedList", preview: "auxPreview", previewStatus: "auxPreviewStatus", count: "auxCount"},
      activeRef: () => $("reference").value,
      selected: auxSelected,
      onChange: null
    };
    const relayAux = {
      ids: {search: "relayAuxSearch", list: "relayAuxRefsList", selected: "relayAuxSelectedList", preview: "relayAuxPreview", previewStatus: "relayAuxPreviewStatus", count: "relayAuxCount"},
      activeRef: () => $("relayReference").value,
      selected: relayAuxSelected,
      onChange: () => {
        localStorage.setItem("voiceDashboardRelayAuxSelected", JSON.stringify(Array.from(relayAuxSelected)));
        updateRelayPresetStatus();
      }
    };

    function renderAuxPicker(picker, refs) {
      const activeRef = picker.activeRef();
      const validIds = new Set((state ? state.all_references : refs).filter(ref => ref.valid_aux_reference).map(ref => ref.id));
      Array.from(picker.selected).forEach(id => {
        if (id === activeRef || !validIds.has(id)) picker.selected.delete(id);
      });
      const filter = $(picker.ids.search).value.trim().toLowerCase();
      const list = $(picker.ids.list);
      list.innerHTML = "";
      refs
        .filter(ref => !filter || ref.label.toLowerCase().includes(filter) || ref.text.toLowerCase().includes(filter))
        .forEach(ref => {
          const row = document.createElement("div");
          const selected = picker.selected.has(ref.id);
          const canAdd = ref.valid_aux_reference && ref.id !== activeRef;
          row.className = "check-row" + (!canAdd ? " disabled" : "") + (selected ? " selected" : "") + (!ref.valid_aux_reference ? " invalid" : "");
          const detail = document.createElement("div");
          detail.className = "clip-title";
          detail.textContent = ref.label;
          const duration = document.createElement("span");
          duration.className = "clip-time";
          duration.textContent = ref.duration ? `${ref.duration}s${ref.valid_aux_reference ? "" : " long"}` : "unknown";
          detail.appendChild(duration);
          const toggle = document.createElement("button");
          toggle.type = "button";
          toggle.className = selected ? "mini-button" : "secondary mini-button";
          toggle.textContent = selected ? "Remove" : (ref.valid_aux_reference ? "Add" : "Preview only");
          toggle.disabled = !canAdd && !selected;
          toggle.onclick = () => {
            if (picker.selected.has(ref.id)) picker.selected.delete(ref.id);
            else if (canAdd) picker.selected.add(ref.id);
            renderAuxPicker(picker, refs);
          };
          const play = document.createElement("button");
          play.type = "button";
          play.className = "secondary mini-button";
          play.textContent = "Play";
          play.onclick = event => {
            event.preventDefault();
            event.stopPropagation();
            previewClip(picker, ref);
          };
          row.appendChild(detail);
          row.appendChild(toggle);
          row.appendChild(play);
          list.appendChild(row);
        });
      updatePickerCount(picker);
      if (picker.onChange) picker.onChange();
    }

    function updatePickerCount(picker) {
      $(picker.ids.count).textContent = `${picker.selected.size} selected`;
      const holder = $(picker.ids.selected);
      holder.innerHTML = "";
      if (!state) return;
      Array.from(picker.selected)
        .map(id => state.all_references.find(ref => ref.id === id))
        .filter(Boolean)
        .forEach(ref => {
          const pill = document.createElement("span");
          pill.className = "pill";
          const text = document.createElement("button");
          text.type = "button";
          text.className = "pill-play";
          text.textContent = ref.label;
          text.onclick = () => previewClip(picker, ref);
          const remove = document.createElement("button");
          remove.type = "button";
          remove.textContent = "x";
          remove.setAttribute("aria-label", `Remove ${ref.label}`);
          remove.onclick = () => {
            picker.selected.delete(ref.id);
            renderAuxPicker(picker, state.all_references);
          };
          pill.appendChild(text);
          pill.appendChild(remove);
          holder.appendChild(pill);
        });
    }

    function previewClip(picker, ref) {
      const player = $(picker.ids.preview);
      $(picker.ids.previewStatus).textContent = `Previewing: ${ref.label}`;
      player.pause();
      player.src = ref.local_url;
      player.load();
      const started = player.play();
      if (started && typeof started.catch === "function") {
        started.catch(error => {
          $(picker.ids.previewStatus).textContent = `Preview loaded: ${ref.label}. Press play on the audio bar.`;
          console.warn("Preview playback did not auto-start:", error);
        });
      }
    }

    function renderAuxRefs(refs) { renderAuxPicker(generateAux, refs); }
    function renderRelayAuxRefs() { if (state) renderAuxPicker(relayAux, state.all_references); }

    function setReference(ref, rewriteText = true) {
      $("refAudio").src = ref.local_url;
      if (rewriteText) $("promptText").value = ref.text;
      if (!$("promptText").value) $("promptText").value = ref.text;
    }

    function setRelayReference(ref) {
      if (!ref) return;
      $("relayRefAudio").src = ref.local_url;
    }

    function fillRelayReferences(refs) {
      const old = $("relayReference").value || relaySavedSettings.relayReference || "";
      $("relayReference").innerHTML = "";
      refs.forEach(ref => option($("relayReference"), ref.id, ref.label));
      if (old && refs.some(ref => ref.id === old)) $("relayReference").value = old;
      const selected = refs.find(ref => ref.id === $("relayReference").value) || refs[0];
      if (selected) {
        $("relayReference").value = selected.id;
        setRelayReference(selected);
      }
      updateRelayPresetStatus();
    }

    function relayRef() {
      if (!state) return null;
      return state.references.find(ref => ref.id === $("relayReference").value) || state.references[0] || null;
    }

    function relayAuxRefs() {
      if (!state) return [];
      const ref = relayRef();
      const activeId = ref ? ref.id : "";
      return Array.from(relayAuxSelected)
        .map(id => state.all_references.find(item => item.id === id))
        .filter(Boolean)
        .filter(item => item.valid_aux_reference && item.id !== activeId);
    }

    function updateRelayPresetStatus() {
      $("relayAuxCount").textContent = `${relayAuxRefs().length} selected`;
    }

    function updateRelayVolumeLabel() {
      const value = Number($("relayVolume").value || 0);
      $("relayVolumeLabel").textContent = `${Math.round(value)}%`;
      $("relayAudio").volume = Math.max(0, Math.min(1, value / 100));
    }

    function describePttKey(code) {
      const names = {
        Mouse0: "Mouse Left",
        Mouse1: "Mouse Middle",
        Mouse2: "Mouse Right",
        Mouse3: "Mouse Back",
        Mouse4: "Mouse Forward",
        ShiftLeft: "Left Shift",
        ShiftRight: "Right Shift",
        ControlLeft: "Left Ctrl",
        ControlRight: "Right Ctrl",
        AltLeft: "Left Alt",
        AltRight: "Right Alt",
        Space: "Space",
        Enter: "Enter",
        Tab: "Tab",
        Escape: "Escape"
      };
      if (names[code]) return names[code];
      if (/^Key[A-Z]$/.test(code)) return code.slice(3);
      if (/^Digit[0-9]$/.test(code)) return code.slice(5);
      if (/^Numpad[0-9]$/.test(code)) return "Numpad " + code.slice(6);
      return code || "Unassigned";
    }

    function pttMouseCode(event) {
      return `Mouse${event.button}`;
    }

    // Codes the global helper (ptt_helper.py) can observe; anything else only works while this page is focused.
    const GLOBAL_KEY_CODES = /^(ShiftLeft|ShiftRight|ControlLeft|ControlRight|AltLeft|AltRight|Space|Enter|Tab|Escape|Key[A-Z]|Digit[0-9]|Numpad[0-9]|Mouse[0-4])$/;

    function renderPttKey() {
      const code = $("relayPttKey").value;
      const control = $("relaySetPttKey");
      const caps = $("relayPttKeyLabel");
      caps.innerHTML = "";
      if (relayBindingPtt) {
        caps.textContent = "Press a key or mouse button";
        $("relayShortcutNote").textContent = "Escape keeps the current key.";
        control.setAttribute("aria-label", "Press a key or mouse button to set the push-to-talk shortcut. Escape keeps the current key.");
      } else {
        const cap = document.createElement("kbd");
        cap.className = "keycap";
        cap.textContent = describePttKey(code);
        caps.appendChild(cap);
        const note = document.createElement("span");
        note.textContent = "push-to-talk key";
        caps.appendChild(note);
        $("relayShortcutNote").textContent = GLOBAL_KEY_CODES.test(code) ? "" : "This key only works while this page is focused.";
        control.setAttribute("aria-label", `Push-to-talk shortcut: ${describePttKey(code)}. Activate to change.`);
      }
      control.classList.toggle("capturing", relayBindingPtt);
    }

    function startPttBinding() {
      if (relayBindingPtt || Date.now() - pttBindingCancelledAt < 400) return;
      relayBindingPtt = true;
      renderPttKey();
    }

    function cancelPttBinding() {
      relayBindingPtt = false;
      pttBindingCancelledAt = Date.now();
      setRelayPttKey($("relayPttKey").value);
    }

    function setRelayPttKey(code) {
      const key = code || "ShiftLeft";
      $("relayPttKey").value = key;
      renderPttKey();
      localStorage.setItem("voiceDashboardRelayPttKey", key);
      saveRelaySettings();
      post("/api/ptt-binding", {binding: key}).catch(error => showError(error.message));
    }

    async function refreshRelayDevices() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        relayStatus("audio device list unavailable");
        return;
      }
      try {
        const permissionStream = await navigator.mediaDevices.getUserMedia({audio: true});
        permissionStream.getTracks().forEach(track => track.stop());
        const oldOutput = $("relayOutputDevice").value || relaySavedSettings.relayOutputDevice || localStorage.getItem("voiceDashboardRelayOutputDevice") || "";
        const devices = await navigator.mediaDevices.enumerateDevices();
        $("relayOutputDevice").innerHTML = "";
        option($("relayOutputDevice"), "", "Default speakers");
        devices.filter(device => device.kind === "audiooutput").forEach(device => {
          option($("relayOutputDevice"), device.deviceId, device.label || `Output ${$("relayOutputDevice").length + 1}`);
        });
        if (oldOutput) $("relayOutputDevice").value = oldOutput;
        await applyRelayOutputDevice();
        relayStatus("audio devices ready");
      } catch (error) {
        relayStatus("audio devices failed");
        console.error(error);
      }
    }

    async function autoLoadRelayDevices() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
      try {
        // Device labels are only exposed once mic permission is granted; when they are
        // present, refreshRelayDevices() will not trigger a permission prompt.
        const devices = await navigator.mediaDevices.enumerateDevices();
        if (devices.some(device => device.kind === "audiooutput" && device.label)) await refreshRelayDevices();
      } catch (error) {
        console.error(error);
      }
    }

    async function applyRelayOutputDevice() {
      const audio = $("relayAudio");
      const deviceId = $("relayOutputDevice").value;
      localStorage.setItem("voiceDashboardRelayOutputDevice", deviceId);
      saveRelaySettings();
      if (!audio.setSinkId) {
        relayStatus("output selection needs Chrome or Edge");
        return;
      }
      await audio.setSinkId(deviceId).catch(error => {
        relayStatus("output route failed");
        console.error(error);
      });
    }

    // Internal status strings map to a short hint under the main status label. Empty = nothing extra to say.
    const STATUS_HINTS = {
      "starting local speech": "Starting speech recognition. The first recording can take a moment.",
      "starting TTS engine": "Starting the voice engine. This can take a minute or two.",
      "transcription failed": "Transcription failed. Check the Logs page.",
      "local speech failed": "The microphone or speech recognition failed. Check the Logs page.",
      "audio devices failed": "The audio device list failed. Allow microphone access and try again.",
      "audio device list unavailable": "Audio devices are unavailable in this browser.",
      "output selection needs Chrome or Edge": "Output device selection needs Chrome or Edge.",
      "output route failed": "Audio could not be routed to the selected output device.",
      "press play on output": "The browser blocked automatic playback. Press play on the player.",
      "no speech recognized": "No speech was recognized. Try again, a little closer to the microphone.",
      "recording too short": "That was too short. Hold the key a little longer while speaking.",
      "relay failed": "", "generating": "", "playing generated line": "", "line queued": "", "idle": "",
      "push-to-talk ready": "", "listening": "", "not listening": "", "listening local": "",
      "recording push-to-talk": "", "transcribing locally": "", "current line cancelled": "",
      "audio devices ready": "", "transcribed": ""
    };
    const ACTIVE_STATES = new Set(["queued", "generating", "ready", "playing", "failed"]);
    const HISTORY_STATES = {queued: "Waiting", generating: "Generating", ready: "Ready", playing: "Playing", completed: "", failed: "Failed", cancelled: "Cancelled", transcribed: "Not spoken"};
    const ICONS = {
      mic: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/><path d="M8 21h8"/></svg>',
      dot: '<span class="rec-dot"></span>',
      pen: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L18 10l-4-4L4 16v4z"/><path d="M13 7l4 4"/><path d="M4 22h16"/></svg>',
      clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><path class="hand" d="M12 8v4l3 2"/></svg>',
      bars: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="5" width="3" height="14" rx="1"/><rect x="8" y="5" width="3" height="14" rx="1"/><rect x="13" y="5" width="3" height="14" rx="1"/><rect x="18" y="5" width="3" height="14" rx="1"/></svg>',
      speaker: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9v6h4l5 4V5L8 9H4z"/><path class="wave" d="M16 9a4 4 0 0 1 0 6"/><path class="wave" d="M18.5 6.5a7.5 7.5 0 0 1 0 11"/></svg>',
      alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l10 18H2L12 3z"/><path d="M12 10v4"/><path d="M12 17.5h.01"/></svg>'
    };

    function relayStatus(text) {
      relayStatusText = text || "";
      recentActivity(text);
      renderRelayCenter();
    }

    function statusHint() {
      if (!relayStatusText) return "";
      if (/^push-to-talk key:/.test(relayStatusText)) return "";
      return STATUS_HINTS[relayStatusText] !== undefined ? STATUS_HINTS[relayStatusText] : relayStatusText;
    }

    function formatClock(stamp) {
      try { return new Date(stamp).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }
      catch (_) { return ""; }
    }

    function viewedRelayItem() {
      if (relayView.mode !== "result") return null;
      return relayQueue.find(item => item.id === relayView.itemId) || null;
    }

    function selectRelayItem(id) {
      relayView = {mode: "result", itemId: id};
      closeHistoryDrawer();
      renderRelayQueue();
    }

    function showCaptureView() {
      relayView = {mode: "capture", itemId: null};
      closeHistoryDrawer();
      renderRelayQueue();
    }

    function closeHistoryDrawer() {
      $("relayHistory").classList.remove("open");
      $("historyToggle").setAttribute("aria-expanded", "false");
    }

    function renderRelayQueue() {
      const list = $("relayQueue");
      list.innerHTML = "";
      const items = relayQueue.slice(-60).reverse();
      if (!items.length) {
        const empty = document.createElement("div");
        empty.className = "history-empty";
        empty.textContent = "Recordings from this session appear here.";
        list.appendChild(empty);
      }
      items.forEach(item => {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "history-item";
        row.setAttribute("role", "option");
        row.setAttribute("aria-selected", relayView.mode === "result" && relayView.itemId === item.id ? "true" : "false");
        const text = document.createElement("span");
        text.className = "history-text";
        text.textContent = item.text;
        const meta = document.createElement("span");
        meta.className = "history-meta";
        const stateSpan = document.createElement("span");
        stateSpan.className = `history-state ${item.status}`;
        stateSpan.textContent = HISTORY_STATES[item.status] !== undefined ? HISTORY_STATES[item.status] : item.status;
        const time = document.createElement("span");
        time.textContent = formatClock(item.createdAt);
        meta.append(stateSpan, time);
        row.append(text, meta);
        row.onclick = () => selectRelayItem(item.id);
        list.appendChild(row);
      });
      const waiting = relayQueue.filter(item => item.status === "queued").length;
      $("relayQueueCount").textContent = waiting ? `${waiting} waiting` : (relayQueue.length ? `${relayQueue.length} this session` : "Nothing recorded yet");
      $("relayClearQueue").hidden = !waiting;
      $("historyCount").textContent = relayQueue.length ? `(${relayQueue.length})` : "";
      renderRelayCenter();
    }

    function relayPrimaryState(item) {
      const ptt = $("relayPushToTalk").checked;
      const system = lastSystemSnapshot || {};
      if (relayPushHeld || (!ptt && relayCapturing)) {
        return {key: "recording", icon: "dot", label: ptt ? "Listening" : "Hearing speech", hint: ptt ? "Release to transcribe." : "Pause to transcribe."};
      }
      if (relayTranscribing) return {key: "transcribing", icon: "pen", label: "Transcribing", hint: ""};
      // The viewed item while it is still being worked on, or, in the capture view, whatever is being worked on now.
      const active = item && ACTIVE_STATES.has(item.status) ? item : (!item && relayCurrentItem ? relayCurrentItem : null);
      if (active) {
        const suffix = active === item ? "" : " for the previous line";
        switch (active.status) {
          case "queued": {
            const ahead = relayQueue.filter(row => (row.status === "queued" && row.id < active.id) || ["generating", "ready", "playing"].includes(row.status)).length;
            return {key: "queued", icon: "clock", label: ahead ? `Waiting, ${ahead} ahead` : "Waiting for the voice engine", hint: statusHint()};
          }
          case "generating": return {key: "generating", icon: "bars", label: "Generating voice" + suffix, hint: statusHint()};
          case "ready": return {key: "generating", icon: "bars", label: "Voice ready, starting playback", hint: ""};
          case "playing": return {key: "playing", icon: "speaker", label: "Playing" + suffix, hint: ""};
          case "failed": return {key: "failed", icon: "alert", label: "Voice generation failed", hint: active.error || statusHint()};
        }
      }
      if (!ptt) {
        if (relayListening && relayAudioCtx) return {key: "ready", icon: "", label: "Listening for speech", hint: "Voice activation is on. Speak, then pause, to transcribe."};
        if (relayListening) return {key: "ready", icon: "", label: "Starting microphone", hint: statusHint()};
        return {key: "ready", icon: "", label: "Press to start listening", hint: "Voice activation is on. Turn Hold to talk on to record by holding instead."};
      }
      let hint = statusHint();
      if (!hint && relayCurrentItem && item && item !== relayCurrentItem) hint = "Another line is still being generated or played.";
      if (!hint) {
        if (relayListening && !relayAudioCtx) hint = "Speech recognition is starting.";
        else if (system.status === "stopped" && !relayListening) hint = "Services start on first use. Start System also enables the global key while this page is not focused.";
        else if (system.status === "starting") hint = system.message || "Starting the system.";
        else if (system.status === "failed" || system.status === "degraded") hint = system.message || "";
      }
      return {key: "ready", icon: "", label: "Hold to talk", hint};
    }

    function renderRelayCenter() {
      const item = viewedRelayItem();
      $("relayWorkspace").dataset.view = item ? "result" : "capture";
      $("relayResult").hidden = !item;
      const primary = relayPrimaryState(item);
      $("relayStatusBox").className = `capture-status ${primary.key}`;
      $("relayStatusIcon").className = `status-icon ${primary.key}`;
      $("relayStatusIcon").innerHTML = ICONS[primary.icon] || "";
      $("relayStatusText").textContent = primary.label;
      $("relayStatusHint").textContent = primary.hint || "";
      const ptt = $("relayPushToTalk").checked;
      $("relayPttButton").setAttribute("aria-label", ptt ? "Hold to talk" : "Start listening");
      $("relayPttButton").setAttribute("aria-pressed", relayPushHeld ? "true" : "false");
      const active = item && ACTIVE_STATES.has(item.status) ? item : (!item ? relayCurrentItem : null);
      $("relayCancelCurrent").classList.toggle("visible", Boolean(relayCurrentItem && active === relayCurrentItem));
      if (item) renderRelayResult(item);
    }

    function renderRelayResult(item) {
      $("relayTranscript").textContent = item.text;
      const hasAudio = Boolean(item.url);
      $("relayPlayerBlock").hidden = !hasAudio;
      if (hasAudio) {
        const audio = $("relayAudio");
        const mismatch = relayLoadedUrl !== item.url;
        if (mismatch && (audio.paused || audio.ended)) {
          // Nothing is playing, so load this item's audio without starting it.
          relayLoadedUrl = item.url;
          audio.src = item.url;
          $("relayAudio").hidden = false;
          $("relayPlayThis").hidden = true;
        } else {
          // Another line is playing through the shared player; offer to switch instead of showing its controls here.
          $("relayAudio").hidden = mismatch;
          $("relayPlayThis").hidden = !mismatch;
        }
      }
      const actions = $("relayResultActions");
      actions.innerHTML = "";
      if (["transcribed", "failed", "cancelled"].includes(item.status)) {
        const speak = document.createElement("button");
        speak.type = "button";
        speak.className = "mini-button";
        speak.textContent = item.status === "transcribed" ? "Speak this line" : "Try again";
        speak.onclick = () => requeueRelayItem(item);
        actions.appendChild(speak);
      }
      const meta = [];
      if (item.status === "transcribed") meta.push("Transcribed only. Speak automatically is off.");
      else if (item.status === "cancelled") meta.push("Cancelled before playback.");
      else if (item.status === "completed") meta.push("Spoken");
      if (item.name) meta.push(item.name);
      meta.push(formatClock(item.createdAt));
      $("relayNowPlaying").textContent = meta.filter(Boolean).join(" · ");
    }

    function setRelayLevel(db) {
      if (++relayLevelTick % 2) return;
      const level = Math.max(0, Math.min(1, (db + 60) / 50));
      $("relayPttButton").style.setProperty("--level", level.toFixed(2));
    }

    async function ensureRelayApiReady() {
      if (state && state.api_ready) return;
      relayStatus("starting TTS engine");
      await post("/api/start-api", {gpt: $("relayGptModel").value, sovits: $("relaySovitsModel").value});
      for (let i = 0; i < 80; i++) {
        await new Promise(resolve => setTimeout(resolve, 1500));
        await refresh();
        if (state && state.api_ready) return;
      }
      throw new Error("TTS engine did not become ready.");
    }

    async function generateRelayLine(text, generationId) {
      const ref = relayRef();
      if (!ref) throw new Error("Pick a relay reference clip.");
      const aux = relayAuxRefs().map(item => item.wsl_path);
      return await post("/api/generate", {
        gpt: $("relayGptModel").value,
        sovits: $("relaySovitsModel").value,
        ref_audio_path: ref.wsl_path,
        aux_ref_audio_paths: aux,
        prompt_text: ref.text,
        prompt_lang: ref.lang,
        text,
        // Speech recognition is English only, so relay lines are English.
        text_lang: "en",
        text_split_method: "cut5",
        seed: -1,
        top_k: $("relayTopK").value,
        top_p: $("relayTopP").value,
        temperature: $("relayTemperature").value,
        speed_factor: $("relaySpeed").value,
        fragment_interval: $("relayPause").value,
        repetition_penalty: $("relayRepPenalty").value,
        streaming_mode: 3,
        overlap_length: 2,
        min_chunk_length: 8,
        generation_id: generationId,
        voice: voiceDataset,
        // Relay lines are played once; the server keeps only the newest ones.
        relay: true
      });
    }

    function playRelayUrl(url, label) {
      relayLastUrl = url;
      relayLoadedUrl = url;
      const audio = $("relayAudio");
      audio.pause();
      audio.src = url + "?t=" + Date.now();
      applyRelayOutputDevice();
      updateRelayVolumeLabel();
      renderRelayCenter();
      const started = audio.play();
      if (started && typeof started.catch === "function") {
        started.catch(() => relayStatus("press play on output"));
      }
      if ($("relayMonitorLocal").checked && $("relayOutputDevice").value) {
        const monitor = document.getElementById("relayLocalMonitor") || document.createElement("audio");
        monitor.id = "relayLocalMonitor";
        monitor.style.display = "none";
        monitor.volume = audio.volume;
        monitor.src = url + "?monitor=" + Date.now();
        if (!monitor.parentElement) document.body.appendChild(monitor);
        const monitorStarted = monitor.play();
        if (monitorStarted && typeof monitorStarted.catch === "function") monitorStarted.catch(() => {});
      }
    }

    function newRelayItem(text, status) {
      const clean = (text || "").replace(/\s+/g, " ").trim();
      if (!clean) return null;
      const item = {id: relayNextId++, generationId: `relay-${Date.now()}-${relayNextId}`, text: clean, status, url: "", name: "", error: "", createdAt: Date.now()};
      relayQueue.push(item);
      relayView = {mode: "result", itemId: item.id};
      return item;
    }

    function enqueueRelayLine(text) {
      const item = newRelayItem(text, "queued");
      if (!item) return;
      renderRelayQueue();
      relayStatus("line queued");
      processRelayQueue();
    }

    // Used for "Speak this line" (transcribed while Speak automatically was off) and "Try again" after a failure.
    function requeueRelayItem(item) {
      item.status = "queued";
      item.url = "";
      item.name = "";
      item.error = "";
      item.generationId = `relay-${Date.now()}-${item.id}`;
      relayView = {mode: "result", itemId: item.id};
      renderRelayQueue();
      relayStatus("line queued");
      processRelayQueue();
    }

    async function processRelayQueue() {
      if (relayBusy) return;
      relayBusy = true;
      try {
        while (relayQueue.some(item => item.status === "queued")) {
          const item = relayQueue.find(row => row.status === "queued");
          relayCurrentItem = item;
          item.status = "generating";
          $("relayCancelCurrent").classList.add("visible");
          renderRelayQueue();
          relayStatus("generating");
          try {
            await ensureRelayApiReady();
            const result = await generateRelayLine(item.text, item.generationId);
            if (item.status === "cancelled") continue;
            item.status = "ready";
            item.url = result.url;
            item.name = result.name;
            renderRelayQueue();
            item.status = "playing";
            renderRelayQueue();
            relayStatus("playing generated line");
            playRelayUrl(item.url, item.name);
            await new Promise(resolve => {
              relayPlaybackResolve = resolve;
              const audio = $("relayAudio");
              audio.onended = resolve;
              audio.onerror = resolve;
            });
            relayPlaybackResolve = null;
            if (item.status !== "cancelled") item.status = "completed";
            renderRelayQueue();
          } catch (error) {
            if (item.status !== "cancelled") {
              item.status = "failed";
              item.error = error.message;
              relayStatus("relay failed");
              showError(error.message);
            }
            renderRelayQueue();
          } finally {
            relayCurrentItem = null;
            $("relayCancelCurrent").classList.remove("visible");
            renderRelayCenter();
          }
        }
        if (!relayListening) relayStatus("idle");
      } finally {
        relayBusy = false;
      }
    }

    function mergeFloatChunks(chunks, frameCount) {
      const merged = new Float32Array(frameCount);
      let offset = 0;
      chunks.forEach(chunk => {
        merged.set(chunk, offset);
        offset += chunk.length;
      });
      return merged;
    }

    function encodeWav(samples, sampleRate) {
      const bytesPerSample = 2;
      const dataSize = samples.length * bytesPerSample;
      const buffer = new ArrayBuffer(44 + dataSize);
      const view = new DataView(buffer);
      const writeString = (offset, value) => {
        for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
      };
      writeString(0, "RIFF");
      view.setUint32(4, 36 + dataSize, true);
      writeString(8, "WAVE");
      writeString(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * bytesPerSample, true);
      view.setUint16(32, bytesPerSample, true);
      view.setUint16(34, 16, true);
      writeString(36, "data");
      view.setUint32(40, dataSize, true);
      let offset = 44;
      for (let i = 0; i < samples.length; i++, offset += 2) {
        const sample = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      }
      return new Blob([buffer], {type: "audio/wav"});
    }

    function blobToBase64(blob) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    }

    async function ensureRelayAsrReady() {
      if (state && state.relay_asr_ready) return;
      relayStatus("starting local speech");
      await post("/api/start-relay-asr");
      for (let i = 0; i < 80; i++) {
        await new Promise(resolve => setTimeout(resolve, 500));
        await refresh();
        if (state && state.relay_asr_ready) return;
      }
      throw new Error("Local speech recognizer did not become ready.");
    }

    async function transcribeRelayAudio(samples, sampleRate) {
      const wav = encodeWav(samples, sampleRate);
      const audio_base64 = await blobToBase64(wav);
      const result = await post("/api/transcribe", {audio_base64});
      return (result.text || "").replace(/\s+/g, " ").trim();
    }

    function resetRelayCapture() {
      relayCapturing = false;
      relayCurrentChunks = [];
      relayCurrentFrames = 0;
      relayLastVoiceAt = 0;
    }

    function beginRelayPush() {
      if (!$("relayPushToTalk").checked) return;
      if (!relayListening) {
        try { startRelayListening(); }
        catch (e) { alert(e.message); return; }
      }
      if (!relayAudioCtx) {
        relayStatus("starting local speech");
        return;
      }
      if (relayPushHeld) return;
      resetRelayCapture();
      relayPushHeld = true;
      relayCapturing = true;
      $("relayPttButton").classList.add("active");
      relayStatus("recording push-to-talk");
    }

    function endRelayPush() {
      if (!relayPushHeld) return;
      relayPushHeld = false;
      $("relayPttButton").classList.remove("active");
      finishRelayCapture();
      if (relayListening && !relayBusy) relayStatus("push-to-talk ready");
    }

    function finishRelayCapture() {
      if (!relayCurrentFrames) {
        resetRelayCapture();
        return;
      }
      const frames = relayCurrentFrames;
      const chunks = relayCurrentChunks.slice();
      resetRelayCapture();
      const minFrames = Math.floor((relayAudioCtx ? relayAudioCtx.sampleRate : 48000) * 0.45);
      if (frames < minFrames) {
        relayStatus("recording too short");
        return;
      }
      const samples = mergeFloatChunks(chunks, frames);
      const sampleRate = relayAudioCtx ? relayAudioCtx.sampleRate : 48000;
      relayTranscribing = true;
      relayStatus("transcribing locally");
      transcribeRelayAudio(samples, sampleRate)
        .then(text => {
          relayTranscribing = false;
          if (!text) {
            relayStatus("no speech recognized");
            return;
          }
          if ($("relayAutoSpeak").checked) {
            enqueueRelayLine(text);
          } else {
            newRelayItem(text, "transcribed");
            renderRelayQueue();
            relayStatus("transcribed");
          }
        })
        .catch(error => {
          relayTranscribing = false;
          relayStatus("transcription failed");
          console.error(error);
        });
    }

    function startRelayListening() {
      if (!navigator.mediaDevices) throw new Error("Browser microphone capture is unavailable.");
      if (relayListening && relayAudioCtx) {
        relayStatus("listening local");
        return;
      }
      relayListening = true;
      relayStatus("starting local speech");
      ensureRelayAsrReady().then(async () => {
        relayStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: {ideal: 1},
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          }
        });
        relayAudioCtx = new AudioContext({latencyHint: "interactive"});
        relaySource = relayAudioCtx.createMediaStreamSource(relayStream);
        relayProcessor = relayAudioCtx.createScriptProcessor(4096, 1, 1);
        relayKeepAlive = relayAudioCtx.createGain();
        relayKeepAlive.gain.value = 0;
        relayProcessor.onaudioprocess = event => {
          if (!relayListening) return;
          const input = event.inputBuffer.getChannelData(0);
          const db = dbFromChunk(input);
          relayLastLevelDb = db === null ? -120 : db;
          const now = performance.now();
          if ($("relayPushToTalk").checked) {
            if (relayPushHeld) {
              if (!relayCapturing) {
                relayCapturing = true;
                relayCurrentChunks = [];
                relayCurrentFrames = 0;
              }
              relayCurrentChunks.push(new Float32Array(input));
              relayCurrentFrames += input.length;
              setRelayLevel(relayLastLevelDb);
            }
            return;
          }
          const isVoice = relayLastLevelDb > -48;
          if (isVoice && !relayCapturing) {
            relayCapturing = true;
            relayLastVoiceAt = now;
            $("relayPttButton").classList.add("active");
            renderRelayCenter();
          }
          if (relayCapturing) {
            relayCurrentChunks.push(new Float32Array(input));
            relayCurrentFrames += input.length;
            if (isVoice) relayLastVoiceAt = now;
            const capturedMs = relayCurrentFrames / relayAudioCtx.sampleRate * 1000;
            setRelayLevel(relayLastLevelDb);
            if ((now - relayLastVoiceAt > 850 && capturedMs > 700) || capturedMs > 9000) {
              $("relayPttButton").classList.remove("active");
              finishRelayCapture();
              renderRelayCenter();
            }
          }
        };
        relaySource.connect(relayProcessor);
        relayProcessor.connect(relayKeepAlive);
        relayKeepAlive.connect(relayAudioCtx.destination);
        if (relayAudioCtx.state === "suspended") await relayAudioCtx.resume();
        relayStatus($("relayPushToTalk").checked ? "push-to-talk ready" : "listening local");
      }).catch(error => {
        relayListening = false;
        relayStatus("local speech failed");
        alert(error.message);
      });
    }

    function stopRelayListening() {
      relayListening = false;
      relayPushHeld = false;
      $("relayPttButton").classList.remove("active");
      if (relayProcessor) relayProcessor.disconnect();
      relayProcessor = null;
      if (relaySource) relaySource.disconnect();
      relaySource = null;
      if (relayKeepAlive) relayKeepAlive.disconnect();
      relayKeepAlive = null;
      if (relayStream) relayStream.getTracks().forEach(track => track.stop());
      relayStream = null;
      if (relayAudioCtx) relayAudioCtx.close();
      relayAudioCtx = null;
      finishRelayCapture();
      relayStatus(relayBusy ? "generating" : "not listening");
    }

    function updateSystemView(system) {
      if (!system) return;
      const status = system.status || "stopped";
      lastSystemSnapshot = system;
      $("systemChip").textContent = status === "ready" ? "System ready" : (status === "stopped" ? "System stopped" : (system.message || status));
      $("systemChip").title = system.error ? `${system.message} ${system.error}` : (system.message || status);
      $("systemChip").className = status === "ready" ? "island-system-state good" : (status === "failed" || status === "degraded" ? "island-system-state bad" : "island-system-state");
      $("startSystem").disabled = status === "starting" || status === "ready";
      if (status !== lastSystemStatus) {
        lastSystemStatus = status;
        renderRelayCenter();
      }
      $("startSystem").textContent = status === "failed" || status === "degraded" ? "Retry System" : "Start System";
      // Stop Everything stays clickable: services can be running even while the
      // system status reads "stopped", and stop-system is safe to call at any time.
      const components = system.components || {};
      componentCard("relayAsrCard", "relayAsrStatus", status === "starting" && system.phase === "asr" ? "starting" : components.asr);
      componentCard("relayTtsCard", "relayTtsStatus", status === "starting" && system.phase === "tts" ? "starting" : components.tts);
      componentCard("relayPttCard", "relayPttStatus", status === "starting" && system.phase === "ptt" ? "starting" : components.ptt);
      if (system.message && system.message !== lastSystemMessage) {
        lastSystemMessage = system.message;
        recentActivity(system.error ? `${system.message} ${system.error}` : system.message);
        if (system.error) showError(`${system.message} ${system.error}`);
      }
      if (status === "ready" && armRelayWhenReady && !relayListening) {
        armRelayWhenReady = false;
        startRelayListening();
      }
    }

    async function refreshSystem() {
      try {
        const response = await fetch("/api/system-state");
        const data = await response.json();
        updateSystemView(data.system);
      } catch (_) {
        $("systemChip").textContent = "dashboard connection lost";
        $("systemChip").title = "dashboard connection lost";
        $("systemChip").className = "chip bad";
      }
    }

    async function runSetupCheck() {
      $("setupSummary").textContent = "Checking local components...";
      const list = $("setupList");
      list.innerHTML = "";
      try {
        const response = await fetch("/api/setup-status");
        const checks = await response.json();
        try {
          const permissionStream = await navigator.mediaDevices.getUserMedia({audio: true});
          permissionStream.getTracks().forEach(track => track.stop());
          const devices = await navigator.mediaDevices.enumerateDevices();
          const microphones = devices.filter(device => device.kind === "audioinput");
          const outputs = devices.filter(device => device.kind === "audiooutput");
          const cableDevices = devices.filter(device => /VB-Audio|CABLE Input|CABLE Output/i.test(device.label || ""));
          checks.microphone = {status: microphones.length ? "ready" : "missing", detail: `${microphones.length} microphone input(s) detected`};
          checks.audio_output = {status: outputs.length ? "ready" : "missing", detail: `${outputs.length} audio output(s) detected`};
          checks.vb_cable = {status: cableDevices.length ? "ready" : "missing", detail: cableDevices.length ? cableDevices.map(device => device.label).join(", ") : "VB-CABLE was not found in browser audio devices."};
        } catch (error) {
          checks.microphone = {status: "needs_configuration", detail: "Allow microphone access to complete this check."};
          checks.audio_output = {status: "needs_configuration", detail: "Allow audio-device access to complete this check."};
        }
        const names = {
          config: "config.json",
          dashboard_python: "Dashboard Python",
          wsl_files: "GPT-SoVITS / WSL",
          gpt_models: "GPT models",
          sovits_models: "SoVITS models",
          asr_environment: "Speech recognition",
          asr_model_cache: "ASR model cache",
          ptt_helper: "Global PTT helper",
          gpu: "Voice engine device",
          microphone: "Microphone",
          audio_output: "Audio outputs",
          vb_cable: "VB-CABLE / audio"
        };
        let ready = 0;
        Object.entries(checks).forEach(([key, check]) => {
          if (check.status === "ready") ready += 1;
          const row = document.createElement("div");
          row.className = "setup-row";
          const name = document.createElement("strong");
          name.textContent = names[key] || key;
          const status = document.createElement("span");
          status.className = `setup-state ${check.status}`;
          status.textContent = check.status.replaceAll("_", " ");
          const detail = document.createElement("span");
          detail.className = "small";
          detail.textContent = check.detail || "";
          row.append(name, status, detail);
          list.appendChild(row);
        });
        $("setupSummary").textContent = `${ready} of ${Object.keys(checks).length} checks ready.`;
      } catch (error) {
        $("setupSummary").textContent = "System check failed.";
        showError(error.message);
      }
    }

    async function refresh() {
      if (refreshBusy) return;
      refreshBusy = true;
      try {
      const res = await fetch(`/api/state?dataset=${encodeURIComponent(voiceDataset)}`);
      state = await res.json();
      if (state.config_error && !configErrorShown) {
        configErrorShown = true;
        showError(state.config_error);
      }
      fillTrainingDatasets(state.datasets || []);
      fillDatasetSelects(state.datasets || [], state.reference_dataset || "");
      const gptOld = $("gptModel").value;
      const sovitsOld = $("sovitsModel").value;
      const relayGptOld = $("relayGptModel").value || relaySavedSettings.relayGptModel || "";
      const relaySovitsOld = $("relaySovitsModel").value || relaySavedSettings.relaySovitsModel || "";
      const nextModelsSignature = [...state.models.gpt, ...state.models.sovits].map(row => `${row.path}:${row.mtime}`).join("|");
      if (nextModelsSignature !== modelsSignature) {
        modelsSignature = nextModelsSignature;
        fillSelect($("gptModel"), state.models.gpt, gptOld);
        fillSelect($("sovitsModel"), state.models.sovits, sovitsOld);
        fillSelect($("relayGptModel"), state.models.gpt, relayGptOld);
        fillSelect($("relaySovitsModel"), state.models.sovits, relaySovitsOld);
      }
      const nextRefsSignature = (state.reference_dataset || "") + "|" + state.all_references.map(ref => `${ref.id}:${ref.duration || ""}:${ref.valid_reference}:${ref.valid_aux_reference}`).join("|");
      if (nextRefsSignature !== refsSignature) {
        refsSignature = nextRefsSignature;
        fillReferences(state.references);
        fillRelayReferences(state.references);
        renderRelayAuxRefs();
      }
      if (!engineBusy) {
        $("engineStatus").textContent = state.api_ready ? "Engine ready" : "Engine off";
        $("engineStatus").className = state.api_ready ? "status-line good" : "status-line";
      }
      updateSystemView(state.system);
      const preparation = state.jobs && state.jobs["dataset-prepare"];
      if (preparation) {
        $("datasetPrepareStatus").textContent = preparation.status === "running" ? `${preparation.name} is running...` : `${preparation.name}: ${preparation.status}`;
        $("datasetPrepareStatus").className = preparation.status === "failed" ? "status-line bad" : (preparation.status === "done" ? "status-line good" : "status-line");
        $("prepareDataset").disabled = preparation.status === "running";
      }
      const jobLine = (key, element) => {
        const job = state.jobs && state.jobs[key];
        if (!job) { $(element).textContent = ""; return; }
        const text = job.status === "running" ? "Running..." : `${job.status}${job.returncode !== null && job.returncode !== undefined ? ` (exit ${job.returncode})` : ""}`;
        $(element).textContent = text;
        $(element).className = job.status === "failed" ? "status-line bad" : (job.status === "done" ? "status-line good" : "status-line");
      };
      jobLine("train-sovits", "trainSovitsStatus");
      jobLine("train-gpt", "trainGptStatus");
      const trainingJob = ["dataset-sync", "dataset-prepare", "train-sovits", "train-gpt"].map(key => state.jobs[key]).find(job => job && job.status === "running");
      $("trainingJobsStatus").textContent = trainingJob ? `${trainingJob.name} is running` : "";
      const jobLogs = Object.values(state.jobs || {})
        .filter(job => job.lines && job.lines.length)
        .map(job => [`--- ${job.name}: ${job.status} ---`, ...job.lines].join("\n"))
        .join("\n\n");
      logRaw = [state.log || "", jobLogs].filter(Boolean).join("\n\n");
      renderLog();
      $("modelList").innerHTML = "";
      const savedModels = [...state.models.gpt, ...state.models.sovits];
      $("modelCount").textContent = savedModels.length ? `${savedModels.length} files` : (state.models_listed ? "none yet" : "listed once WSL is running");
      savedModels.forEach(row => $("modelList").appendChild(modelLine(row.path, row.size_mb)));
      } finally {
        refreshBusy = false;
      }
    }

    function modelLine(label, sizeMb) {
      const div = document.createElement("div");
      div.className = "model-line";
      const name = document.createElement("span");
      name.textContent = label;
      const size = document.createElement("span");
      size.textContent = `${sizeMb} MB`;
      div.append(name, size);
      return div;
    }

    async function post(path, data = {}) {
      const res = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
      const json = await res.json();
      if (!res.ok || json.ok === false) throw new Error(json.error || "Request failed");
      return json;
    }

    $("reference").addEventListener("change", () => {
      const ref = state.references.find(r => r.id === $("reference").value);
      if (ref) setReference(ref, true);
      renderAuxRefs(state.all_references);
    });
    $("auxSearch").addEventListener("input", () => {
      if (state) renderAuxRefs(state.all_references);
    });
    $("clearAuxRefs").onclick = () => {
      auxSelected.clear();
      if (state) renderAuxRefs(state.all_references);
    };

    // Starts the voice engine with the Generate page's model selection and waits until it accepts connections.
    async function ensureGenerateEngine() {
      if (state && state.api_ready) return true;
      engineBusy = true;
      $("engineStatus").className = "status-line warn";
      $("engineStatus").textContent = "Starting the voice engine, this can take a minute or two...";
      try {
        await post("/api/start-api", {gpt: $("gptModel").value, sovits: $("sovitsModel").value});
        // The server replies immediately after spawning the engine; poll until it accepts connections.
        for (let i = 0; i < 160; i++) {
          await new Promise(resolve => setTimeout(resolve, 1500));
          await refresh();
          if (state && state.api_ready) break;
        }
        const ready = Boolean(state && state.api_ready);
        $("engineStatus").className = ready ? "status-line good" : "status-line bad";
        $("engineStatus").textContent = ready ? "Engine ready" : "Engine did not become ready. Check the Logs page.";
        return ready;
      } catch (e) {
        $("engineStatus").className = "status-line bad";
        $("engineStatus").textContent = "Engine failed to start.";
        throw e;
      } finally {
        engineBusy = false;
      }
    }
    $("startApi").onclick = async () => {
      $("startApi").disabled = true;
      try { await ensureGenerateEngine(); }
      catch (e) { alert(e.message); }
      finally { $("startApi").disabled = false; refresh(); }
    };
    $("stopApi").onclick = async () => { await post("/api/stop-api"); $("engineStatus").className = "status-line"; $("engineStatus").textContent = "Engine stopped"; refresh(); };

    function renderLog() {
      const filter = $("logFilter").value.trim().toLowerCase();
      const lines = logRaw ? logRaw.split("\n") : [];
      const shown = filter ? lines.filter(line => line.toLowerCase().includes(filter)) : lines;
      const log = $("log");
      log.textContent = shown.join("\n");
      $("logStatus").textContent = filter ? `${shown.length} of ${lines.length} lines` : "";
      if ($("logFollow").checked) log.scrollTop = log.scrollHeight;
    }
    $("logFilter").addEventListener("input", renderLog);
    $("logFollow").addEventListener("change", renderLog);
    $("copyLog").onclick = async () => {
      try {
        await navigator.clipboard.writeText(logRaw);
        $("logStatus").textContent = "Copied.";
      } catch (error) {
        $("logStatus").textContent = "Copy failed. Select the text and copy it manually.";
      }
    };

    function renderGenerateHistory() {
      const holder = $("generateHistory");
      holder.innerHTML = "";
      const earlier = generateOutputs.slice(0, -1).reverse();
      holder.hidden = !earlier.length;
      earlier.forEach(item => {
        const row = document.createElement("div");
        row.className = "history-line";
        const text = document.createElement("span");
        text.textContent = item.text;
        text.title = `${item.name}: ${item.text}`;
        const play = document.createElement("button");
        play.type = "button";
        play.className = "secondary mini-button";
        play.textContent = "Play";
        play.onclick = () => {
          $("outputAudio").src = item.url + "?t=" + Date.now();
          $("outputName").textContent = item.name;
          $("outputAudio").play().catch(() => {});
        };
        row.append(text, play);
        holder.appendChild(row);
      });
    }
    $("relayReference").addEventListener("change", () => {
      const ref = relayRef();
      if (ref) setRelayReference(ref);
      renderRelayAuxRefs();
    });
    $("relayAuxSearch").addEventListener("input", renderRelayAuxRefs);
    $("relayClearAuxRefs").onclick = () => {
      relayAuxSelected.clear();
      renderRelayAuxRefs();
    };
    $("relayVolume").addEventListener("input", updateRelayVolumeLabel);
    $("refreshRelayDevices").onclick = refreshRelayDevices;
    $("relayOutputDevice").addEventListener("change", applyRelayOutputDevice);
    $("relaySetPttKey").onclick = () => startPttBinding();
    $("relayPushToTalk").addEventListener("change", () => {
      if (!$("relayPushToTalk").checked) endRelayPush();
      relayStatus($("relayPushToTalk").checked ? "push-to-talk ready" : "listening local");
    });
    $("relayAutoSpeak").addEventListener("change", renderRelayCenter);
    $("relayPttButton").addEventListener("pointerdown", event => {
      event.preventDefault();
      // Keep pointer events on the button while held, even if the pointer drifts off it.
      try { event.currentTarget.setPointerCapture(event.pointerId); } catch (_) {}
      beginRelayPush();
    });
    // Voice-activated mode: the button starts the microphone, since nothing is held.
    $("relayPttButton").addEventListener("click", () => {
      if ($("relayPushToTalk").checked || relayListening) return;
      try { startRelayListening(); }
      catch (error) { showError(error.message); }
    });
    // Keyboard users can hold Space or Enter on the focused button, mirroring pointer hold and release.
    $("relayPttButton").addEventListener("keydown", event => {
      if (event.code !== "Space" && event.code !== "Enter") return;
      if (event.code === $("relayPttKey").value) return;
      event.preventDefault();
      if (event.repeat) return;
      if (!$("relayPushToTalk").checked) {
        if (!relayListening) $("relayPttButton").click();
        return;
      }
      beginRelayPush();
    });
    $("relayPttButton").addEventListener("keyup", event => {
      if (event.code !== "Space" && event.code !== "Enter") return;
      if (event.code === $("relayPttKey").value) return;
      event.preventDefault();
      endRelayPush();
    });
    $("relayNewRecording").onclick = () => {
      showCaptureView();
      $("relayPttButton").focus();
    };
    $("historyToggle").onclick = () => {
      const open = $("relayHistory").classList.toggle("open");
      $("historyToggle").setAttribute("aria-expanded", open ? "true" : "false");
    };
    $("openRelaySettings").onclick = () => $("relaySettingsDialog").showModal();
    $("relayPlayThis").onclick = () => {
      const item = viewedRelayItem();
      if (item && item.url) playRelayUrl(item.url, item.name);
    };
    ["pointerup", "pointercancel", "pointerleave"].forEach(name => {
      $("relayPttButton").addEventListener(name, event => {
        event.preventDefault();
        endRelayPush();
      });
    });
    document.addEventListener("keydown", event => {
      if (relayBindingPtt) {
        event.preventDefault();
        if (event.code === "Escape") {
          cancelPttBinding();
          return;
        }
        relayBindingPtt = false;
        setRelayPttKey(event.code);
        relayStatus(`push-to-talk key: ${describePttKey(event.code)}`);
        return;
      }
      const tag = (event.target && event.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "select" || tag === "textarea") return;
      if (event.code === $("relayPttKey").value && !event.repeat) {
        event.preventDefault();
        beginRelayPush();
      }
    });

    $("startSystem").onclick = async () => {
      $("startSystem").disabled = true;
      $("globalError").classList.remove("visible");
      saveRelaySettings();
      armRelayWhenReady = true;
      recentActivity("Requesting microphone access...");
      try {
        await refreshRelayDevices();
        await post("/api/start-system", {
          gpt: $("relayGptModel").value,
          sovits: $("relaySovitsModel").value,
          binding: $("relayPttKey").value
        });
        recentActivity("System startup requested.");
        await refreshSystem();
      } catch (error) {
        armRelayWhenReady = false;
        showError(error.message);
        $("startSystem").disabled = false;
      }
    };

    $("stopSystem").onclick = async () => {
      $("stopSystem").disabled = true;
      armRelayWhenReady = false;
      stopRelayListening();
      try {
        await post("/api/stop-system");
      } catch (error) {
        showError(error.message);
      } finally {
        $("stopSystem").disabled = false;
        await refreshSystem();
      }
    };

    $("runSetupCheck").onclick = runSetupCheck;

    RELAY_SETTING_IDS.forEach(id => {
      const element = $(id);
      if (!element) return;
      element.addEventListener(element.type === "range" ? "input" : "change", saveRelaySettings);
    });

    const relayProfiles = {
      fast: {relayTopK: "10", relayTopP: "0.5", relayTemperature: "0.45", relaySpeed: "1", relayPause: "0.15", relayRepPenalty: "1.4"},
      balanced: {relayTopK: "15", relayTopP: "0.6", relayTemperature: "0.55", relaySpeed: "0.95", relayPause: "0.25", relayRepPenalty: "1.35"},
      quality: {relayTopK: "20", relayTopP: "0.7", relayTemperature: "0.6", relaySpeed: "0.95", relayPause: "0.3", relayRepPenalty: "1.3"}
    };
    document.querySelectorAll("[data-relay-profile]").forEach(button => {
      button.onclick = () => {
        Object.entries(relayProfiles[button.dataset.relayProfile] || {}).forEach(([id, value]) => { $(id).value = value; });
        saveRelaySettings();
        recentActivity(`${button.textContent.trim()} preset applied.`);
      };
    });

    const pttEvents = new EventSource("/api/events");
    pttEvents.addEventListener("ptt", event => {
      try {
        const ptt = JSON.parse(event.data);
        if (ptt.held) beginRelayPush();
        else endRelayPush();
      } catch (error) {
        console.error(error);
      }
    });
    document.addEventListener("keyup", event => {
      if (event.code === $("relayPttKey").value) {
        event.preventDefault();
        endRelayPush();
      }
    });
    document.addEventListener("mousedown", event => {
      if (relayBindingPtt) {
        event.preventDefault();
        if (event.target && event.target.closest && event.target.closest("#relaySetPttKey")) {
          // Clicking the shortcut control again cancels capture instead of binding that mouse button.
          cancelPttBinding();
          return;
        }
        relayBindingPtt = false;
        const code = pttMouseCode(event);
        setRelayPttKey(code);
        relayStatus(`push-to-talk key: ${describePttKey(code)}`);
        return;
      }
      const tag = (event.target && event.target.tagName || "").toLowerCase();
      if (event.button < 3 && (tag === "input" || tag === "select" || tag === "textarea" || tag === "button")) return;
      if (pttMouseCode(event) === $("relayPttKey").value) {
        event.preventDefault();
        beginRelayPush();
      }
    });
    document.addEventListener("mouseup", event => {
      if (pttMouseCode(event) === $("relayPttKey").value) {
        event.preventDefault();
        endRelayPush();
      }
    });
    document.addEventListener("auxclick", event => {
      if (relayBindingPtt || pttMouseCode(event) === $("relayPttKey").value) event.preventDefault();
    });
    document.addEventListener("contextmenu", event => {
      if (relayBindingPtt || $("relayPttKey").value === "Mouse2") event.preventDefault();
    });
    setRelayPttKey(localStorage.getItem("voiceDashboardRelayPttKey") || $("relayPttKey").value);
    updateRelayVolumeLabel();
    autoLoadRelayDevices();
    $("relaySpeakManual").onclick = () => enqueueRelayLine($("relayManualText").value);
    $("relayClearQueue").onclick = () => {
      relayQueue = relayQueue.filter(item => item.status !== "queued");
      if (!viewedRelayItem()) relayView = {mode: "capture", itemId: null};
      renderRelayQueue();
      relayStatus(relayListening ? "push-to-talk ready" : "idle");
    };
    $("relayCancelCurrent").onclick = async () => {
      const item = relayCurrentItem;
      if (!item) return;
      item.status = "cancelled";
      renderRelayQueue();
      $("relayAudio").pause();
      if (relayPlaybackResolve) {
        relayPlaybackResolve();
        relayPlaybackResolve = null;
      }
      $("relayCancelCurrent").classList.remove("visible");
      relayStatus("current line cancelled");
      try { await post("/api/cancel-generation", {generation_id: item.generationId}); }
      catch (error) { showError(error.message); }
    };
    $("relayReplayLast").onclick = () => {
      const item = viewedRelayItem();
      if (item && item.url) playRelayUrl(item.url, item.name);
      else if (relayLastUrl) playRelayUrl(relayLastUrl, "replay");
    };
    renderRelayQueue();
    $("trainingDataset").onchange = () => {
      localStorage.setItem("voiceDashboardTrainingDataset", $("trainingDataset").value);
      datasetUploadFiles = [];
      $("datasetAudioFiles").value = "";
      renderDatasetUploads();
      fillTrainingDatasets(state ? state.datasets : []);
    };
    $("trainingModelName").value = localStorage.getItem("voiceDashboardTrainingModelName") || $("trainingModelName").value;
    const updateTrainingModelName = () => {
      const slug = modelSlug($("trainingModelName").value);
      $("trainingModelSlug").textContent = slug;
      localStorage.setItem("voiceDashboardTrainingModelName", $("trainingModelName").value);
    };
    $("trainingModelName").oninput = updateTrainingModelName;
    updateTrainingModelName();

    $("createDataset").onclick = async () => {
      $("createDataset").disabled = true;
      try {
        const result = await post("/api/create-dataset", {
          name: $("newDatasetName").value,
          speaker: $("newDatasetSpeaker").value,
          language: $("newDatasetLanguage").value
        });
        localStorage.setItem("voiceDashboardTrainingDataset", result.dataset.id);
        $("newDatasetName").value = "";
        recentActivity(`Dataset ${result.dataset.name} created.`);
        await refresh();
      } catch (error) {
        showError(error.message);
      } finally {
        $("createDataset").disabled = false;
      }
    };

    $("datasetAudioFiles").onchange = () => {
      datasetUploadFiles = Array.from($("datasetAudioFiles").files || []);
      renderDatasetUploads();
    };

    $("uploadDatasetAudio").onclick = async () => {
      if (!datasetUploadFiles.length) return;
      const transcripts = Array.from(document.querySelectorAll("[data-upload-transcript]")).map(input => input.value.trim());
      if (transcripts.some(text => !text)) return showError("Every selected audio file needs a transcript.");
      $("uploadDatasetAudio").disabled = true;
      try {
        for (let index = 0; index < datasetUploadFiles.length; index++) {
          const file = datasetUploadFiles[index];
          $("datasetUploadProgress").textContent = `Converting and adding ${index + 1} of ${datasetUploadFiles.length}: ${file.name}`;
          await post("/api/upload-dataset-audio", {
            dataset_id: $("trainingDataset").value,
            filename: file.name,
            audio_base64: await fileToBase64(file),
            text: transcripts[index],
            speaker: "",
            language: ""
          });
        }
        $("datasetUploadProgress").textContent = `${datasetUploadFiles.length} file(s) added successfully.`;
        datasetUploadFiles = [];
        $("datasetAudioFiles").value = "";
        renderDatasetUploads();
        await refresh();
      } catch (error) {
        showError(error.message);
      } finally {
        $("uploadDatasetAudio").disabled = !datasetUploadFiles.length;
      }
    };

    $("syncDataset").onclick = async () => {
      try {
        await post("/api/sync-dataset", {dataset_id: $("trainingDataset").value});
        $("datasetPrepareStatus").textContent = "Dataset sync started.";
        refresh();
      } catch (error) { showError(error.message); }
    };
    $("prepareDataset").onclick = async () => {
      $("prepareDataset").disabled = true;
      try {
        await post("/api/prepare-dataset", {dataset_id: $("trainingDataset").value, model_name: $("trainingModelName").value});
        $("datasetPrepareStatus").textContent = `Preparing features for ${modelSlug($("trainingModelName").value)}...`;
        refresh();
      } catch (error) { showError(error.message); }
      finally { $("prepareDataset").disabled = false; }
    };
    ["voiceDataset", "relayDataset"].forEach(id => $(id).addEventListener("change", () => {
      voiceDataset = $(id).value;
      try { localStorage.setItem("voiceDashboardVoiceDataset", voiceDataset); } catch (_) {}
      refsSignature = "";
      refresh();
    }));
    $("trainSovits").onclick = async () => {
      try {
        await post("/api/start-train", {kind: "sovits", dataset_id: $("trainingDataset").value, model_name: $("trainingModelName").value, epochs: $("sovitsEpochs").value, batch_size: $("sovitsBatch").value, save_every: $("sovitsSave").value});
        refresh();
      } catch (error) { showError(error.message); }
    };
    $("trainGpt").onclick = async () => {
      try {
        await post("/api/start-train", {kind: "gpt", dataset_id: $("trainingDataset").value, model_name: $("trainingModelName").value, epochs: $("gptEpochs").value, batch_size: $("gptBatch").value, save_every: $("gptSave").value});
        refresh();
      } catch (error) { showError(error.message); }
    };
    document.querySelectorAll("[data-stop]").forEach(btn => {
      btn.onclick = async () => { await post("/api/stop-job", {key: btn.dataset.stop}); refresh(); };
    });
    $("generate").onclick = async () => {
      const ref = state.references.find(r => r.id === $("reference").value);
      if (!ref) return alert("Pick a reference clip.");
      $("generate").disabled = true;
      $("generateStatus").className = "status-line";
      if (!(state && state.api_ready)) {
        $("generateStatus").textContent = "Starting the voice engine first...";
        try {
          if (!await ensureGenerateEngine()) throw new Error("The voice engine did not become ready.");
        } catch (e) {
          $("generateStatus").className = "status-line bad";
          $("generateStatus").textContent = "Engine failed to start.";
          $("generate").disabled = false;
          return alert(e.message);
        }
      }
      $("generateStatus").textContent = "Generating...";
      const aux = Array.from(auxSelected)
        .map(id => state.all_references.find(r => r.id === id))
        .filter(Boolean)
        .filter(r => r.valid_aux_reference)
        .filter(r => r.id !== ref.id)
        .map(r => r.wsl_path);
      try {
        const result = await post("/api/generate", {
          gpt: $("gptModel").value,
          sovits: $("sovitsModel").value,
          ref_audio_path: ref.wsl_path,
          aux_ref_audio_paths: aux,
          prompt_text: $("promptText").value,
          prompt_lang: ref.lang,
          text: $("targetText").value,
          text_lang: $("textLang").value,
          text_split_method: $("splitMethod").value,
          seed: $("seed").value,
          top_k: $("topK").value,
          top_p: $("topP").value,
          temperature: $("temperature").value,
          speed_factor: $("speed").value,
          fragment_interval: $("pause").value,
          repetition_penalty: $("repPenalty").value,
          voice: voiceDataset
        });
        $("outputAudio").src = result.url + "?t=" + Date.now();
        $("outputName").textContent = result.name;
        $("generateResult").hidden = false;
        generateOutputs.push({text: $("targetText").value.trim(), url: result.url, name: result.name});
        renderGenerateHistory();
        $("generateStatus").className = "status-line good";
        $("generateStatus").textContent = "Done.";
      } catch (e) {
        $("generateStatus").className = "status-line bad";
        $("generateStatus").textContent = "Failed.";
        alert(e.message);
      } finally {
        $("generate").disabled = false;
        refresh();
      }
    };

    refresh();
    refreshSystem();
    setInterval(refresh, 5000);
    setInterval(refreshSystem, 750);
  </script>
</body>
</html>
"""


def main():
    port = DASHBOARD_PORT
    log(f"Local Voice UI starting on http://localhost:{port}")
    if CONFIG_ERROR:
        log(CONFIG_ERROR)
    server = DashboardServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for key in ("ptt-helper", "relay-asr"):
            proc = JOBS.get(key)
            if proc and proc.process and proc.process.poll() is None:
                proc.stop()
        if API_PROCESS and API_PROCESS.process and API_PROCESS.process.poll() is None:
            stop_api()


if __name__ == "__main__":
    main()
