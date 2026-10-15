import array
import json
import base64
import contextlib
import ctypes
import hashlib
import hmac
import io
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import wave
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def load_config(path):
    """Read config.json. Returns the settings and an error message for the page, or "" when there is none."""
    if not path.exists():
        return {}, "config.json is missing. Copy config.example.json to config.json and set the paths in it."
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return {}, f"config.json could not be read ({exc}). In Windows paths, write \\\\ or / instead of \\."
    if not isinstance(config, dict):
        return {}, "config.json must contain one JSON object."
    return config, ""


WSL_PATH_KEYS = ("wsl_gpt_sovits_root", "wsl_python", "wsl_datasets_root")


def wsl_paths_error(config):
    """Why the WSL paths in config.json cannot be used, or "". They have no defaults, because each
    install has its own Linux user name in them."""
    if any("YOUR_WSL_USER" in str(config.get(key)) for key in WSL_PATH_KEYS):
        return "In config.json, replace YOUR_WSL_USER with your Linux user name."
    unset = [key for key in WSL_PATH_KEYS if not str(config.get(key) or "").startswith("/")]
    if unset:
        return f"config.json must set {', '.join(unset)} to Linux paths that start with /, as in config.example.json."
    return ""


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
# Every action that needs the WSL paths stops with this message while they are unusable.
# A missing or unreadable config.json is itself the reason they are unset.
WSL_PATHS_ERROR = CONFIG_ERROR or wsl_paths_error(APP_CONFIG)
DASHBOARD_PORT, _port_error = config_port(APP_CONFIG, "dashboard_port", 8790)
RELAY_ASR_PORT, _asr_port_error = config_port(APP_CONFIG, "asr_port", 8792)
TTS_PORT, _tts_port_error = config_port(APP_CONFIG, "tts_port", 9880)
CONFIG_ERROR = " ".join(filter(None, [WSL_PATHS_ERROR, _port_error, _asr_port_error, _tts_port_error]))
# For running a second copy beside the installed one, for example while testing.
if os.environ.get("VOICE_DASHBOARD_PORT"):
    DASHBOARD_PORT = int(os.environ["VOICE_DASHBOARD_PORT"])
DATASETS_DIR = ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
# The page: index.html with its app.css and app.js. They are read for each request, so an edit
# shows when the page is reloaded, without a restart.
STATIC_DIR = ROOT / "static"
STATIC_TYPES = {"html": "text/html", "css": "text/css", "js": "text/javascript"}
CACHE_DIR = Path(APP_CONFIG.get("cache_dir") or ROOT / ".cache")
TMP_DIR = Path(APP_CONFIG.get("tmp_dir") or CACHE_DIR / "tmp")

WSL_DISTRO = APP_CONFIG.get("wsl_distro") or "Ubuntu-24.04"
WSL_GSV_ROOT = str(APP_CONFIG.get("wsl_gpt_sovits_root") or "")
WSL_PYTHON = str(APP_CONFIG.get("wsl_python") or "")
WSL_DATASETS_ROOT = str(APP_CONFIG.get("wsl_datasets_root") or "")
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
    "separate": "pkill -f '[s]eparate_audio.py' || true",
}
# Jobs that use the GPU, and with it the memory the voice engine also needs.
GPU_JOBS = ("dataset-prepare", "train-sovits", "train-gpt")
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 10.0
LANGUAGES = {"en", "zh", "ja", "ko", "yue"}
# SoVITS model versions. v2ProPlus has a larger decoder. Both share the GPT model and the features
# that preparation extracts, as their pretrained models have the same quantizer.
SOVITS_VERSIONS = ("v2Pro", "v2ProPlus")
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


def wsl_bash(script):
    # --exec hands the script to bash unchanged. With "--", wsl.exe first runs the command line
    # through the default shell, which expands the script's own variables before bash sets them.
    return ["wsl", "-d", WSL_DISTRO, "--exec", "bash", "-lc", script]


def wsl_command(script):
    """Run script in WSL with the voice engine's Python first on PATH."""
    if WSL_PATHS_ERROR:
        raise ValueError(WSL_PATHS_ERROR)
    python_dir = WSL_PYTHON.rsplit("/", 1)[0]
    lib_dir = python_dir.rsplit("/", 1)[0] + "/lib"
    script = (
        f'export PATH={shquote(python_dir)}:"$PATH":/usr/lib/wsl/lib; '
        f'export LD_LIBRARY_PATH={shquote(lib_dir)}"${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"; '
        + script
    )
    return wsl_bash(script)


def run_wsl(script, **kwargs):
    return subprocess.run(
        wsl_command(script),
        text=True,
        capture_output=True,
        **kwargs,
    )


def stop_in_wsl(script):
    """Run a pkill stop script in WSL. A stopped distro runs nothing to stop, and starting it
    only for this took longer than the timeout, so it is skipped. pkill needs none of the paths
    from config.json, so this also works while they are unset."""
    if not wsl_distro_running():
        return
    try:
        subprocess.run(wsl_bash(script), text=True, capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"Could not run the stop command in WSL: {exc}")


def voice_runtime():
    # The engine's config files are written into its folder in WSL, so its paths come first.
    if WSL_PATHS_ERROR:
        raise ValueError(WSL_PATHS_ERROR)
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
        # Called with each line of output, as TrainingProgress is.
        self.watch = None

    def start(self):
        if self.process and self.process.poll() is None:
            return False
        self.returncode = None
        self.status = "running"
        self.lines = []
        try:
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
        except OSError as exc:
            self.status = "failed"
            self.lines.append(f"Could not start {self.name}: {exc}")
            raise
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
        with self.process.stdout:
            for line in self.process.stdout:
                line = line.rstrip()
                with STATE_LOCK:
                    self.lines.append(line)
                    del self.lines[:-1200]
                if self.watch:
                    self.watch(line)

    def _wait(self):
        if not self.process:
            return
        self.returncode = self.process.wait()
        if self.status in ("stopping", "stopped"):
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
            "progress": self.watch.state() if self.watch else None,
        }


class TrainingProgress:
    """Follows a training job's output: the last epoch it finished, and the time an epoch takes.

    SoVITS training logs "====> Epoch: N" after epoch N. GPT training shows Lightning's progress
    bar, "Epoch N: 100%", counting from 0. Both continue from an earlier run's checkpoint, so the
    first epoch seen need not be 1."""

    PATTERNS = {"sovits": re.compile(r"====> Epoch: (\d+)"), "gpt": re.compile(r"\bEpoch (\d+):\s*100%")}

    def __init__(self, kind, epochs):
        self.kind = kind
        self.epochs = int(epochs)
        self.epoch = 0
        self.finished = []

    def __call__(self, line):
        match = self.PATTERNS[self.kind].search(line)
        if not match:
            return
        epoch = int(match.group(1)) + (1 if self.kind == "gpt" else 0)
        # The bar shows 100% more than once while it updates its numbers.
        if epoch != self.epoch:
            self.epoch = epoch
            self.finished.append(time.time())

    def state(self):
        left = None
        if len(self.finished) >= 2:
            per_epoch = (self.finished[-1] - self.finished[0]) / (len(self.finished) - 1)
            left = round(per_epoch * max(0, self.epochs - self.epoch))
        return {"epoch": self.epoch, "epochs": self.epochs, "seconds_left": left}


class JobStopped(Exception):
    pass


class ThreadJob:
    """A job that runs a function in a thread of the dashboard, shown and stopped like a
    ManagedProcess. The function reports progress with say(), runs programs with run(), and
    calls check_stop() between steps."""

    def __init__(self, name, target):
        self.name = name
        self.target = target
        self.lines = []
        self.status = "idle"
        self.returncode = None
        self.stopping = False
        self.thread = None
        self.child = None
        self.child_stop_script = None
        self.uses_gpu = False

    def start(self):
        self.lines = []
        self.returncode = None
        self.stopping = False
        self.status = "running"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def _run(self):
        try:
            self.target(self)
            self.returncode = 0
            self.status = "stopped" if self.stopping else "done"
        except JobStopped:
            self.returncode = 1
            self.status = "stopped"
        except Exception as exc:
            self.returncode = 1
            self.status = "stopped" if self.stopping else "failed"
            if not self.stopping:
                self.say(f"Failed: {exc}")
        log(f"{self.name} {self.status}.")

    def say(self, line):
        with STATE_LOCK:
            self.lines.append(line)
            del self.lines[:-1200]

    def check_stop(self):
        if self.stopping:
            raise JobStopped()

    def run(self, command, timeout=None):
        """Run a program and return its finished process, with its output as text."""
        self.check_stop()
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as child:
            self.child = child
            if self.stopping:
                child.kill()
            try:
                stdout, stderr = child.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate()
                raise RuntimeError(f"{Path(command[0]).name} did not finish within {timeout} seconds.")
            finally:
                self.child = None
        self.check_stop()
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)

    def stream(self, command, env=None, stop_script=None):
        """Run a program and show its output as it comes. A stop ends it, and a program in WSL
        also by stop_script, as ending wsl.exe leaves the Linux process running."""
        self.check_stop()
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as child:
            self.child, self.child_stop_script = child, stop_script
            if self.stopping:
                child.kill()
                if stop_script:
                    stop_in_wsl(stop_script)
            try:
                for line in child.stdout:
                    line = line.rstrip()
                    # Progress bars redraw many times a second; their last state says nothing new.
                    if line and not re.search(r"\d+%\||it/s\]|s/it\]|GetSegmentId", line):
                        self.say(line)
                child.wait()
            finally:
                self.child = self.child_stop_script = None
        self.check_stop()
        return child.returncode

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def last_lines(self, count):
        return self.lines[-count:]

    def stop(self):
        if not self.running():
            return
        self.stopping = True
        child, stop_script = self.child, getattr(self, "child_stop_script", None)
        if child and child.poll() is None:
            child.kill()
        if stop_script:
            stop_in_wsl(stop_script)
        self.thread.join(timeout=15)
        log(f"{self.name} stopped.")

    def snapshot(self):
        return {
            "name": self.name,
            "status": "running" if self.running() else self.status,
            "returncode": self.returncode,
            "lines": self.lines[-250:],
            "dataset": getattr(self, "dataset_id", ""),
            "results": getattr(self, "results", None),
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


def sovits_version(path):
    """The version of a SoVITS model, from its folder or the name of a pretrained one."""
    return "v2ProPlus" if "v2ProPlus" in str(path) else "v2Pro"


def model_files():
    def collect(folders, pattern):
        rows = []
        for folder in folders:
            try:
                for path in (WSL_GSV_WIN / folder).glob(pattern):
                    if path.is_file():
                        rows.append(
                            {
                                "name": path.name,
                                "path": f"{folder}/{path.name}",
                                "version": sovits_version(f"{folder}/{path.name}"),
                                "mtime": path.stat().st_mtime,
                                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                            }
                        )
            except OSError:
                pass
        rows.sort(key=lambda row: row["mtime"], reverse=True)
        return rows

    # Trained models first, newest first, then the pretrained ones they start from.
    models = {
        "gpt": collect([f"GPT_weights_{version}" for version in SOVITS_VERSIONS], "*.ckpt")
        + collect(["GPT_SoVITS/pretrained_models"], "s1v3.ckpt"),
        "sovits": collect([f"SoVITS_weights_{version}" for version in SOVITS_VERSIONS], "*.pth")
        + collect(["GPT_SoVITS/pretrained_models/v2Pro"], "s2Gv2Pro.pth")
        + collect(["GPT_SoVITS/pretrained_models/v2Pro"], "s2Gv2ProPlus.pth"),
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
        "language": language_code(manifest.get("default_language")),
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


def convert_audio(source, target, sample_rate=44100):
    """Convert to a mono 16-bit WAV, at sample_rate, or at the source's own rate when it is None."""
    rate = ["-ar", str(sample_rate)] if sample_rate else []
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), *rate, "-ac", "1", "-sample_fmt", "s16", str(target)],
        text=True,
        capture_output=True,
        timeout=180,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg could not convert this audio file.")


def add_dataset_clip(root, converted, stem, source_name, transcript, speaker="", language=""):
    """Move a converted WAV into a dataset under a free name, list it, and return the name."""
    with dataset_lock(root):
        manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        taken = {entry.get("wav") for entry in manifest.get("entries", [])}
        wav_name = f"{stem}.wav"
        counter = 2
        while wav_name in taken or (root / "wavs" / wav_name).exists():
            wav_name = f"{stem}-{counter}.wav"
            counter += 1
        # The clip can come from another drive, such as TMP_DIR's, which os.replace cannot cross.
        # It is moved under another name and renamed, so wavs/ never holds half a clip.
        part = root / "wavs" / (wav_name + ".part")
        shutil.move(str(converted), str(part))
        os.replace(part, root / "wavs" / wav_name)
        manifest.setdefault("entries", []).append(
            {
                "wav": wav_name,
                "source_name": source_name,
                "text": transcript,
                "speaker": speaker or manifest.get("default_speaker") or "speaker",
                "language": language or manifest.get("default_language") or "en",
            }
        )
        write_dataset_manifest(root, manifest)
    return wav_name


def add_dataset_audio(dataset_id, filename, audio_base64, text, speaker="", language="en"):
    descriptor = dataset_descriptor(dataset_id)
    transcript = clean_transcript(text)
    if not transcript:
        raise RuntimeError("Every audio file needs a transcript.")
    audio = decode_audio_upload(audio_base64)
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
        wav_name = add_dataset_clip(root, converted, stem, Path(filename).name, transcript, speaker, language)
    finally:
        for path in (source, converted):
            try:
                path.unlink()
            except OSError:
                pass
    log(f"Added {wav_name} to dataset {descriptor['name']}.")
    return dataset_summary(dataset_descriptor(dataset_id))


# SoVITS training skips clips this short or this long (GPT_SoVITS/module/data_utils.py).
TRAINING_MIN_SECONDS = 0.6
TRAINING_MAX_SECONDS = 54.0


def wav_seconds(path):
    try:
        with wave.open(str(path), "rb") as audio:
            return round(audio.getnframes() / audio.getframerate(), 2)
    except (OSError, EOFError, ZeroDivisionError, wave.Error):
        return None


def dataset_clips(dataset_id):
    """The clips of a dataset, in the order they were added, with their transcripts and lengths."""
    descriptor = dataset_descriptor(dataset_id)
    clips = []
    for entry in descriptor["entries"]:
        wav = str(entry.get("wav", ""))
        path = descriptor["root"] / "wavs" / wav
        try:
            # The time in the address makes a browser load a clip again after it was replaced.
            version = int(path.stat().st_mtime)
        except OSError:
            version = 0
        clips.append(
            {
                "wav": wav,
                "text": entry.get("text", ""),
                "duration": wav_seconds(path),
                "url": f"/datasets/{descriptor['id']}/wavs/{urllib.parse.quote(wav)}?v={version}",
            }
        )
    return {"clips": clips, "seconds": round(sum(clip["duration"] or 0 for clip in clips), 1)}


def update_dataset_clip(dataset_id, wav, text):
    descriptor = dataset_descriptor(dataset_id)
    transcript = clean_transcript(text)
    if not transcript:
        raise RuntimeError("A clip needs its transcript. Delete the clip instead.")
    root = descriptor["root"]
    with dataset_lock(root):
        manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        entry = next((row for row in manifest.get("entries", []) if row.get("wav") == wav), None)
        if not entry:
            raise RuntimeError("That clip is no longer in the dataset.")
        entry["text"] = transcript
        write_dataset_manifest(root, manifest)
    return transcript


def delete_dataset_clip(dataset_id, wav):
    descriptor = dataset_descriptor(dataset_id)
    root = descriptor["root"]
    with dataset_lock(root):
        manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        entries = manifest.get("entries", [])
        kept = [row for row in entries if row.get("wav") != wav]
        if len(kept) == len(entries):
            raise RuntimeError("That clip is no longer in the dataset.")
        manifest["entries"] = kept
        write_dataset_manifest(root, manifest)
    path = root / "wavs" / Path(str(wav)).name
    for attempt in range(30):
        try:
            path.unlink(missing_ok=True)
            break
        except PermissionError:
            # Still being sent to the page's player, which lets go a moment after it stops.
            if attempt == 29:
                log(f"{path.name} is out of the dataset, but its file is still in use, so it stays in wavs/.")
            time.sleep(0.1)
    log(f"Deleted {wav} from dataset {descriptor['name']}.")


# Dataset check. Measured on clean game recordings, speech stood 20 dB or more above the quietest
# moments of each clip, and noise 10 dB below the speech brought that to about 17 dB.
CHECK_NOISE_DB = 16
# Less than this, and nothing stands out: a tone, a hum, or noise without speech.
CHECK_FLAT_DB = 6
CHECK_QUIET_DB = -30
# Places where three or more samples in a row are at full volume, where the sound was cut off.
CHECK_CLIPPED_PLACES = 10
# The share of a transcript's words that speech recognition hears differently.
CHECK_MISMATCH = 0.3


def clip_sound(path):
    """The loudest level of a 16-bit WAV clip in dB below full volume, how far its speech stands
    above its quietest moments, the places where it is clipped, and a digest of its samples.
    None when the file cannot be read."""
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getsampwidth() != 2:
                return None
            rate = audio.getframerate() * audio.getnchannels()
            frames = audio.readframes(audio.getnframes())
    except (OSError, EOFError, wave.Error):
        return None
    # A file cut short can end in half a sample.
    samples = array.array("h", frames[:len(frames) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return None
    step = max(1, int(rate * 0.02))
    levels = sorted(
        10 * math.log10(max(sum(x * x for x in samples[i:i + step]) / step, 1) / 32768 ** 2)
        for i in range(0, max(1, len(samples) - step + 1), step)
    )
    clipped = run = rail = 0
    for x in samples:
        side = 1 if x >= 32760 else (-1 if x <= -32760 else 0)
        if side and side == rail:
            run += 1
        else:
            clipped += run >= 3
            run, rail = (1, side) if side else (0, 0)
    clipped += run >= 3
    peak = max(max(samples), -min(samples))
    return {
        "peak_db": 20 * math.log10(max(peak, 1) / 32768),
        # The loud 5 % of a clip is speech, and its quietest 10 % the background between words.
        "speech_db": levels[int(len(levels) * 0.95)] - levels[int(len(levels) * 0.1)],
        "clipped": clipped,
        "digest": hashlib.sha1(samples.tobytes()).hexdigest(),
    }


def transcript_words(text, language):
    """A transcript as words to compare, without case or punctuation; Chinese, Japanese and
    Cantonese as characters, as they have no spaces between words, and Korean too, as its
    spacing varies."""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    # 1,210 and 1210 are one number.
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = "".join(" " if unicodedata.category(char)[0] in "PS" else char for char in text)
    if language in {"zh", "ja", "yue", "ko"}:
        return [char for char in text if not char.isspace()]
    return text.split()


def transcript_difference(expected, heard, language):
    """The words that differ between two transcripts, as a share of the expected words."""
    expected, heard = transcript_words(expected, language), transcript_words(heard, language)
    if not expected:
        return 1.0 if heard else 0.0
    previous = list(range(len(heard) + 1))
    for i, word in enumerate(expected, 1):
        current = [i]
        for j, other in enumerate(heard, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1] / len(expected)


def start_dataset_check(dataset_id):
    descriptor = dataset_descriptor(dataset_id)
    with STATE_LOCK:
        current = JOBS.get("dataset-check")
        if current and current.running():
            raise RuntimeError("A dataset check is already running.")
        job = ThreadJob(f"Check {descriptor['name']}", lambda job: check_dataset(job, descriptor["id"]))
        job.dataset_id = descriptor["id"]
        job.results = []
        JOBS["dataset-check"] = job
        job.start()
    log(f"Checking the clips of {descriptor['name']}.")


def check_dataset(job, dataset_id):
    """Find clips that can make a voice worse. Each clip with problems goes to job.results with
    what was found, and, when its transcript differs, the words speech recognition heard."""
    descriptor = dataset_descriptor(dataset_id)
    clips = dataset_clips(dataset_id)["clips"]
    found = {}

    def flag(wav, problem, heard=None):
        with STATE_LOCK:
            if wav not in found:
                found[wav] = {"wav": wav, "problems": [], "heard": None}
                job.results.append(found[wav])
            found[wav]["problems"].append(problem)
            if heard is not None:
                found[wav]["heard"] = heard

    job.say(f"Checking the sound of {len(clips)} clips...")
    first = {}
    readable = []
    for clip in clips:
        job.check_stop()
        sound = clip_sound(descriptor["root"] / "wavs" / clip["wav"])
        if not sound:
            flag(clip["wav"], "The file cannot be read.")
            continue
        readable.append(clip)
        if sound["peak_db"] < CHECK_QUIET_DB:
            flag(clip["wav"], f"Very quiet: its loudest moment is {-round(sound['peak_db'])} dB below full volume.")
        elif sound["speech_db"] < CHECK_FLAT_DB:
            flag(clip["wav"], "No speech stands out: the sound is about as loud from start to end.")
        elif sound["speech_db"] < CHECK_NOISE_DB:
            flag(clip["wav"], f"Loud background: the speech is only {round(sound['speech_db'])} dB above it. Restore audio can remove noise and music.")
        if sound["clipped"] >= CHECK_CLIPPED_PLACES:
            flag(clip["wav"], f"Distorted: the sound is cut off at full volume in {sound['clipped']} places.")
        if sound["digest"] in first:
            flag(clip["wav"], f"The same audio as {first[sound['digest']]}.")
        else:
            first[sound["digest"]] = clip["wav"]
    language = descriptor["language"]
    skipped = ""
    if language == "yue" and "large-v3" not in dataset_asr_model(language):
        # Whisper before large-v3 writes Cantonese as Mandarin, which would flag every transcript.
        skipped = " Cantonese transcripts are compared only with a large-v3 dataset_asr_model in config.json."
        readable = []
    else:
        job.say(f"Sound checked. Comparing transcripts with what speech recognition hears, 0 of {len(readable)}...")
    for index, clip in enumerate(readable, 1):
        job.check_stop()
        try:
            heard = transcribe_dataset_wav(descriptor["root"] / "wavs" / clip["wav"], language)
        except FileNotFoundError:
            # Deleted while the check runs.
            continue
        if transcript_difference(clip["text"], heard, language) >= CHECK_MISMATCH:
            flag(clip["wav"], "The transcript differs from what speech recognition hears.", heard)
        if index % 10 == 0 or index == len(readable):
            job.say(f"Comparing transcripts with what speech recognition hears, {index} of {len(readable)}...")
    job.say((f"Done. {len(found)} of {len(clips)} clips need a look." if found else f"Done. No problems found in {len(clips)} clips.") + skipped)


# Dataset transcripts come from faster-whisper in the speech recognition server, with a larger
# model than Mic to Voice uses, since a transcript is written once and should be exact.
DATASET_ASR_MODEL = str(APP_CONFIG.get("dataset_asr_model") or "")


def dataset_asr_model(language):
    return DATASET_ASR_MODEL or ("small.en" if language == "en" else "small")


def ensure_relay_asr():
    """Start speech recognition when it is not running, and wait until it listens."""
    if relay_asr_ready():
        return
    start_relay_asr()
    # A cold start, as after booting, reads its libraries from disk and can take a minute.
    for _ in range(480):
        if relay_asr_ready():
            return
        with STATE_LOCK:
            proc = JOBS.get("relay-asr")
        if proc and not proc.running():
            detail = " ".join(proc.lines[-3:])[-300:]
            raise RuntimeError(f"Speech recognition stopped while starting. {detail}".strip())
        time.sleep(0.25)
    raise RuntimeError("Speech recognition did not start within 2 minutes. See Logs for details.")


def transcribe_dataset_wav(wav_path, language):
    """The words spoken in a WAV file, for a dataset in this language."""
    model = dataset_asr_model(language)
    # Whisper models before large-v3 have no Cantonese; they write it as Chinese.
    whisper_language = "zh" if language == "yue" and "large-v3" not in model else language
    payload = {
        "audio_base64": base64.b64encode(Path(wav_path).read_bytes()).decode("ascii"),
        "model": model,
        "language": whisper_language,
        "beam_size": 5,
        "vad": False,
    }
    # Stop Everything, or freeing the GPU for training, stops speech recognition, perhaps in the
    # middle of a recording being cut; it is started again, once.
    for attempt in (1, 2):
        ensure_relay_asr()
        request = urllib.request.Request(
            RELAY_ASR_URL + "/transcribe",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # The first use downloads the model, which can take minutes.
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                body = response.read()
            break
        except urllib.error.HTTPError as exc:
            body = exc.read()
            break
        except (urllib.error.URLError, ConnectionError):
            if attempt == 2:
                raise RuntimeError("Speech recognition stopped while it was writing a transcript.")
            time.sleep(1)
    try:
        result = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        result = {}
    if not result.get("ok"):
        raise RuntimeError(f"Speech recognition failed: {result.get('error') or 'no answer'}")
    if result.get("model") != model:
        # A speech recognition server from before an update knows only the Mic to Voice model.
        raise RuntimeError("Speech recognition is running from an older version. Press Stop Everything, then try again.")
    return clean_transcript(result.get("text"))


def transcribe_dataset_upload(dataset_id, filename, audio_base64):
    descriptor = dataset_descriptor(dataset_id)
    audio = decode_audio_upload(audio_base64)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, source_name = tempfile.mkstemp(prefix="transcribe-", suffix=Path(str(filename or "")).suffix.lower() or ".audio", dir=str(TMP_DIR))
    os.close(fd)
    source = Path(source_name)
    converted = source.with_name(source.stem + "-16k.wav")
    try:
        source.write_bytes(audio)
        convert_audio(source, converted, sample_rate=16000)
        return transcribe_dataset_wav(converted, descriptor["language"])
    finally:
        for path in (source, converted):
            path.unlink(missing_ok=True)


# Long recordings are cut into clips at pauses. ffmpeg measures the loudness of every 20 ms, so
# the dashboard needs no audio libraries; each clip is transcribed and added to the dataset. The
# page sends recordings longer than 20 seconds here.
CUT_STEP = 0.02
CUT_DIR = OUTPUT_DIR / "cut"


def frame_levels(job, wav_path):
    """The loudness of each 20 ms of a recording, in dB; silence is -inf."""
    result = job.run(
        [
            "ffmpeg", "-v", "error", "-i", str(wav_path), "-af",
            "aresample=16000,asetnsamples=n=320:p=0,"
            "astats=metadata=1:reset=1:measure_overall=RMS_level:measure_perchannel=none,"
            "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null", "-",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg could not measure the recording.")
    levels = []
    for line in result.stdout.splitlines():
        if line.startswith("lavfi.astats.Overall.RMS_level="):
            try:
                levels.append(float(line.split("=", 1)[1]))
            except ValueError:
                levels.append(float("-inf"))
    return levels


def plan_clips(levels, step=CUT_STEP, min_seconds=3.0, max_seconds=10.0):
    """Where to cut a recording, as (start, end) seconds, from the loudness of each step.

    Speech is what is louder than the quiet parts by a third of the way to the loud parts, so the
    level of the recording and of its background does not matter. The loud parts are measured
    among what stands out from the quiet ones, so a recording that is mostly silence still has its
    speech as its loud part; one where nothing stands out has no speech to cut. A clip ends at the
    first pause of 0.3 s once it is min_seconds long, always at a pause of a second, and before it
    would grow past max_seconds; speech that goes on longer is cut at its quietest moment. Each
    clip keeps up to 0.2 s of the pause on both sides."""
    # Digital silence measures -inf; it counts as -120 dB, so it is the quiet part of a recording.
    heard = sorted(max(level, -120.0) for level in levels)
    if not heard or heard[-1] <= -120:
        return []
    quiet = heard[len(heard) // 20]
    standing_out = [level for level in heard if level > quiet + 6]
    if not standing_out:
        return []
    loud = standing_out[len(standing_out) * 9 // 10]
    threshold = quiet + (loud - quiet) / 3
    # Runs of speech, joined across dips shorter than 0.1 s, such as the stop in "kept".
    runs = []
    for index, level in enumerate(levels):
        if level <= threshold:
            continue
        if runs and (index - runs[-1][1]) * step < 0.1:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1])
    if not runs:
        return []
    clips = []
    start, end = runs[0]
    for run_start, run_end in runs[1:]:
        pause = (run_start - end) * step
        if pause >= 1.0 or (run_end - start) * step > max_seconds or ((end - start) * step >= min_seconds and pause >= 0.3):
            clips.append((start, end))
            start = run_start
        end = run_end
    clips.append((start, end))

    def split(start, end):
        if (end - start) * step <= max_seconds:
            return [(start, end)]
        margin = int(min_seconds / step)
        middle = range(start + margin, end - margin)
        cut = min(middle, key=lambda index: levels[index]) if len(middle) else (start + end) // 2
        return split(start, cut) + split(cut, end)

    clips = [piece for start, end in clips for piece in split(start, end)]
    total = len(levels) * step
    padded = []
    for index, (start, end) in enumerate(clips):
        before = (start - clips[index - 1][1]) * step / 2 if index else 0.2
        after = (clips[index + 1][0] - end) * step / 2 if index + 1 < len(clips) else 0.2
        begin = max(0.0, start * step - min(0.2, before))
        finish = min(total, end * step + min(0.2, after))
        # Training skips clips of 0.6 s or less; shorter sounds are clicks and breaths.
        if finish - begin > 0.7:
            padded.append((round(begin, 3), round(finish, 3)))
    return padded


def write_wav_part(source, target, start, end):
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        reader.setpos(min(reader.getnframes(), int(start * rate)))
        frames = reader.readframes(int((end - start) * rate))
        with wave.open(str(target), "wb") as writer:
            writer.setparams(reader.getparams())
            writer.writeframes(frames)


def cut_recordings(job, dataset_id):
    """The work of the dataset-cut job: each recording in job.sources, in turn."""
    try:
        cut_each_recording(job, dataset_descriptor(dataset_id))
    finally:
        # Recordings it did not reach, because it failed or was stopped.
        with STATE_LOCK:
            job.accepting = False
            left, job.sources = job.sources, []
        for source, _ in left:
            source.unlink(missing_ok=True)


def cut_each_recording(job, descriptor):
    added = skipped = 0
    failed = []
    while True:
        with STATE_LOCK:
            if not job.sources:
                # From here on start_dataset_cut starts a new job instead of adding to this one.
                job.accepting = False
                break
            source, name = job.sources.pop(0)
        # One recording that cannot be read does not stop the ones after it.
        try:
            made, left_out = cut_recording(job, descriptor, source, name)
            added += made
            skipped += left_out
        except JobStopped:
            raise
        except Exception as exc:
            failed.append(name)
            job.say(f"Could not cut {name}: {exc}")
        finally:
            source.unlink(missing_ok=True)
            source.with_name(source.stem + "-44k.wav").unlink(missing_ok=True)
    job.say(f"Done. {added} clip{'' if added == 1 else 's'} to check in the list below."
            + (f" {skipped} had no words and were left out." if skipped else ""))
    if failed:
        raise RuntimeError(f"{len(failed)} recording{'' if len(failed) == 1 else 's'} could not be cut: {', '.join(failed)}.")


def cut_recording(job, descriptor, source, name):
    """Cut one recording into clips and transcribe them. The clips with words go to job.results,
    from where the page puts them in the list of files to check before they are added.
    Returns the clips made and those left out."""
    job.check_stop()
    stem = slugify(Path(name).stem, "clip")[:48]
    converted = source.with_name(source.stem + "-44k.wav")
    job.say(f"Converting {name}...")
    result = job.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-ar", "44100", "-ac", "1", "-sample_fmt", "s16", str(converted)])
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read it: {result.stderr.strip()}")
    job.say(f"Finding the pauses in {name}...")
    clips = plan_clips(frame_levels(job, converted))
    if not clips:
        job.say(f"Found no speech in {name}.")
        return 0, 0
    job.say(f"Cut {name} into {len(clips)} clips. Writing their transcripts...")
    added = skipped = 0
    CUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, (start, end) in enumerate(clips, 1):
        job.check_stop()
        clip = f"{stem}-{index:03d}.wav"
        part = CUT_DIR / f"{os.urandom(4).hex()}-{clip}"
        write_wav_part(converted, part, start, end)
        text = transcribe_dataset_wav(part, descriptor["language"])
        if not text:
            part.unlink(missing_ok=True)
            skipped += 1
            continue
        with STATE_LOCK:
            job.results.append({"name": clip, "url": f"/outputs/cut/{part.name}", "text": text})
        added += 1
        job.say(f"Transcribed {index} of {len(clips)} from {name}: {text}")
    return added, skipped


def start_dataset_cut(dataset_id, sources):
    """Cut recordings, given as (temporary file, name) pairs, into clips of a dataset. A run for the
    same dataset takes them on; the files are deleted when they are done."""
    try:
        descriptor = dataset_descriptor(dataset_id)
    except Exception:
        for source, _ in sources:
            source.unlink(missing_ok=True)
        raise
    with STATE_LOCK:
        current = JOBS.get("dataset-cut")
        busy = bool(current and current.running() and current.accepting and not current.stopping)
        if busy and current.dataset_id == descriptor["id"]:
            current.sources.extend(sources)
        elif not busy:
            # The clips of an earlier run were taken by the page then, or are no longer wanted.
            for old in CUT_DIR.glob("*.wav"):
                old.unlink(missing_ok=True)
            job = ThreadJob(f"Cut recordings for {descriptor['name']}", lambda job: cut_recordings(job, descriptor["id"]))
            job.dataset_id = descriptor["id"]
            job.sources = list(sources)
            job.results = []
            job.accepting = True
            JOBS["dataset-cut"] = job
            job.start()
    if busy and current.dataset_id != descriptor["id"]:
        for source, _ in sources:
            source.unlink(missing_ok=True)
        raise RuntimeError("Another dataset's recordings are being cut. Wait until that ends.")
    if busy:
        current.say(f"{len(sources)} more recording{'' if len(sources) == 1 else 's'} to cut.")
    else:
        log(f"Cutting {len(sources)} recording(s) into clips for {descriptor['name']}.")


def read_upload(handler, limit=4 * 1024 ** 3):
    """Save the request body, a file the page sends as it is, to a temporary file."""
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0 or length > limit:
        raise RuntimeError("Files must be between 1 byte and 4 GB.")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="upload-", suffix=".audio", dir=str(TMP_DIR))
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


def cut_restored(dataset_id, item_ids):
    """Cut recordings from Restore audio into clips of a dataset."""
    known = {row["id"]: row for row in restore_items()}
    sources = []
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for item_id in item_ids or []:
        row = known.get(item_id)
        if not row:
            continue
        copy = TMP_DIR / f"upload-{os.urandom(4).hex()}.wav"
        shutil.copyfile(RESTORE_DIR / row["file"], copy)
        sources.append((copy, f"{row['name']}.wav"))
    if not sources:
        raise RuntimeError("Those recordings no longer exist.")
    start_dataset_cut(dataset_id, sources)


# Restore audio. The LavaSR worker runs in restore_env, which Setup-Restore-Env.bat creates, so the
# dashboard itself needs no audio libraries. Each recording and each result is a WAV in RESTORE_DIR
# with a JSON file beside it, so results written by the worker appear without shared state.
RESTORE_DIR = OUTPUT_DIR / "restored"
RESTORE_PYTHON = Path(APP_CONFIG.get("restore_python") or ROOT / "restore_env" / "Scripts" / "python.exe")
RESTORE_QUALITIES = {"detect", "file", 8000, 16000, 22050, 24000, 32000, 44100, 48000}


def decode_audio_upload(audio_base64):
    try:
        audio = base64.b64decode(audio_base64, validate=True)
    except Exception as exc:
        raise RuntimeError("The uploaded audio data is invalid.") from exc
    if not audio or len(audio) > 100 * 1024 * 1024:
        raise RuntimeError("Audio files must be between 1 byte and 100 MB.")
    return audio


def restore_item_id(name):
    # A result keeps its recording's name, and the clock can read the same twice in one run, so
    # random digits keep the ids of one run apart.
    return f"{slugify(name, 'audio')[:40]}-{datetime.now():%Y%m%d-%H%M%S}-{os.urandom(4).hex()}"


def restore_items():
    items = []
    for sidecar in RESTORE_DIR.glob("*.json"):
        try:
            item = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(item, dict) or not (RESTORE_DIR / str(item.get("file", ""))).is_file():
            continue
        item["url"] = f"/outputs/restored/{item['file']}"
        items.append(item)
    return sorted(items, key=lambda item: item.get("created", 0), reverse=True)


def add_restore_audio(filename, audio_base64):
    """Keep an uploaded recording as a WAV at its own sample rate, which Detect needs."""
    audio = decode_audio_upload(audio_base64)
    filename = str(filename or "audio")
    name = Path(filename).stem or "audio"
    item_id = restore_item_id(name)
    RESTORE_DIR.mkdir(parents=True, exist_ok=True)
    fd, source_name = tempfile.mkstemp(prefix="restore-", suffix=Path(filename).suffix.lower() or ".audio", dir=str(OUTPUT_DIR))
    os.close(fd)
    source = Path(source_name)
    target = RESTORE_DIR / f"{item_id}.wav"
    try:
        source.write_bytes(audio)
        convert_audio(source, target, sample_rate=None)
        with wave.open(str(target), "rb") as wav:
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
        if not frames:
            raise RuntimeError("This file has no sound in it.")
    except BaseException:
        # A failed conversion leaves no WAV without its JSON file behind.
        target.unlink(missing_ok=True)
        raise
    finally:
        try:
            source.unlink()
        except OSError:
            pass
    duration = round(frames / sample_rate, 2)
    item = {
        "id": item_id,
        "name": name,
        "file": target.name,
        "sample_rate": sample_rate,
        "duration": duration,
        "made_from": None,
        "steps": [],
        "created": time.time(),
    }
    write_text_atomic(RESTORE_DIR / f"{item_id}.json", json.dumps(item, indent=2))
    log(f"Added {Path(filename).name} to Restore audio.")
    item["url"] = f"/outputs/restored/{target.name}"
    return item


def delete_restore_item(item_id):
    item = next((row for row in restore_items() if row["id"] == item_id), None)
    if not item:
        raise RuntimeError("That recording no longer exists.")
    for path in (RESTORE_DIR / item["file"], RESTORE_DIR / f"{item['id']}.json"):
        for attempt in range(30):
            try:
                path.unlink(missing_ok=True)
                break
            except PermissionError:
                # Still being sent to a player, which lets go a moment after the page clears it.
                if attempt == 29:
                    raise RuntimeError("The recording is still in use. Stop playing it, then delete it again.")
                time.sleep(0.1)


# Separation models of GPT-SoVITS's tools/uvr5, run in WSL by separate_audio.py, and the words
# a result's history uses for them.
SEPARATION_NOTES = {
    "best": "music removed (BS-Roformer)",
    "main": "music and backing voices removed (HP5)",
    "all": "music removed (HP2)",
    "echo": "echo and reverb removed",
}


def restore_settings(settings):
    """Checked settings for the steps; LavaSR's own options, named as on the page."""
    settings = settings if isinstance(settings, dict) else {}
    music = settings.get("music") or ""
    if music not in {"", "best", "main", "all"}:
        raise ValueError("Choose a separation model from the list.")
    echo = bool(settings.get("echo"))
    denoise = bool(settings.get("denoise"))
    enhance = bool(settings.get("enhance"))
    if not (music or echo or denoise or enhance):
        raise ValueError("Choose at least one thing to remove or restore.")
    model = settings.get("model") or "v2"
    if model not in {"v1", "v2"}:
        raise ValueError("The LavaSR model must be version 1 or 2.")
    quality = settings.get("quality") or "detect"
    if quality not in {"detect", "file"}:
        try:
            quality = int(quality)
        except (TypeError, ValueError):
            quality = None
    if quality not in RESTORE_QUALITIES:
        raise ValueError("Choose the original recording quality from the list.")
    keep_below = settings.get("keep_below")
    if keep_below in (None, ""):
        keep_below = None
    else:
        try:
            keep_below = float(keep_below)
        except (TypeError, ValueError):
            keep_below = 0
        if not 1000 <= keep_below <= 24000:
            raise ValueError("Keep the original sound up to a value from 1 to 24 kHz, or leave it empty.")
    return {"music": music, "echo": echo, "denoise": denoise, "enhance": enhance, "model": model,
            "quality": quality, "keep_below": keep_below, "pieces": bool(settings.get("pieces"))}


def start_restore(item_ids, settings):
    settings = restore_settings(settings)
    separation = [step for step in (settings["music"], "echo" if settings["echo"] else "") if step]
    lavasr = settings["denoise"] or settings["enhance"]
    if lavasr and not RESTORE_PYTHON.exists():
        raise RuntimeError("Remove background noise and Restore high frequencies need Setup-Restore-Env.bat. Run it first.")
    if separation:
        if WSL_PATHS_ERROR:
            raise RuntimeError(WSL_PATHS_ERROR)
        with STATE_LOCK:
            starting = GPU_RESERVATIONS > 0
        if starting or any(job_running(key) for key in GPU_JOBS):
            raise RuntimeError("Removing music or echo needs the GPU, which training is using. Wait until it ends.")
        if job_running("compare"):
            raise RuntimeError("Compare epochs is using the voice engine. Remove music or echo when it ends, or stop it.")
    known = {row["id"]: row for row in restore_items()}
    chosen = [known[item_id] for item_id in (item_ids or []) if item_id in known]
    if not chosen:
        raise RuntimeError("Choose at least one recording to restore.")
    items = [
        {"id": row["id"], "name": row["name"], "path": str(RESTORE_DIR / row["file"]),
         "steps": row.get("steps", []), "out_id": restore_item_id(row["name"])}
        for row in chosen
    ]
    with STATE_LOCK:
        current = JOBS.get("restore")
        if current and current.running():
            raise RuntimeError("Restore audio is already running.")
        # A stopped run leaves its work folder, and a result it was still writing.
        for leftover in [*TMP_DIR.glob(f"restore-{DASHBOARD_PORT}-*"), *RESTORE_DIR.glob("*.part")]:
            if leftover.is_dir():
                shutil.rmtree(leftover, ignore_errors=True)
            else:
                leftover.unlink(missing_ok=True)
        job = ThreadJob("Restore audio", lambda job: run_restore(job, items, settings, separation, lavasr))
        # Counted as a GPU job from the start, so nothing else starts on the GPU meanwhile.
        job.uses_gpu = bool(separation)
        JOBS["restore"] = job
        job.start()
    log(f"Restoring {len(chosen)} recording(s).")
    return {item["id"]: item["out_id"] for item in items}


def run_restore(job, items, settings, separation, lavasr):
    """The steps of a Restore audio run, in order: music and echo in WSL, then LavaSR."""
    # The port in the name tells this dashboard's runs from those of another one with the same TMP_DIR.
    work = TMP_DIR / f"restore-{DASHBOARD_PORT}-{datetime.now():%Y%m%d-%H%M%S}-{os.urandom(4).hex()}"
    work.mkdir(parents=True)
    count = len(items)
    try:
        if separation:
            items = separate_for_restore(job, items, separation, work)
        if items and lavasr:
            job_path = work / "lavasr.json"
            write_text_atomic(job_path, json.dumps({"out_dir": str(RESTORE_DIR), "settings": settings, "items": items}, indent=2))
            env = os.environ.copy()
            env.update({"HF_HOME": str(CACHE_DIR / "huggingface"), "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
                        "PYTHONIOENCODING": "utf-8", "TEMP": str(TMP_DIR), "TMP": str(TMP_DIR)})
            job.stream([str(RESTORE_PYTHON), str(ROOT / "restore_audio.py"), str(job_path)], env=env)
            made = sum((RESTORE_DIR / f"{item['out_id']}.json").is_file() for item in items)
        else:
            for item in items:
                save_restore_result(item, Path(item["path"]))
            made = len(items)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    job.check_stop()
    if made < count:
        raise RuntimeError(f"{made} of {count} file(s) restored; {count - made} failed.")
    job.say(f"Done. {count} file(s) restored.")


def separate_for_restore(job, items, steps, work):
    """Remove music or echo from each recording with separate_audio.py in WSL. Returns the items
    that worked, now pointing at their voice alone, with the step in their history."""
    for item in items:
        # The models need 44.1 kHz stereo.
        item["work"] = str(work / f"{item['out_id']}-in.wav")
        item["output"] = str(work / f"{item['out_id']}-voice.wav")
        result = job.run(["ffmpeg", "-y", "-v", "error", "-i", item["path"], "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", item["work"]])
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg could not read {item['name']}: {result.stderr.strip()}")
    job_path = work / "separate.json"
    runtime = voice_runtime()
    write_text_atomic(job_path, json.dumps({"steps": steps, "device": runtime["device"], "is_half": runtime["is_half"], "items": items}, indent=2))
    free_gpu_for("music and echo removal")
    try:
        script = (
            f"cd {shquote(WSL_GSV_ROOT)} && "
            f"{shquote(WSL_PYTHON)} -s \"$(wslpath -u {shquote(str(ROOT / 'separate_audio.py'))})\" {shquote(str(job_path))}"
        )
        job.stream(wsl_command(voice_environment() + script), stop_script=WSL_STOP_SCRIPTS["separate"])
    finally:
        job.uses_gpu = False
    note = ", ".join(SEPARATION_NOTES[step] for step in steps)
    done = []
    for item in items:
        voice = Path(item.pop("output"))
        item.pop("work")
        if voice.is_file():
            done.append({**item, "path": str(voice), "steps": item["steps"] + [note]})
    return done


def save_restore_result(item, wav):
    """Keep a recording made by the steps in WSL as a result in the list."""
    target = RESTORE_DIR / f"{item['out_id']}.wav"
    part = RESTORE_DIR / f"{item['out_id']}.wav.part"
    shutil.copyfile(wav, part)
    os.replace(part, target)
    sidecar = {
        "id": item["out_id"],
        "name": item["name"],
        "file": target.name,
        "sample_rate": 44100,
        "duration": wav_seconds(target),
        "made_from": item["id"],
        "steps": item["steps"],
        "created": time.time(),
    }
    write_text_atomic(RESTORE_DIR / f"{item['out_id']}.json", json.dumps(sidecar, indent=2))


def kill_orphan_restore_workers():
    """Stop Restore audio runs that an earlier run of this dashboard left, when it was closed or
    crashed. Only restore_audio.py with a job file of this dashboard, by TMP_DIR and port, matches."""
    marker = str(TMP_DIR / f"restore-{DASHBOARD_PORT}-").replace("'", "''")
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
        f"Where-Object {{ $_.CommandLine -like '*restore_audio.py*' -and $_.CommandLine.Contains('{marker}') }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    try:
        run_powershell(script, timeout=8)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"Could not stop an earlier Restore audio run: {exc}")


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
        log("Speech recognition is already running.")
        return True
    if not RELAY_ASR_PYTHON.exists():
        raise RuntimeError("The speech recognition environment is missing. Run Setup-Relay-Env.bat.")
    RELAY_ASR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RELAY_ASR_TMP.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    relay_env = os.environ.copy()
    relay_env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
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
        "Speech recognition",
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
    log(f"Speech recognition starting on {RELAY_ASR_URL}.")
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


def model_pair(gpt_path, sovits_path):
    """The requested models, or the first listed ones for a request that names none or a model
    that no longer exists. The page can only send the choice it saved while its list is empty."""
    models = model_files()
    gpt_paths = [row["path"] for row in models["gpt"]]
    sovits_paths = [row["path"] for row in models["sovits"]]
    gpt = gpt_path if gpt_path in gpt_paths else (gpt_paths[0] if gpt_paths else "")
    sovits = sovits_path if sovits_path in sovits_paths else (sovits_paths[0] if sovits_paths else "")
    return gpt, sovits


def _start_system_worker(gpt_path, sovits_path, run_id):
    try:
        result = run_wsl("true", timeout=30)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "WSL failed to start.").replace("\x00", "").strip()
            raise RuntimeError(detail)
        gpt_path, sovits_path = model_pair(gpt_path, sovits_path)
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
    """Stop push-to-talk helpers that an earlier run of this dashboard left. Only Python processes
    whose command line ends with this dashboard's address match, as start_ptt_helper starts them,
    so editors, shells and other dashboards that name ptt_helper.py are left alone."""
    marker = f"--dashboard http://127.0.0.1:{DASHBOARD_PORT}"
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
        f"Where-Object {{ $_.CommandLine -like '*ptt_helper.py*' -and $_.CommandLine.TrimEnd().EndsWith('{marker}') }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    try:
        run_powershell(script, timeout=8)
    except (OSError, subprocess.SubprocessError) as exc:
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
            "status": "missing" if not CONFIG_PATH.exists() else ("needs_configuration" if CONFIG_ERROR else "ready"),
            "detail": CONFIG_ERROR or str(CONFIG_PATH),
        },
        "dashboard_python": {
            "status": "ready" if Path(sys.executable).exists() else "missing",
            "detail": sys.executable,
        },
        "wsl_files": {
            "status": "needs_configuration" if WSL_PATHS_ERROR else ("ready" if safe_exists(WSL_GSV_WIN) else "missing"),
            "detail": WSL_PATHS_ERROR or str(WSL_GSV_WIN),
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


def write_s2_config(epochs, batch_size, save_every, model_name="voice-model", include_model_version=True, version="v2Pro"):
    runtime = voice_runtime()
    model_name = slugify(model_name, "voice-model")
    plus = version == "v2ProPlus"
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
            "pretrained_s2G": f"GPT_SoVITS/pretrained_models/v2Pro/s2G{version}.pth",
            "pretrained_s2D": f"GPT_SoVITS/pretrained_models/v2Pro/s2D{version}.pth",
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
            # GPT-SoVITS's configs/s2v2ProPlus.json differs from s2v2Pro.json only in these two.
            "upsample_initial_channel": 768 if plus else 512,
            "upsample_kernel_sizes": [20 if plus else 16, 16, 8, 2, 2],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "gin_channels": 1024,
            "semantic_frame_rate": "25hz",
            "freeze_quantizer": True,
            "version": version,
        },
        "s2_ckpt_dir": f"logs/{model_name}",
        "content_module": "cnhubert",
        "save_weight_dir": f"SoVITS_weights_{version}",
        "name": model_name,
        "version": version,
    }
    if not include_model_version:
        config["model"].pop("version", None)
    path = WSL_GSV_WIN / "TEMP" / f"{model_name}_s2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return f"{WSL_GSV_ROOT}/TEMP/{model_name}_s2.json"


def start_api(gpt_path, sovits_path):
    global API_PROCESS, API_GPT_PATH, API_SOVITS_PATH
    # Text to Voice and Mic to Voice can both start the engine; one check-and-start runs at a time.
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


GPU_RESERVATIONS = 0


@contextlib.contextmanager
def gpu_reserved():
    """Count a GPU job as running while its request is handled, before its process exists."""
    global GPU_RESERVATIONS
    with STATE_LOCK:
        GPU_RESERVATIONS += 1
    try:
        yield
    finally:
        with STATE_LOCK:
            GPU_RESERVATIONS -= 1


def gpu_busy():
    """True while a job uses, or is about to use, the GPU that the voice engine runs on."""
    try:
        if voice_runtime()["device"] != "cuda":
            return False
    except ValueError:
        return False
    with STATE_LOCK:
        return GPU_RESERVATIONS > 0 or any(
            job.running() and (key in GPU_JOBS or getattr(job, "uses_gpu", False)) for key, job in JOBS.items()
        )


def separation_running():
    """True while Restore audio removes music or echo, on the GPU in WSL."""
    with STATE_LOCK:
        job = JOBS.get("restore")
    return bool(job and job.running() and getattr(job, "uses_gpu", False))


def refuse_gpu_conflicts(starting):
    """Training, preparation, music removal and Compare epochs each need the GPU or the voice
    engine for a long time, and together they can run out of memory or stop each other."""
    if separation_running():
        raise RuntimeError(f"Removing music or echo is using the GPU. Start {starting} when it ends.")
    if job_running("compare"):
        raise RuntimeError(f"Compare epochs is using the voice engine. Start {starting} when it ends, or stop it.")


def job_running(key):
    with STATE_LOCK:
        job = JOBS.get(key)
        return bool(job and job.running())


def free_gpu_for(job_name):
    """Stop the voice before a GPU job. Both on one GPU can run out of memory, and the voice
    starts again when it is used."""
    if voice_runtime()["device"] != "cuda":
        return
    with STATE_LOCK:
        status = SYSTEM_STATE["status"]
    if status not in {"stopped", "stopping"}:
        log(f"Stopping Mic to Voice to free GPU memory for {job_name}.")
        stop_system()
    elif api_ready() or (API_PROCESS and API_PROCESS.running()):
        log(f"Stopping the voice engine to free GPU memory for {job_name}.")
        stop_api()


def prepare_tts_dataset_job(dataset_id, model_name):
    descriptor = dataset_descriptor(dataset_id)
    summary = dataset_summary(descriptor)
    if not summary["ready"]:
        raise RuntimeError("The selected dataset is incomplete. Add matching audio and transcripts first.")
    if job_running("dataset-prepare"):
        raise RuntimeError("Dataset preparation is already running.")
    refuse_gpu_conflicts("preparation")
    free_gpu_for("dataset preparation")
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


def start_train(kind, epochs, batch_size, save_every, dataset_id="", model_name="voice-model", version="v2Pro"):
    descriptor = dataset_descriptor(dataset_id)
    if version not in SOVITS_VERSIONS:
        raise RuntimeError("Choose v2Pro or v2ProPlus.")
    if job_running(f"train-{kind}"):
        raise RuntimeError(f"{'GPT' if kind == 'gpt' else 'SoVITS'} training is already running.")
    refuse_gpu_conflicts("training")
    model_name = slugify(model_name, "voice-model")
    feature_root = f"{WSL_GSV_ROOT}/logs/{model_name}"
    check = run_wsl(
        f"test -s {shquote(feature_root + '/2-name2text.txt')} -a -s {shquote(feature_root + '/6-name2semantic.tsv')}",
        timeout=15,
    )
    if check.returncode != 0:
        raise RuntimeError("Training features are missing. Run Sync & Prepare before starting training.")
    if kind == "sovits" and version != "v2Pro":
        pretrained = f"{WSL_GSV_ROOT}/GPT_SoVITS/pretrained_models/v2Pro"
        check = run_wsl(f"test -s {shquote(f'{pretrained}/s2G{version}.pth')} -a -s {shquote(f'{pretrained}/s2D{version}.pth')}", timeout=15)
        if check.returncode != 0:
            raise RuntimeError(f"GPT-SoVITS is missing the pretrained {version} models, s2G{version}.pth and s2D{version}.pth in "
                               "GPT_SoVITS/pretrained_models/v2Pro. Update GPT-SoVITS, or train v2Pro.")
    free_gpu_for("training")
    if kind == "gpt":
        config_path = write_s1_config(epochs, batch_size, save_every, model_name)
        script = (
            voice_environment() + "ulimit -l 2097152; "
            f"cd {shquote(WSL_GSV_ROOT)} && "
            f"{shquote(WSL_PYTHON)} -s GPT_SoVITS/s1_train.py --config_file {shquote(config_path)}"
        )
        name = f"GPT training: {model_name}"
    elif kind == "sovits":
        config_path = write_s2_config(epochs, batch_size, save_every, model_name, version=version)
        script = (
            voice_environment() + "ulimit -l 2097152; "
            f"cd {shquote(WSL_GSV_ROOT)} && "
            # s2_train.py saves its checkpoints in this folder but does not create it.
            f"mkdir -p {shquote(f'logs/{model_name}/logs_s2_{version}')} && "
            f"{shquote(WSL_PYTHON)} -s GPT_SoVITS/s2_train.py --config {shquote(config_path)}"
        )
        name = f"SoVITS training: {model_name}" + (f" ({version})" if version != "v2Pro" else "")
    else:
        raise ValueError("Unknown training kind.")
    key = f"train-{kind}"
    proc = ManagedProcess(name, wsl_command(script), stop_script=WSL_STOP_SCRIPTS[key])
    proc.watch = TrainingProgress(kind, epochs)
    with STATE_LOCK:
        current = JOBS.get(key)
        if current and current.process and current.process.poll() is None:
            raise RuntimeError(f"{name} is already running.")
        JOBS[key] = proc
    proc.start()
    log(f"{name} started from {descriptor['name']}: epochs={epochs}, batch={batch_size}, save_every={save_every}.")


# A voice file holds voice.json, both models in models/, and the reference clips. Its clips become a
# small dataset on import, as the references of every voice are dataset clips.
VOICE_FILE_FORMAT = 1
VOICE_EXPORT = re.compile(r"^voice-export-[0-9a-f]{8}\.zip$")


def wake_wsl():
    """Start the distro before its files are read through \\wsl.localhost."""
    if run_wsl(":", timeout=90).returncode != 0:
        raise RuntimeError("WSL did not start. Run System Check.")


def export_voice(gpt, sovits, dataset_id, reference_id, prompt_text, extra_ids):
    """Pack a voice into a zip file in TMP_DIR. Returns the file and a name to save it as."""
    wake_wsl()
    models = model_files()
    if gpt not in {row["path"] for row in models["gpt"]} or sovits not in {row["path"] for row in models["sovits"]}:
        raise RuntimeError("Choose a GPT and a SoVITS model first.")
    descriptor = dataset_descriptor(dataset_id)
    references = {ref["id"]: ref for ref in parse_references(descriptor["id"], valid_only=False)}
    reference = references.get(reference_id)
    if not reference or not reference["valid_reference"]:
        raise RuntimeError("Choose a reference clip of 3 to 10 seconds first.")
    extras = [references[ref_id] for ref_id in extra_ids or [] if ref_id in references and ref_id != reference_id
              and references[ref_id]["valid_aux_reference"]]
    manifest = {
        "format": VOICE_FILE_FORMAT,
        "name": descriptor["name"],
        "language": descriptor["language"],
        "version": sovits_version(sovits),
        "gpt": Path(gpt).name,
        "sovits": Path(sovits).name,
        "reference": {"file": "reference.wav", "text": clean_transcript(prompt_text) or reference["text"], "language": reference["lang"]},
        "extra_references": [{"file": f"extra-{index}.wav", "text": ref["text"], "language": ref["lang"]} for index, ref in enumerate(extras, 1)],
    }
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / f"voice-export-{os.urandom(4).hex()}.zip"
    try:
        # The models are already compressed, so the zip only stores them.
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
            bundle.writestr("voice.json", json.dumps(manifest, indent=2))
            bundle.write(WSL_GSV_WIN / gpt, f"models/{Path(gpt).name}")
            bundle.write(WSL_GSV_WIN / sovits, f"models/{Path(sovits).name}")
            for ref, entry in [(reference, manifest["reference"]), *zip(extras, manifest["extra_references"])]:
                bundle.write(descriptor["root"] / "wavs" / f"{ref['id']}.wav", entry["file"])
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    log(f"Exported the voice {descriptor['name']}.")
    return path, f"{slugify(descriptor['name'], 'voice')}.voice.zip"


def install_weight(bundle, member, folder):
    """Copy a model from a voice file into GPT-SoVITS, under a name no other model has."""
    name = Path(member).name
    target_dir = WSL_GSV_WIN / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, counter = name, 2
    while (target_dir / candidate).exists():
        # The same model imported again is used as it is. Models of one kind all have the same
        # size, so only equal contents count.
        existing = target_dir / candidate
        if existing.stat().st_size == bundle.getinfo(member).file_size:
            with bundle.open(member) as packed, existing.open("rb") as installed:
                if hashlib.file_digest(packed, "sha256").digest() == hashlib.file_digest(installed, "sha256").digest():
                    return f"{folder}/{candidate}"
        candidate = f"{stem}-{counter}{suffix}"
        counter += 1
    part = target_dir / (candidate + ".part")
    with bundle.open(member) as source, open(part, "wb") as copy:
        shutil.copyfileobj(source, copy, 1024 * 1024)
    os.replace(part, target_dir / candidate)
    return f"{folder}/{candidate}"


def import_voice(path):
    try:
        bundle = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise RuntimeError("This is not a voice file.") from exc
    with bundle:
        try:
            manifest = json.loads(bundle.read("voice.json").decode("utf-8"))
        except (KeyError, ValueError) as exc:
            raise RuntimeError("This is not a voice file from Local Voice Systems.") from exc
        if manifest.get("format") != VOICE_FILE_FORMAT:
            raise RuntimeError("This voice file is from a newer version of Local Voice Systems. Update the dashboard.")
        version = manifest.get("version")
        if version not in SOVITS_VERSIONS:
            raise RuntimeError(f"This voice is for GPT-SoVITS {version}, and the dashboard uses v2Pro and v2ProPlus.")
        gpt = f"models/{Path(str(manifest.get('gpt'))).name}"
        sovits = f"models/{Path(str(manifest.get('sovits'))).name}"
        references = [manifest.get("reference") or {}, *(manifest.get("extra_references") or [])][:10]
        names = set(bundle.namelist())
        if not gpt.endswith(".ckpt") or not sovits.endswith(".pth") or gpt not in names or sovits not in names:
            raise RuntimeError("The voice file is missing its models.")
        wake_wsl()
        if any(Path(str(ref.get("file"))).name not in names or not clean_transcript(ref.get("text")) for ref in references):
            raise RuntimeError("The voice file is missing a reference clip or its text.")
        if (any(bundle.getinfo(Path(str(ref["file"])).name).file_size > 100 * 1024 ** 2 for ref in references)
                or max(bundle.getinfo(gpt).file_size, bundle.getinfo(sovits).file_size) > 4 * 1024 ** 3):
            raise RuntimeError("The voice file holds files far larger than a voice needs.")
        gpt_path = install_weight(bundle, gpt, "GPT_weights_v2Pro")
        sovits_path = install_weight(bundle, sovits, f"SoVITS_weights_{version}")
        name = str(manifest.get("name") or "Imported voice").strip()[:60] or "Imported voice"
        base, counter = name, 2
        while (DATASETS_DIR / dataset_id_for_name(name)).exists():
            name = f"{base} {counter}"
            counter += 1
        dataset_id = create_dataset(name, language=language_code(manifest.get("language")))["id"]
        root = DATASETS_DIR / dataset_id
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        try:
            for index, ref in enumerate(references):
                member = Path(str(ref["file"])).name
                source = TMP_DIR / f"import-{os.urandom(4).hex()}.wav"
                converted = source.with_name(source.stem + "-44k.wav")
                try:
                    source.write_bytes(bundle.read(member))
                    convert_audio(source, converted)
                    add_dataset_clip(root, converted, "reference" if index == 0 else f"extra-{index}", member,
                                     clean_transcript(ref["text"]), language=language_code(ref.get("language")))
                finally:
                    source.unlink(missing_ok=True)
                    converted.unlink(missing_ok=True)
        except BaseException:
            # No half-made voice stays behind, and a second try gets the same name.
            shutil.rmtree(root, ignore_errors=True)
            raise
    # The voice engine reads the clips in WSL.
    result = run_wsl(dataset_sync_script(dataset_descriptor(dataset_id)), timeout=180)
    if result.returncode != 0:
        log(f"The imported voice's clips could not be synced: {result.stdout} {result.stderr}")
        raise RuntimeError("The voice was imported, but its clips could not be copied to WSL. Press Sync Only on Voice Training.")
    save_speech_settings(dataset_id, {
        "gpt": gpt_path, "sovits": sovits_path, "reference": "reference",
        "extra_references": [f"extra-{index}" for index in range(1, len(references))],
        "prompt_text": clean_transcript(references[0]["text"]), "text_lang": language_code(manifest.get("language")),
    })
    model_files()
    log(f"Imported the voice {name}.")
    return {"dataset": dataset_id, "name": name, "gpt": gpt_path, "sovits": sovits_path, "reference": "reference"}


# OpenAI-style speech for other programs on this PC: POST /v1/audio/speech, and the lists that
# Open WebUI reads, GET /v1/audio/voices, /v1/audio/models and /v1/models. A voice speaks as it
# last did on Text to Voice, with the models, reference clip and settings kept in speech.json.
SPEECH_SETTINGS = "speech.json"
SPEECH_API_KEY = str(APP_CONFIG.get("api_key") or "")
SPEECH_MODEL = "gpt-sovits"
SPEECH_FORMATS = {
    "mp3": ("audio/mpeg", ["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k"]),
    "opus": ("audio/ogg", ["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"]),
    "aac": ("audio/aac", ["-f", "adts", "-c:a", "aac", "-b:a", "128k"]),
    "flac": ("audio/flac", ["-f", "flac"]),
    "wav": ("audio/wav", None),
    # OpenAI's pcm: 24 kHz, 16-bit signed little-endian, one channel, no header.
    "pcm": ("audio/pcm", ["-f", "s16le", "-ar", "24000", "-ac", "1"]),
}
SPEECH_TUNING = ("text_split_method", "seed", "top_k", "top_p", "temperature", "speed_factor", "fragment_interval", "repetition_penalty")


class SpeechApiError(Exception):
    """An error answered in OpenAI's shape."""

    def __init__(self, status, message, kind="invalid_request_error", param=None):
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.param = param

    def body(self):
        return {"error": {"message": str(self), "type": self.kind, "param": self.param, "code": None}}


def save_speech_settings(dataset_id, settings):
    """Keep how a voice last spoke on Text to Voice, for the speech API."""
    try:
        descriptor = dataset_descriptor(dataset_id)
    except (RuntimeError, ValueError):
        return
    if not (settings.get("gpt") and settings.get("sovits") and settings.get("reference")):
        return
    reference = next((ref for ref in parse_references(descriptor["id"], valid_only=False) if ref["id"] == settings["reference"]), None)
    if reference and clean_transcript(settings.get("prompt_text")) == clean_transcript(reference["text"]):
        # Only a reference text edited on Text to Voice is kept. Otherwise the clip's transcript is
        # read when the voice speaks, so a later correction of it counts.
        settings = {key: value for key, value in settings.items() if key != "prompt_text"}
    write_text_atomic(descriptor["root"] / SPEECH_SETTINGS, json.dumps(settings, indent=2))


def speech_settings_from_generation(body):
    return {
        "gpt": str(body.get("gpt") or ""),
        "sovits": str(body.get("sovits") or ""),
        # Clips by name, so a voice still finds them if the datasets folder in WSL moves.
        "reference": Path(str(body.get("ref_audio_path") or "")).stem,
        "extra_references": [Path(str(path)).stem for path in body.get("aux_ref_audio_paths") or []],
        "prompt_text": clean_transcript(body.get("prompt_text")),
        "text_lang": language_code(body.get("text_lang")),
        **{key: body[key] for key in SPEECH_TUNING if key in body},
    }


def speech_voices():
    """The voices other programs can use: those that have spoken a line on Text to Voice."""
    return [{"id": row["id"], "name": row["name"]} for row in list_datasets() if (DATASETS_DIR / row["id"] / SPEECH_SETTINGS).is_file()]


def speech_voice(name):
    wanted = str(name or "").strip().lower()
    voices = speech_voices()
    voice = next((row for row in voices if wanted in {row["id"].lower(), row["name"].lower()}), None)
    if not voice:
        names = ", ".join(row["id"] for row in voices) or "none yet. Speak a line with a voice on Text to Voice first"
        raise SpeechApiError(400, f"There is no voice named {name!r}. Voices: {names}.", param="voice")
    return voice


def speak_for_api(body):
    """Speak the text of an OpenAI-style request. Returns the audio and its content type."""
    if WSL_PATHS_ERROR:
        raise SpeechApiError(503, WSL_PATHS_ERROR, "server_error")
    if not isinstance(body.get("input"), str):
        raise SpeechApiError(400, "input must be text.", param="input")
    text = " ".join(body["input"].split())
    if not text:
        raise SpeechApiError(400, "input is empty.", param="input")
    if len(text) > 4096:
        raise SpeechApiError(400, "input is longer than 4096 characters.", param="input")
    response_format = str(body.get("response_format") or "mp3").lower()
    if response_format not in SPEECH_FORMATS:
        raise SpeechApiError(400, f"response_format must be one of {', '.join(SPEECH_FORMATS)}.", param="response_format")
    try:
        speed = float(body.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = None
    if speed is None or not 0.25 <= speed <= 4.0:
        raise SpeechApiError(400, "speed must be a number from 0.25 to 4.0.", param="speed")
    voice = speech_voice(body.get("voice"))
    descriptor = dataset_descriptor(voice["id"])
    settings = json.loads((descriptor["root"] / SPEECH_SETTINGS).read_text(encoding="utf-8"))
    references = {ref["id"]: ref for ref in parse_references(descriptor["id"], valid_only=False)}
    reference = references.get(settings.get("reference"))
    if not reference or not reference["valid_reference"]:
        raise SpeechApiError(400, f"The reference clip of {voice['name']} is no longer in its dataset. Speak a line with it on Text to Voice again.",
                             param="voice")
    extras = [references[ref_id]["wsl_path"] for ref_id in settings.get("extra_references") or []
              if ref_id in references and ref_id != reference["id"] and references[ref_id]["valid_aux_reference"]]
    # Without this, the engine would speak with its pretrained models instead of a missing one.
    models = model_files()
    if not models["gpt"] and not models["sovits"]:
        raise SpeechApiError(503, "The models of GPT-SoVITS cannot be read. Run System Check in the dashboard.", "server_error")
    for kind, label in (("gpt", "GPT"), ("sovits", "SoVITS")):
        if settings.get(kind) not in {row["path"] for row in models[kind]}:
            raise SpeechApiError(400, f"The {label} model of {voice['name']}, {settings.get(kind)}, is no longer in GPT-SoVITS. "
                                      "Speak a line with the voice on Text to Voice again.", param="voice")
    if not api_ready():
        # As with Mic to Voice, nothing starts by itself while a job uses the GPU.
        if gpu_busy():
            raise SpeechApiError(503, "Training or another GPU job is running, and the voice engine starts when it ends.", "server_error")
        start_api(settings["gpt"], settings["sovits"])
        for _ in range(720):
            if api_ready():
                break
            if API_PROCESS and not API_PROCESS.running():
                raise SpeechApiError(503, "The voice engine stopped while starting. See Logs in the dashboard.", "server_error")
            time.sleep(0.25)
        else:
            raise SpeechApiError(503, "The voice engine did not start within 3 minutes.", "server_error")
    tuning = {key: settings[key] for key in SPEECH_TUNING if key in settings}
    payload = {
        "text": text, "text_lang": settings.get("text_lang") or descriptor["language"],
        "ref_audio_path": reference["wsl_path"], "aux_ref_audio_paths": extras,
        "prompt_text": settings.get("prompt_text") or reference["text"], "prompt_lang": reference["lang"],
        "text_split_method": str(tuning.get("text_split_method", "cut5")),
        "seed": int(tuning.get("seed", -1)),
        "top_k": int(tuning.get("top_k", 15)),
        "top_p": float(tuning.get("top_p", 0.6)),
        "temperature": float(tuning.get("temperature", 0.6)),
        "speed_factor": float(tuning.get("speed_factor", 1.0)) * speed,
        "fragment_interval": float(tuning.get("fragment_interval", 0.3)),
        "repetition_penalty": float(tuning.get("repetition_penalty", 1.35)),
        "batch_size": 1, "media_type": "wav", "parallel_infer": True, "streaming_mode": 0,
    }
    request = urllib.request.Request(API_URL + "/tts", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.time()
    with GENERATION_LOCK:
        try:
            set_api_weights(settings["gpt"], settings["sovits"])
            with urllib.request.urlopen(request, timeout=600) as response:
                wav = response.read()
        except urllib.error.HTTPError as exc:
            raise SpeechApiError(500, f"The voice engine could not speak this: {exc.read().decode('utf-8', errors='replace')[:300]}",
                                 "server_error") from exc
    if not wav.startswith(b"RIFF"):
        raise SpeechApiError(500, f"The voice engine could not speak this: {wav.decode('utf-8', errors='replace')[:300]}", "server_error")
    content_type, ffmpeg_args = SPEECH_FORMATS[response_format]
    audio = wav
    if ffmpeg_args:
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", *ffmpeg_args, "pipe:1"], input=wav, capture_output=True,
                                timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode != 0 or not result.stdout:
            raise SpeechApiError(500, f"ffmpeg could not make {response_format}: {result.stderr.decode('utf-8', errors='replace')[-300:]}",
                                 "server_error")
        audio = result.stdout
    log(f"Speech API: {voice['name']} said {len(text)} characters in {time.time() - started:.1f}s.")
    return audio, content_type


COMPARE_DIR = OUTPUT_DIR / "compare"


def model_epochs(model_name, kind, version="v2Pro"):
    """The saved checkpoints of a trained model, as (epoch, path) pairs. GPT training saves
    name-eN.ckpt and SoVITS training name_eN_sSTEPS.pth, in the folder of its version."""
    name = re.escape(slugify(model_name, "voice-model"))
    pattern = re.compile(rf"^{name}-e(\d+)\.ckpt$" if kind == "gpt" else rf"^{name}_e(\d+)_s\d+\.pth$")
    epochs = {}
    # The list is newest first, so of two files for one epoch the newer is kept.
    for row in model_files()[kind]:
        if (match := pattern.match(row["name"])) and (kind == "gpt" or sovits_version(row["path"]) == version):
            epochs.setdefault(int(match.group(1)), row["path"])
    return sorted(epochs.items())


def start_compare(model_name, kind, dataset_id, reference_wav, text, version="v2Pro"):
    """Generate one line with each saved epoch of a model, the other model at its last epoch.
    The SoVITS model is of the version given."""
    if kind not in {"gpt", "sovits"}:
        raise RuntimeError("Choose GPT or SoVITS epochs to compare.")
    if version not in SOVITS_VERSIONS:
        raise RuntimeError("Choose v2Pro or v2ProPlus.")
    text = " ".join(str(text or "").split())
    if not text:
        raise RuntimeError("Type a test line first.")
    if any(job_running(key) for key in GPU_JOBS):
        raise RuntimeError("Training is using the GPU. Compare epochs when it ends.")
    if separation_running():
        raise RuntimeError("Removing music or echo is using the GPU. Compare epochs when it ends.")
    descriptor = dataset_descriptor(dataset_id)
    wake_wsl()
    reference = next((ref for ref in parse_references(descriptor["id"]) if f"{ref['id']}.wav" == reference_wav), None)
    if not reference:
        raise RuntimeError("Choose a reference clip of 3 to 10 seconds.")
    epochs = model_epochs(model_name, kind, version)
    if not epochs:
        label = "GPT" if kind == "gpt" else ("SoVITS" if version == "v2Pro" else f"{version} SoVITS")
        raise RuntimeError(f"{slugify(model_name, 'voice-model')} has no saved {label} epochs yet.")
    other_kind = "sovits" if kind == "gpt" else "gpt"
    others = model_epochs(model_name, other_kind, version)
    if not others and other_kind == "sovits":
        # A trained SoVITS of the other version still speaks closer to the voice than a pretrained one.
        others = [row for other_version in SOVITS_VERSIONS if other_version != version
                  for row in model_epochs(model_name, "sovits", other_version)]
    # Without saved epochs of the other model, the pretrained one GPT-SoVITS starts from.
    other = others[-1][1] if others else next((row["path"] for row in model_files()[other_kind] if "pretrained" in row["path"]
                                               and (other_kind == "gpt" or sovits_version(row["path"]) == version)), "")
    if not other:
        raise RuntimeError("The pretrained models of GPT-SoVITS are missing. Run System Check.")
    settings = {"name": slugify(model_name, "voice-model"), "kind": kind, "epochs": epochs, "other": other,
                "reference": reference, "text": text, "language": descriptor["language"], "dataset": descriptor}
    with STATE_LOCK:
        current = JOBS.get("compare")
        if current and current.running():
            raise RuntimeError("A comparison is already running.")
        job = ThreadJob(f"Compare {settings['name']} epochs", lambda job: compare_epochs(job, settings))
        job.results = []
        JOBS["compare"] = job
        job.start()
    log(f"Comparing {len(epochs)} {kind} epochs of {settings['name']}.")


def compare_epochs(job, settings):
    # The voice engine reads the reference clip in WSL, where a clip added since the last sync is not yet.
    job.say("Syncing the dataset to WSL...")
    result = job.run(wsl_command(dataset_sync_script(settings["dataset"])), timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"The dataset could not be synced: {(result.stderr or result.stdout).strip()[-300:]}")
    if not api_ready():
        job.say("Starting the voice engine...")
        first = settings["epochs"][0][1]
        start_api(first if settings["kind"] == "gpt" else settings["other"], first if settings["kind"] == "sovits" else settings["other"])
        for _ in range(720):
            job.check_stop()
            if api_ready():
                break
            if API_PROCESS and not API_PROCESS.running():
                raise RuntimeError("The voice engine stopped while starting. See Logs for details.")
            time.sleep(0.25)
        else:
            raise RuntimeError("The voice engine did not start within 3 minutes.")
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    reference = settings["reference"]
    for index, (epoch, path) in enumerate(settings["epochs"], 1):
        job.check_stop()
        gpt, sovits = (path, settings["other"]) if settings["kind"] == "gpt" else (settings["other"], path)
        payload = {
            "text": settings["text"], "text_lang": settings["language"],
            "ref_audio_path": reference["wsl_path"], "prompt_text": reference["text"], "prompt_lang": reference["lang"],
            "top_k": 15, "top_p": 0.6, "temperature": 0.6, "text_split_method": "cut5", "batch_size": 1,
            "speed_factor": 1.0, "fragment_interval": 0.3, "media_type": "wav", "parallel_infer": True,
            "repetition_penalty": 1.35, "streaming_mode": 0,
            # One seed for every epoch, so the epoch is what differs.
            "seed": 1234,
        }
        request = urllib.request.Request(API_URL + "/tts", data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with GENERATION_LOCK:
            set_api_weights(gpt, sovits)
            try:
                with urllib.request.urlopen(request, timeout=240) as response:
                    data = response.read()
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"Epoch {epoch}: {exc.read().decode('utf-8', errors='replace')[:300]}") from exc
        if not data.startswith(b"RIFF"):
            raise RuntimeError(f"Epoch {epoch}: {data.decode('utf-8', errors='replace')[:300]}")
        output = COMPARE_DIR / f"{settings['name']}-{settings['kind']}-e{epoch}.wav"
        output.write_bytes(data)
        job.results.append({"epoch": epoch, "kind": settings["kind"], "gpt": gpt, "sovits": sovits,
                            "url": f"/outputs/compare/{output.name}?v={int(time.time())}"})
        job.say(f"Generated epoch {epoch}, {index} of {len(settings['epochs'])}.")
    job.say("Done. Listen to each epoch and use the best.")


def jobs_now():
    with STATE_LOCK:
        return list(JOBS.items())


def clean_old_temp_files(max_age=3600):
    """Remove the files a crash or a closed dashboard left in TMP_DIR: uploads, cut clips and
    voice files. Only older ones go, as another dashboard may share TMP_DIR."""
    patterns = ("upload-*", "cut-*", "transcribe-*", "import-*", "voice-export-*.zip")
    for pattern in patterns:
        for path in TMP_DIR.glob(pattern):
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
        if not SPEECH_API_KEY or hmac.compare_digest(self.headers.get("Authorization", "").encode(), f"Bearer {SPEECH_API_KEY}".encode()):
            return True
        self.send_json(SpeechApiError(401, "Incorrect API key. Use the api_key from config.json.").body(), 401)
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
                self.send_json({"voices": speech_voices()})
            elif path == "/v1/audio/models":
                self.send_json({"models": [{"id": SPEECH_MODEL}]})
            else:
                self.send_json({"object": "list", "data": [{"id": SPEECH_MODEL, "object": "model", "created": 0, "owned_by": "local-voice-systems"}]})
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
                "gpu_busy": gpu_busy(),
                "restore": {"installed": RESTORE_PYTHON.exists(), "items": restore_items()},
                "relay_asr_ready": relay_asr_ready(),
                "jobs": {key: proc.snapshot() for key, proc in jobs_now()},
                "system": system_snapshot(),
                "log": latest_log_text(),
            }
            self.send_json(data)
            return
        if path.startswith("/api/voice-download/"):
            name = path.rsplit("/", 1)[1]
            file = TMP_DIR / name
            if not VOICE_EXPORT.match(name) or not file.is_file():
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
                self.send_json({"ok": True, **dataset_clips(query.get("dataset", [""])[0])})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 404)
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
                    raise SpeechApiError(400, "The body must be a JSON object of up to 1 MB.") from exc
                audio, content_type = speak_for_api(body)
                self.send_audio(audio, content_type)
            except SpeechApiError as exc:
                self.send_json(exc.body(), exc.status)
            except Exception as exc:
                log(f"Speech API error: {exc}")
                self.send_json(SpeechApiError(500, str(exc), "server_error").body(), 500)
            return
        if path == "/api/voice-import":
            source = None
            try:
                source = read_upload(self)
                self.send_json({"ok": True, **import_voice(source)})
            except Exception as exc:
                log(f"Error: {exc}")
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
                start_dataset_cut(query.get("dataset", [""])[0], [(source, query.get("filename", ["recording"])[0])])
                self.send_json({"ok": True})
            except Exception as exc:
                log(f"Error: {exc}")
                self.send_json({"ok": False, "error": str(exc)}, 500)
            return
        try:
            body = read_json(self)
            if path == "/api/start-system":
                if body.get("auto") and gpu_busy():
                    self.send_json({"ok": True, "started": False})
                    return
                # The startup worker checks the models once WSL is running; see model_pair.
                start_system(body.get("gpt") or "", body.get("sovits") or "", body.get("binding") or "ShiftLeft")
                self.send_json({"ok": True, "started": True})
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
                    log(f"Cancelled Mic to Voice line {generation_id}.")
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
            if path == "/api/voice-export":
                file, name = export_voice(body.get("gpt"), body.get("sovits"), body.get("dataset_id"), body.get("reference"),
                                          body.get("prompt_text"), body.get("extra_references"))
                self.send_json({"ok": True, "url": f"/api/voice-download/{file.name}?name={urllib.parse.quote(name)}", "name": name})
                return
            if path == "/api/compare-epochs":
                start_compare(body.get("model"), body.get("kind"), body.get("dataset_id"), body.get("reference"), body.get("text"),
                              body.get("version") or "v2Pro")
                self.send_json({"ok": True})
                return
            if path == "/api/dataset-check":
                start_dataset_check(body.get("dataset_id"))
                self.send_json({"ok": True})
                return
            if path == "/api/dataset-cut-restored":
                cut_restored(body.get("dataset_id"), body.get("item_ids"))
                self.send_json({"ok": True})
                return
            if path == "/api/dataset-clip-text":
                self.send_json({"ok": True, "text": update_dataset_clip(body.get("dataset_id"), body.get("wav"), body.get("text"))})
                return
            if path == "/api/dataset-clip-delete":
                delete_dataset_clip(body.get("dataset_id"), body.get("wav"))
                self.send_json({"ok": True})
                return
            if path == "/api/dataset-transcribe":
                text = transcribe_dataset_upload(body.get("dataset_id"), body.get("filename"), body.get("audio_base64"))
                self.send_json({"ok": True, "text": text})
                return
            if path == "/api/restore-upload":
                self.send_json({"ok": True, "item": add_restore_audio(body.get("filename"), body.get("audio_base64"))})
                return
            if path == "/api/restore-run":
                self.send_json({"ok": True, "results": start_restore(body.get("item_ids"), body.get("settings"))})
                return
            if path == "/api/restore-delete":
                delete_restore_item(str(body.get("item_id") or ""))
                self.send_json({"ok": True})
                return
            if path == "/api/sync-dataset":
                sync_dataset_job(body.get("dataset_id") or "")
                self.send_json({"ok": True})
                return
            if path == "/api/prepare-dataset":
                with gpu_reserved():
                    prepare_tts_dataset_job(
                        body.get("dataset_id") or "",
                        body.get("model_name") or "voice-model",
                    )
                self.send_json({"ok": True})
                return
            if path == "/api/start-api":
                gpt, sovits = model_pair(body.get("gpt"), body.get("sovits"))
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
                ensure_relay_asr()
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
                with gpu_reserved():
                    start_train(
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
                proc = JOBS.get(key)
                if proc:
                    proc.stop()
                elif key in WSL_STOP_SCRIPTS or key == "restore":
                    # Started before the dashboard restarted, so only WSL still knows the job.
                    stop_in_wsl(WSL_STOP_SCRIPTS["separate" if key == "restore" else key])
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
                            log(f"Discarded cancelled Mic to Voice line {generation_id}.")
                            self.send_json({"ok": False, "cancelled": True, "error": "Generation cancelled."}, 409)
                            return
                    if int(payload.get("streaming_mode", 0)):
                        data = repair_streamed_wav(data)
                    output = new_output_path(RELAY_OUTPUT_DIR if body.get("relay") else OUTPUT_DIR, body.get("voice"))
                    output.write_bytes(data)
                if body.get("relay"):
                    prune_relay_outputs()
                else:
                    save_speech_settings(body.get("voice"), speech_settings_from_generation(body))
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


def main():
    port = DASHBOARD_PORT
    # Launch-Voice-Dashboard.ps1 writes the output to a file, which Python would otherwise write
    # in the Windows code page, failing on names in other alphabets.
    for stream in (sys.stdout, sys.stderr):
        if stream and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    log(f"Local Voice UI starting on http://localhost:{port}")
    threading.Thread(target=kill_orphan_restore_workers, daemon=True).start()
    threading.Thread(target=clean_old_temp_files, daemon=True).start()
    if CONFIG_ERROR:
        log(CONFIG_ERROR)
    server = DashboardServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for key in ("ptt-helper", "relay-asr", "restore", "dataset-cut", "compare"):
            proc = JOBS.get(key)
            if proc and proc.running():
                proc.stop()
        if API_PROCESS and API_PROCESS.process and API_PROCESS.process.poll() is None:
            stop_api()


if __name__ == "__main__":
    main()
