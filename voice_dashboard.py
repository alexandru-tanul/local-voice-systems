import json
import base64
import contextlib
import ctypes
import hashlib
import io
import os
import re
import shutil
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
    for _ in range(80):
        if relay_asr_ready():
            return
        time.sleep(0.25)
    raise RuntimeError("Speech recognition did not start. See Logs for details.")


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


def start_train(kind, epochs, batch_size, save_every, dataset_id="", model_name="voice-model"):
    descriptor = dataset_descriptor(dataset_id)
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
        "version": "v2Pro",
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
        if manifest.get("version") != "v2Pro":
            raise RuntimeError(f"This voice is for GPT-SoVITS {manifest.get('version')}, and the dashboard uses v2Pro.")
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
        sovits_path = install_weight(bundle, sovits, "SoVITS_weights_v2Pro")
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
    model_files()
    log(f"Imported the voice {name}.")
    return {"dataset": dataset_id, "name": name, "gpt": gpt_path, "sovits": sovits_path, "reference": "reference"}


COMPARE_DIR = OUTPUT_DIR / "compare"


def model_epochs(model_name, kind):
    """The saved checkpoints of a trained model, as (epoch, path) pairs. GPT training saves
    name-eN.ckpt and SoVITS training name_eN_sSTEPS.pth."""
    name = re.escape(slugify(model_name, "voice-model"))
    pattern = re.compile(rf"^{name}-e(\d+)\.ckpt$" if kind == "gpt" else rf"^{name}_e(\d+)_s\d+\.pth$")
    rows = [(int(match.group(1)), row["path"]) for row in model_files()[kind] if (match := pattern.match(row["name"]))]
    return sorted(rows)


def start_compare(model_name, kind, dataset_id, reference_wav, text):
    """Generate one line with each saved epoch of a model, the other model at its last epoch."""
    if kind not in {"gpt", "sovits"}:
        raise RuntimeError("Choose GPT or SoVITS epochs to compare.")
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
    epochs = model_epochs(model_name, kind)
    if not epochs:
        raise RuntimeError(f"{slugify(model_name, 'voice-model')} has no saved {'GPT' if kind == 'gpt' else 'SoVITS'} epochs yet.")
    other_kind = "sovits" if kind == "gpt" else "gpt"
    others = model_epochs(model_name, other_kind)
    # Without saved epochs of the other model, the pretrained one GPT-SoVITS starts from.
    other = others[-1][1] if others else next((row["path"] for row in model_files()[other_kind] if "pretrained" in row["path"]), "")
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
                start_compare(body.get("model"), body.get("kind"), body.get("dataset_id"), body.get("reference"), body.get("text"))
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
      min-height: var(--control-height);
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
    /* Logs and System Check are tools, so a line sets them apart from the three voice pages. */
    .tab-button.tab-tool { position: relative; margin-left: var(--space-4); }
    .tab-button.tab-tool::before { content: ""; position: absolute; left: calc(-1 * var(--space-3)); top: 8px; bottom: 8px; border-left: 1px solid var(--line); }
    header .guide-button { justify-self: end; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    section.relay-page.active, section.side-page.active { display: grid; }
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
    .mini-button { white-space: nowrap; }
    /* A square button of the standard height, for an icon. */
    .icon-button { display: inline-grid; place-items: center; width: var(--control-height); padding: 0; }
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
    /* The dashboard's own audio player: one row the size of a text field. */
    .player { display: flex; align-items: center; gap: var(--space-3); width: 100%; min-width: 0; height: var(--control-height); padding-right: 12px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--field); }
    audio[hidden] + .player { display: none; }
    .player .player-toggle { flex: 0 0 auto; display: grid; place-items: center; width: calc(var(--control-height) - 2px); height: 100%; min-height: 0; padding: 0; border: 0; border-right: 1px solid var(--line); border-radius: var(--radius) 0 0 var(--radius); background: transparent; color: var(--text); }
    .player .player-toggle:hover:not(:disabled) { background: var(--panel-3); border-color: var(--line); }
    .player .player-seek { flex: 1 1 auto; min-width: 40px; height: auto; min-height: 0; }
    .player-time { flex: 0 0 auto; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
    .player .player-volume { flex: 0 0 auto; display: grid; place-items: center; width: calc(var(--control-height) - 2px); height: 100%; min-height: 0; padding: 0; border: 0; border-left: 1px solid var(--line); border-radius: 0 var(--radius) var(--radius) 0; background: transparent; color: var(--text); }
    .player .player-volume:hover:not(:disabled) { background: var(--panel-3); border-color: var(--line); }
    .player.has-volume { position: relative; padding-right: 0; }
    /* The volume slider opens above its player, so the position bar keeps its room. */
    .player-volume-panel { position: absolute; right: 0; bottom: calc(100% + 6px); z-index: 30; display: flex; align-items: center; gap: var(--space-2); width: 200px; height: var(--control-height); padding: 0 12px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--panel); box-shadow: 0 8px 24px rgba(0, 0, 0, .4); }
    .player-volume-panel input[type="range"] { flex: 1 1 auto; height: auto; min-height: 0; }
    .player-volume-value { flex: 0 0 auto; min-width: 36px; text-align: right; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
    .relay-block-head { display: flex; align-items: center; justify-content: space-between; gap: var(--space-2); min-height: 32px; }
    .relay-block-head label { margin: 0; }
    .relay-volume-row { display: grid; grid-template-columns: auto minmax(0, 1fr) auto; align-items: center; gap: var(--space-3); }
    .relay-volume-row label { margin: 0; }
    #tab-generate textarea { margin: 0; }
    #targetText { min-height: 160px; }
    #generateSettingsDialog .check-list { height: 190px; margin-top: 0; }
    #generateSettingsDialog .selected-list { margin-top: 0; min-height: var(--control-height); }
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
    .training-project { display: grid; align-content: start; }
    .progress-bar { height: 4px; background: var(--line-subtle); border-radius: 2px; overflow: hidden; margin-top: var(--space-2); }
    .progress-bar span { display: block; height: 100%; width: 0; background: var(--accent); transition: width .4s; }
    .compare-list { display: grid; gap: var(--space-3); }
    .compare-row { display: grid; grid-template-columns: 9em minmax(0, 1fr) auto; gap: var(--space-2); align-items: center; }
    @media (max-width: 600px) { .compare-row { grid-template-columns: minmax(0, 1fr) auto; grid-auto-flow: dense; } .compare-row .player { grid-column: 1 / -1; } }
    .training-project input + label { margin-top: var(--space-3); }
    /* The file chooser as a plain button, without the field box around it. */
    #datasetAudioFiles { border: 0; background: transparent; padding: 0; min-height: 0; color: var(--muted); }
    #datasetAudioFiles::file-selector-button { height: var(--control-height); margin-right: 12px; padding: 0 14px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--panel-3); color: var(--text); font: inherit; font-weight: 700; cursor: pointer; }
    #datasetAudioFiles::file-selector-button:hover { background: #2a2f34; border-color: #444a51; }
    .upload-list { display: grid; gap: var(--space-2); max-height: 310px; overflow: auto; margin-top: var(--space-3); }
    .upload-row { display: grid; grid-template-columns: auto minmax(150px,.7fr) minmax(260px,1.5fr) auto; gap: var(--space-2); align-items: center; border: 1px solid var(--line-subtle); padding: var(--space-2); background: #101214; }
    .clip-summary { color: var(--muted); }
    .clip-summary.warn { color: var(--warn); }
    .clip-list { display: grid; gap: var(--space-1); max-height: 440px; overflow-y: auto; padding-right: var(--space-1); }
    .clip-row { display: grid; grid-template-columns: auto minmax(0, 1fr) auto auto; gap: var(--space-2); align-items: center; }
    .clip-row .clip-length { color: var(--muted); font-size: 12px; white-space: nowrap; text-align: right; min-width: 5.5em; }
    .clip-row.warn .clip-length { color: var(--warn); }
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
      min-height: var(--control-height);
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
      padding: 0 0 0 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
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
    .status-pill strong { font-weight: 600; color: var(--text); display: inline-block; min-width: 50px; }
    .status-pill.ready .dot { background: var(--good); }
    .status-pill.ready strong { color: var(--good); }
    .status-pill.starting .dot { background: var(--warn); animation: rec-blink 1s steps(2, start) infinite; }
    .status-pill.starting strong { color: var(--warn); }
    .status-pill.failed .dot { background: var(--bad); }
    .status-pill.failed strong { color: var(--bad); }
    .island-system-state {
      font-size: 12px;
      color: var(--muted);
      width: 180px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      padding: 0 4px;
    }
    .island-system-state.good { color: var(--good); }
    .island-system-state.bad { color: var(--bad); }
    .island button { white-space: nowrap; }
    /* The bar is one control high. Stop Everything sits flush at its right end, like the play
       button in a player. */
    .island .island-group:last-child { align-self: stretch; }
    .island #stopSystem { border: 0; border-left: 1px solid #704142; border-radius: 0 7px 7px 0; }
    label.switch {
      position: relative;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: var(--control-height);
      margin: 0;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      color: var(--text);
      cursor: pointer;
      white-space: nowrap;
    }
    label.switch:hover { background: var(--panel-3); }
    .relay-options { display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: var(--space-2); }
    .workspace[data-view="result"] .relay-options { flex-basis: 100%; justify-content: flex-start; }
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
    .small-dialog { width: min(600px, calc(100vw - 32px)); }
    .small-dialog h2 { margin: 0; }
    .small-dialog p { margin: 0; }
    /* Guided tour. The spot's shadow darkens everything except the highlighted element. */
    .tour { position: fixed; inset: 0; z-index: 100; }
    .tour.no-spot { background: rgba(0, 0, 0, .6); }
    .tour-spot { position: fixed; border-radius: 8px; box-shadow: 0 0 0 9999px rgba(0, 0, 0, .6); outline: 2px solid var(--accent); pointer-events: none; transition: top .2s ease, left .2s ease, width .2s ease, height .2s ease; }
    .tour-pop { position: fixed; width: min(380px, calc(100vw - 24px)); display: grid; gap: var(--space-3); padding: var(--space-4); border: 1px solid var(--line); border-radius: 8px; background: var(--panel); box-shadow: 0 20px 60px rgba(0, 0, 0, .5); }
    .tour-pop h2 { margin: 0; font-size: 16px; }
    .tour-pop p { margin: 0; line-height: 1.5; }
    .tour-count { font-size: 12px; color: var(--muted); }
    .tour-actions { display: flex; align-items: center; gap: var(--space-2); }
    .tour-actions #tourEnd { margin-right: auto; }
    /* Shown instead of the page's controls while no dataset has clips. */
    .no-voice-note { display: none; max-width: 460px; text-align: center; }
    .no-voice-note p { margin: 0 0 var(--space-3); }
    body[data-voice="no"] .no-voice-note { display: block; }
    body[data-voice="no"] .compose,
    body[data-voice="no"] #relayCapture > :not(.no-voice-note) { display: none; }
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
    .panel h4, .training-project h4 { margin: 0 0 var(--space-3); font-size: 13px; }
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
    /* Text to Voice and Voice Training use the Mic to Voice layout: a list on the left, open space on the right. */
    section.side-page {
      grid-template-columns: 272px minmax(0, 1fr);
      gap: var(--space-5);
      align-items: start;
      background: transparent;
      border: 0;
      box-shadow: none;
      padding: 0;
      min-height: calc(100vh - var(--header-height) - var(--island-height) - 48px);
    }
    /* On wide windows an empty column on the right, as wide as the list on the left, centers the
       middle of each page on the window, in line with the tabs and the bottom bar. */
    @media (min-width: 1280px) {
      section.relay-page, section.side-page { grid-template-columns: 272px minmax(0, 1fr) 272px; }
    }
    /* Voice Training has two modes: training a dataset, and restoring recordings for one. */
    #tab-training:not([data-mode="restore"]) .restore-only,
    #tab-training[data-mode="restore"] .train-only { display: none !important; }
    .mode-tabs { display: flex; gap: var(--space-2); width: 100%; max-width: 920px; margin: 0 auto var(--space-5); }
    .mode-tabs button[aria-selected="true"] { border-color: var(--accent); box-shadow: inset 0 -2px 0 var(--accent); }
    .restore-row { display: grid; grid-template-columns: auto minmax(0, 1fr); align-items: center; gap: var(--space-2); }
    .restore-row input[type="checkbox"] { margin-left: var(--space-2); }
    .restore-options { display: grid; gap: var(--space-2); }
    .warn-text { color: var(--warn); }
    .compose { display: grid; gap: var(--space-4); width: 100%; max-width: 760px; }
    .compose textarea { padding: 14px 16px; font-size: 16px; line-height: 1.5; }
    /* The preview player gets a wider column, so its position bar stays usable beside the volume button. */
    .voice-line { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.3fr); gap: var(--space-3); align-items: end; }
    .voice-line > * { min-width: 0; }
    .compose-actions { display: flex; align-items: center; justify-content: center; gap: var(--space-3); flex-wrap: wrap; }
    .compose-status { text-align: center; min-height: 20px; }
    .generate-result { display: grid; gap: 6px; }
    .training-workspace { justify-content: flex-start; align-items: stretch; }
    .steps { display: grid; gap: 44px; width: 100%; max-width: 920px; margin: 0 auto; }
    .step { display: grid; gap: var(--space-3); }
    .step-head { display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); flex-wrap: wrap; }
    .step-head h3 { margin: 0; font-size: 15px; display: flex; align-items: center; }
    .empty-note { max-width: 460px; margin: auto; text-align: center; }
    .empty-note p { margin: 0 0 var(--space-3); }
    .settings-body h3 .status-line { margin-left: 8px; text-transform: none; letter-spacing: 0; }
    section.page .panel { border: 0; background: transparent; padding: 0; }
    .log-tools { display: flex; align-items: center; gap: var(--space-2); flex-wrap: wrap; }
    .log-tools input:not([type="checkbox"]) { width: 220px; }
    .log-tools label { margin: 0; min-height: var(--control-height); }
    section.page .log { height: calc(100vh - var(--header-height) - var(--island-height) - 120px); min-height: 280px; }
    .setup-list { margin-top: var(--space-3); }
    .setup-list:empty { display: none; }
    @media (max-width: 1020px) {
      .grid2 { grid-template-columns: 1fr; }
      .grid3 { grid-template-columns: 1fr; }
      .setup-scope-grid { grid-template-columns: 1fr; }
      header { grid-template-columns: 1fr; }
      header .tablist { justify-content: flex-start; }
      header .guide-button { position: absolute; top: 10px; right: var(--space-5); }
      section.relay-page { grid-template-columns: 1fr; gap: var(--space-3); }
      .history-toggle { display: inline-flex; gap: 4px; justify-self: start; }
      section.side-page { grid-template-columns: 1fr; gap: var(--space-3); min-height: 0; }
      section.side-page .history { position: static; max-height: 240px; }
      section.side-page .workspace { min-height: 0; justify-content: flex-start; padding-top: 0; }
      .relay-page .history {
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
      .relay-page .history.open { display: flex; }
      .workspace { padding: var(--space-3); min-height: calc(100vh - 200px); }
      section.relay-page { min-height: 0; }
    }
    @media (max-width: 760px) {
      :root { --island-height: 150px; }
      body { padding-bottom: 160px; }
      .island { gap: 6px; padding: 7px 12px; }
      .island .island-group:last-child { align-self: auto; }
      .island #stopSystem { border: 1px solid #704142; border-radius: var(--radius); }
      .island-group + .island-group { border-left: 0; padding-left: 0; margin-left: 0; }
      .island-system-state { width: 160px; }
      header { padding: 8px var(--space-3); gap: var(--space-2); }
      .tab-button { padding: 4px 8px; font-size: 11px; letter-spacing: .04em; }
      .tab-button.tab-tool { margin-left: 0; }
      .voice-line { grid-template-columns: 1fr; }
      .tab-button.tab-tool::before { display: none; }
      header .guide-button { top: 8px; right: var(--space-3); }
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
      .upload-row { grid-template-columns: auto minmax(0, 1fr) auto; }
      .upload-row input { grid-column: 1 / -1; grid-row: 2; }
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
      <button class="tab-button" data-tab="relay" role="tab" aria-selected="false">Mic to Voice</button>
      <button class="tab-button" data-tab="generate" role="tab" aria-selected="false">Text to Voice</button>
      <button class="tab-button" data-tab="training" role="tab" aria-selected="false">Voice Training</button>
      <button class="tab-button tab-tool" data-tab="logs" role="tab" aria-selected="false">Logs</button>
      <button class="tab-button" data-tab="setup" role="tab" aria-selected="false">System Check</button>
    </nav>
    <button class="secondary mini-button guide-button" id="openGuide" title="Take a tour of the dashboard">Guide</button>
    <div class="sr-only" id="recentActivity" aria-live="polite">Dashboard loaded. System is stopped.</div>
  </header>
  <div class="global-error" id="globalError"><span id="globalErrorText"></span><div class="row"><button class="secondary mini-button" id="openErrorLogs">Open Logs</button><button class="danger mini-button" id="dismissError">Dismiss</button></div></div>

  <main>
      <section id="tab-generate" class="tab-panel side-page" role="tabpanel">
        <aside class="history" aria-label="Generated lines">
          <button class="secondary history-new" id="generateNewLine">+ New line</button>
          <div class="history-list" id="generateHistory" role="listbox" aria-label="Lines generated in this session"></div>
          <div class="history-foot"><span class="small" id="generateHistoryCount">Nothing generated yet</span></div>
        </aside>
        <div class="workspace">
          <div class="no-voice-note">
            <p>You have no voice yet. On Voice Training, create a dataset, add one clean 3 to 10 second clip with its exact words, and press Sync Only.</p>
            <div class="row" style="justify-content: center;"><button data-go="training">Go to Voice Training</button><button class="secondary" data-import-voice>Import a voice</button></div>
          </div>
          <div class="compose">
            <label class="sr-only" for="targetText">Text to speak</label>
            <textarea id="targetText" placeholder="Type what the voice should say">Please remain calm. Your message is being processed.</textarea>
            <div class="voice-line">
              <div><label for="voiceDataset">Voice</label><select id="voiceDataset"></select></div>
              <div><label for="reference">Reference clip (3-10 s)</label><select id="reference"></select></div>
              <div><label>Preview</label><audio id="refAudio" data-label="reference clip"></audio></div>
            </div>
            <div class="compose-actions">
              <button id="generate">Generate</button>
              <button class="secondary icon-button" id="openGenerateSettings" aria-haspopup="dialog" aria-label="Settings" title="Settings: reference text, extra clips, models, and generation settings"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></button>
            </div>
            <div class="status-line compose-status" id="generateStatus" role="status"></div>
            <div class="generate-result" id="generateResult" hidden>
              <audio id="outputAudio" data-label="generated line"></audio>
              <div class="small output-name" id="outputName"></div>
            </div>
          </div>
        </div>
      </section>

      <section id="tab-relay" class="tab-panel relay-page" role="tabpanel">
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
            <div class="no-voice-note">
              <p>You have no voice yet. Create one on Voice Training, then come back here to speak through it.</p>
              <div class="row" style="justify-content: center;"><button data-go="training">Go to Voice Training</button><button class="secondary" data-import-voice>Import a voice</button></div>
            </div>
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
            <div class="relay-options">
              <button class="mini-button" id="startSystem" title="Start speech recognition, the voice engine, and the global push-to-talk key" hidden>Start</button>
              <label class="switch" title="On: record while the key or button is held. Off: voice activation records whenever speech is heard."><input id="relayPushToTalk" type="checkbox" checked /><span class="switch-track" aria-hidden="true"></span>Hold to talk<span class="switch-state" aria-hidden="true"></span></label>
              <label class="switch" title="On: each transcript is spoken in the selected voice automatically. Off: transcribe only."><input id="relayAutoSpeak" type="checkbox" checked /><span class="switch-track" aria-hidden="true"></span>Speak automatically<span class="switch-state" aria-hidden="true"></span></label>
              <button class="secondary icon-button" id="openRelaySettings" aria-haspopup="dialog" aria-label="Settings" title="Settings: output device, volume, voice models, and generation settings"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></button>
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
                <audio id="relayAudio" data-label="generated line" data-settings-volume></audio>
                <button class="secondary mini-button" id="relayPlayThis" hidden>Play</button>
                <button class="secondary mini-button" id="relayReplayLast">Replay</button>
              </div>
            </div>
            <div class="result-actions" id="relayResultActions"></div>
            <div class="result-meta" id="relayNowPlaying"></div>
          </div>
        </div>
      </section>

      <section id="tab-training" class="tab-panel side-page" role="tabpanel">
        <aside class="history train-only" aria-label="Datasets">
          <button class="secondary history-new" id="newDatasetButton" aria-haspopup="dialog">+ New dataset</button>
          <div class="history-list" id="datasetList" role="listbox" aria-label="Datasets"></div>
          <div class="history-foot"><span class="small" id="datasetListCount">No datasets yet</span></div>
        </aside>
        <aside class="history restore-only" aria-label="Recordings to restore">
          <button class="secondary history-new" id="restoreAddButton">+ Add audio files</button>
          <input id="restoreFiles" type="file" accept="audio/*,.wav,.ogg,.mp3,.flac,.m4a" multiple hidden />
          <div class="history-list" id="restoreList" aria-label="Recordings to restore"></div>
          <div class="history-foot"><span class="small" id="restoreCount">No audio yet</span></div>
        </aside>
        <div class="workspace training-workspace">
          <!-- The selected dataset. The list on the left sets it. -->
          <select id="trainingDataset" hidden></select>
          <div class="mode-tabs" role="tablist" aria-label="Voice Training">
            <button class="secondary" role="tab" id="trainingModeTrain" data-mode="train" aria-selected="true">Train</button>
            <button class="secondary" role="tab" id="trainingModeRestore" data-mode="restore" aria-selected="false">Restore audio</button>
          </div>
          <div class="empty-note train-only" id="noDatasets" hidden>
            <p>No voice yet. A voice starts as a dataset: one clean 3 to 10 second clip and its exact words.</p>
            <button id="newDatasetEmpty" aria-haspopup="dialog">Create a dataset</button>
          </div>
          <div class="steps train-only" id="trainingSteps">
            <div class="step" id="clipsStep">
              <div class="step-head"><h3><span class="step-number">1</span>Clips</h3><span class="status-line" id="trainingDatasetStatus"></span></div>
              <div class="small clip-summary" id="clipSummary"></div>
              <div class="clip-list" id="clipList" aria-label="Clips in the dataset"></div>
              <div class="row">
                <button class="secondary" id="datasetAddFiles">+ Add audio</button>
                <input id="datasetAudioFiles" type="file" accept="audio/*,.wav,.ogg,.mp3,.flac,.m4a" multiple hidden aria-label="Audio files to add" />
                <button class="danger" data-stop="dataset-cut" id="datasetCutStop" hidden>Stop cutting</button>
                <span class="status-line" id="datasetCutStatus"></span>
              </div>
              <div class="small">Recordings over 20 seconds are cut into clips at pauses.</div>
              <div class="small" id="datasetUploadHint" hidden>Transcripts are written automatically. Check each one, then add the files.</div>
              <div class="upload-list" id="datasetUploadRows"></div>
              <div class="row"><button id="uploadDatasetAudio" disabled hidden>Add Files to Dataset</button><span class="status-line" id="datasetUploadProgress"></span></div>
            </div>

            <div class="step" id="prepareStep">
              <div class="step-head"><h3><span class="step-number">2</span>Prepare</h3><span class="status-line" id="datasetPrepareStatus">Not prepared in this session.</span></div>
              <div class="small">Sync Only copies the clips to WSL, which is all Text to Voice and Mic to Voice need. Sync &amp; Prepare also extracts the features that training needs, on the GPU.</div>
              <div class="field-row">
                <div>
                  <label for="trainingModelName">Model name</label>
                  <input id="trainingModelName" value="voice-model" placeholder="Example: station-announcer" />
                  <div class="small" style="margin-top: 6px;">Saved checkpoints use: <strong id="trainingModelSlug">voice-model</strong></div>
                </div>
              </div>
              <div class="row">
                <button class="secondary" id="syncDataset">Sync Only</button>
                <button id="prepareDataset">Sync &amp; Prepare</button>
                <button class="danger" data-stop="dataset-prepare">Stop Preparation</button>
              </div>
            </div>

            <div class="step" id="trainPanel">
              <div class="step-head"><h3><span class="step-number">3</span>Train</h3><span class="status-line" id="trainingJobsStatus"></span></div>
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
                  <div class="progress-bar" id="trainSovitsBar" hidden><span></span></div>
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
                  <div class="progress-bar" id="trainGptBar" hidden><span></span></div>
                </div>
              </div>
            </div>

            <div class="step" id="compareStep">
              <div class="step-head"><h3><span class="step-number">4</span>Compare epochs</h3><span class="status-line" id="compareStatus"></span></div>
              <div class="small" id="compareInfo">Generates one line with each saved epoch, so you can choose the best by ear.</div>
              <div class="field-row">
                <div><label for="compareKind">Compare</label><select id="compareKind"><option value="sovits">SoVITS epochs</option><option value="gpt">GPT epochs</option></select></div>
                <div><label for="compareReference">Reference clip</label><select id="compareReference"></select></div>
              </div>
              <div><label for="compareText">Test line</label><input id="compareText" value="This is how my new voice sounds." /></div>
              <div class="row"><button id="compareRun">Compare</button><button class="danger" data-stop="compare" id="compareStop" hidden>Stop</button></div>
              <div class="compare-list" id="compareResults"></div>
            </div>

            <details class="sub">
              <summary>Saved models <span class="small" id="modelCount"></span></summary>
              <div class="sub-body"><div class="models" id="modelList" style="border-top: 0; padding-top: 0; margin-top: 0;"></div></div>
            </details>
          </div>
          <div class="restore-only restore-panel">
            <div class="empty-note" id="restoreEmpty" hidden>
              <p>Clean up noisy or low-quality recordings before training.</p>
              <button id="restoreAddEmpty">Add audio files</button>
            </div>
            <div class="steps" id="restoreSteps" hidden>
              <div class="step">
                <div class="step-head"><h3 id="restoreTitle"></h3><span class="status-line" id="restoreInfo"></span></div>
                <div id="restoreParentBlock" hidden><label>Before</label><audio id="restoreParent" data-label="recording before restoring"></audio></div>
                <div><label id="restoreItemLabel">Recording</label><audio id="restoreItem" data-label="recording"></audio></div>
                <div class="small" id="restoreHistory"></div>
                <div class="row">
                  <button class="secondary" id="restoreToDataset">Add to dataset</button>
                  <button class="secondary" id="restoreDownload">Download</button>
                  <button class="danger" id="restoreDelete">Delete</button>
                </div>
              </div>
              <div class="step">
                <div class="step-head"><h3>Restore</h3><span class="status-line" id="restoreStatus"></span></div>
                <div class="restore-options">
                  <label><input type="checkbox" id="restoreMusic" /> Remove music and sound effects</label>
                  <div><select id="restoreMusicModel" aria-label="Separation model">
                    <option value="best">BS-Roformer, best quality</option>
                    <option value="main">HP5, main voice only</option>
                    <option value="all">HP2, all voices</option>
                  </select></div>
                  <label><input type="checkbox" id="restoreEcho" /> Remove echo and reverb</label>
                  <div class="small">These two run on the GPU in WSL.</div>
                  <div class="small warn-text" id="restoreMissing" hidden>Noise removal and high frequencies need <strong>Setup-Restore-Env.bat</strong>.</div>
                  <label><input type="checkbox" id="restoreDenoise" /> Remove background noise</label>
                  <div class="small">Best with Restore high frequencies on.</div>
                  <label><input type="checkbox" id="restoreEnhance" checked /> Restore high frequencies</label>
                </div>
                <div class="field-row">
                  <div><label for="restoreQuality">Original recording quality</label><select id="restoreQuality">
                    <option value="detect">Detect from the audio</option>
                    <option value="file">The file's own quality</option>
                    <option value="8000">8 kHz</option>
                    <option value="16000">16 kHz</option>
                    <option value="22050">22.05 kHz</option>
                    <option value="24000">24 kHz</option>
                    <option value="32000">32 kHz</option>
                    <option value="44100">44.1 kHz</option>
                    <option value="48000">48 kHz</option>
                  </select></div>
                  <div><label for="restoreKeep">Keep the original sound up to (kHz)</label><input id="restoreKeep" type="number" min="1" max="24" step="0.5" placeholder="Automatic" /></div>
                  <div><label for="restoreModel">LavaSR model</label><select id="restoreModel"><option value="v2">Version 2 (newer)</option><option value="v1">Version 1</option></select></div>
                </div>
                <label><input type="checkbox" id="restorePieces" /> Work in short pieces (for long recordings)</label>
                <div class="row"><button id="restoreRun">Restore</button><button class="danger" data-stop="restore">Stop</button></div>
              </div>
            </div>
          </div>
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
        <div class="page-head"><h2>System Check</h2><span class="small">A read-only check of the software, models, GPU access, and audio devices the voice tools need. It starts nothing and changes nothing.</span></div>
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
      <button class="danger mini-button" id="stopSystem" title="Stop all managed services and the microphone. Recordings stay listed.">Stop Everything</button>
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
            <audio id="relayRefAudio" data-label="reference clip"></audio>
          </div>
        </div>
        <details class="sub">
          <summary>Extra reference clips <span class="small" id="relayAuxCount">0 selected</span></summary>
          <div class="sub-body">
            <div class="small">Optional additional clips for style. More clips can improve delivery but slow generation.</div>
            <div class="row"><input id="relayAuxSearch" placeholder="Search clips" aria-label="Search clips" /><button class="secondary mini-button" id="relayClearAuxRefs">Clear</button></div>
            <div id="relayAuxRefsList" class="check-list" style="height: 190px; margin-top: 0;"></div>
            <div id="relayAuxSelectedList" class="selected-list" style="margin-top: 0;"></div>
            <audio id="relayAuxPreview" data-label="clip"></audio>
            <div class="small" id="relayAuxPreviewStatus"></div>
          </div>
        </details>
      </div>
      <div>
        <h3>Generation</h3>
        <label>Voice preset</label>
        <div class="preset-row">
          <button class="secondary mini-button" type="button" data-relay-profile="fast">Fast</button>
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

  <dialog class="settings-dialog" id="generateSettingsDialog" aria-labelledby="generateSettingsTitle">
    <form method="dialog" class="settings-head">
      <h2 id="generateSettingsTitle">Text to Voice settings</h2>
      <button class="secondary mini-button" value="close">Close</button>
    </form>
    <div class="settings-body">
      <div>
        <h3>Reference clip</h3>
        <label for="promptText">Text spoken in the reference clip</label>
        <textarea id="promptText"></textarea>
        <div class="small">Filled from the clip. Edit it only if the clip says something else.</div>
      </div>
      <details class="sub">
        <summary>Extra reference clips <span class="small" id="auxCount">0 selected</span></summary>
        <div class="sub-body">
          <div class="small">Optional additional clips for style. More clips can improve delivery but slow generation.</div>
          <div class="row"><input id="auxSearch" placeholder="Search clips" aria-label="Search clips" /><button class="secondary mini-button" id="clearAuxRefs">Clear</button></div>
          <div id="auxRefsList" class="check-list"></div>
          <div id="auxSelectedList" class="selected-list"></div>
          <audio id="auxPreview" data-label="clip"></audio>
          <div class="small" id="auxPreviewStatus"></div>
        </div>
      </details>
      <div>
        <h3>Engine and models <span class="status-line" id="engineStatus"></span></h3>
        <div class="field-row">
          <div><label for="gptModel">GPT model</label><select id="gptModel"></select></div>
          <div><label for="sovitsModel">SoVITS model</label><select id="sovitsModel"></select></div>
        </div>
        <div class="row" style="margin-top: 10px;"><button class="secondary" id="startApi">Start Engine</button><button class="secondary" id="stopApi">Stop Engine</button><span class="small">Generate starts the engine on its own when it is off.</span></div>
      </div>
      <div>
        <h3>Share this voice <span class="status-line" id="shareStatus"></span></h3>
        <div class="row">
          <button class="secondary" id="exportVoice">Export voice</button>
          <button class="secondary" data-import-voice>Import a voice</button>
        </div>
        <div class="small">One file with both models and the reference clips, to give to someone or to keep.</div>
      </div>
      <div>
        <h3>Generation</h3>
        <div class="grid3">
          <div><label for="textLang" title="Language of the text to speak. It follows the voice dataset's language when you choose a dataset.">Text language</label><select id="textLang"><option value="en">English</option><option value="zh">Chinese</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="yue">Cantonese</option></select></div>
          <div><label for="splitMethod">Slice method</label><select id="splitMethod">
            <option value="cut5">English punctuation</option>
            <option value="cut4">Periods only</option>
            <option value="cut0">Do not split</option>
          </select></div>
          <div><label for="seed">Seed</label><input id="seed" type="number" value="-1" /></div>
          <div><label for="topK">Top K</label><input id="topK" type="number" min="1" max="100" value="15" /></div>
          <div><label for="repPenalty">Repetition penalty</label><input id="repPenalty" type="number" step="0.05" min="1" max="2" value="1.35" /></div>
          <div><label for="topP">Top P</label><input id="topP" type="number" step="0.05" min="0" max="1" value="0.6" /></div>
          <div><label for="temperature">Temperature</label><input id="temperature" type="number" step="0.05" min="0" max="1" value="0.6" /></div>
          <div><label for="speed">Speed</label><input id="speed" type="number" step="0.05" min="0.5" max="1.5" value="1" /></div>
          <div><label for="pause">Pause</label><input id="pause" type="number" step="0.05" min="0" max="2" value="0.3" /></div>
        </div>
      </div>
    </div>
  </dialog>

  <dialog class="settings-dialog small-dialog" id="createDatasetDialog" aria-labelledby="createDatasetTitle">
    <form method="dialog" class="settings-head">
      <h2 id="createDatasetTitle">New dataset</h2>
      <button class="secondary mini-button" value="close">Close</button>
    </form>
    <div class="settings-body">
      <div class="field-row">
        <div><label for="newDatasetName">Dataset name</label><input id="newDatasetName" placeholder="Example: Station Announcer" /></div>
        <div><label for="newDatasetLanguage">Language</label><select id="newDatasetLanguage"><option value="en">English</option><option value="zh">Chinese</option><option value="ja">Japanese</option><option value="ko">Korean</option><option value="yue">Cantonese</option></select></div>
      </div>
      <div><label for="newDatasetSpeaker">Default speaker name</label><input id="newDatasetSpeaker" value="speaker" /></div>
      <div><button id="createDataset">Create Dataset</button></div>
    </div>
  </dialog>

  <dialog class="settings-dialog small-dialog" id="welcomeDialog" aria-labelledby="welcomeTitle">
    <div class="settings-body">
      <h2 id="welcomeTitle">Welcome to Local Voice Systems</h2>
      <p>Looks like this is your first time here. Do you want a quick rundown?</p>
      <div class="row">
        <button id="welcomeYes">Show me</button>
        <button class="secondary" id="welcomeNo">No thanks</button>
      </div>
      <div class="small">You can take the tour later with the Guide button at the top.</div>
    </div>
  </dialog>

  <div class="tour" id="tour" hidden>
    <div class="tour-spot" id="tourSpot"></div>
    <div class="tour-pop" id="tourPop" role="dialog" aria-modal="true" aria-labelledby="tourTitle" aria-describedby="tourText">
      <div class="tour-count" id="tourCount"></div>
      <h2 id="tourTitle"></h2>
      <p id="tourText"></p>
      <div class="tour-actions">
        <button class="secondary" id="tourEnd">End tour</button>
        <button class="secondary" id="tourBack">Back</button>
        <button id="tourNext">Next</button>
      </div>
    </div>
  </div>

  <script>
    // Settings saved by earlier versions used a different key prefix.
    ["RelaySettings", "RelayAuxSelected", "RelayPttKey", "RelayOutputDevice", "TrainingDataset", "TrainingModelName"].forEach(key => {
      try {
        const old = localStorage.getItem("workingJoe" + key);
        if (old !== null && localStorage.getItem("voiceDashboard" + key) === null) localStorage.setItem("voiceDashboard" + key, old);
      } catch (_) {}
    });
    // Settings of the removed voice changer page.
    try { ["voiceDashboardRvcDataset", "voiceDashboardRvcModelName", "voiceDashboardGuide"].forEach(key => localStorage.removeItem(key)); }
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
    const $ = id => document.getElementById(id);

    // The dashboard's own audio player in place of the browser's: a play button, a position bar
    // and the time. Hiding an audio element hides its player too.
    const PLAY_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
    const PAUSE_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>';
    function clockTime(seconds) {
      if (!Number.isFinite(seconds)) return "0:00";
      return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
    }
    function attachPlayer(audio) {
      const label = audio.dataset.label || "audio";
      const bar = document.createElement("div");
      bar.className = "player";
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "player-toggle";
      const seek = document.createElement("input");
      seek.type = "range";
      seek.className = "player-seek";
      seek.min = "0";
      seek.max = "1000";
      seek.value = "0";
      seek.setAttribute("aria-label", `Position in the ${label}`);
      const time = document.createElement("span");
      time.className = "player-time";
      bar.append(toggle, seek, time);
      audio.after(bar);
      let seeking = false;
      const update = () => {
        const loaded = Boolean(audio.currentSrc || audio.getAttribute("src"));
        toggle.disabled = !loaded;
        toggle.innerHTML = audio.paused ? PLAY_ICON : PAUSE_ICON;
        toggle.setAttribute("aria-label", `${audio.paused ? "Play" : "Pause"} the ${label}`);
        seek.disabled = !loaded || !Number.isFinite(audio.duration);
        if (!seeking) seek.value = audio.duration ? String(Math.round(audio.currentTime / audio.duration * 1000)) : "0";
        time.textContent = `${clockTime(audio.currentTime)} / ${clockTime(audio.duration)}`;
      };
      toggle.onclick = () => { if (audio.paused) audio.play().catch(() => {}); else audio.pause(); };
      seek.addEventListener("input", () => {
        seeking = true;
        if (Number.isFinite(audio.duration)) audio.currentTime = seek.value / 1000 * audio.duration;
      });
      seek.addEventListener("change", () => { seeking = false; });
      ["play", "pause", "ended", "timeupdate", "loadedmetadata", "durationchange", "emptied", "loadstart"].forEach(name => audio.addEventListener(name, update));
      update();
      // Mic to Voice's output sets its volume in Settings, so its player has no volume button.
      if (!audio.hasAttribute("data-settings-volume")) addVolume(audio, bar, label);
    }

    // One volume for every player with a volume button, remembered between visits.
    const VOLUME_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 5 6 9H2v6h4l5 4z" fill="currentColor" stroke="none"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M18.5 5.5a9 9 0 0 1 0 13"/></svg>';
    const MUTED_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 5 6 9H2v6h4l5 4z" fill="currentColor" stroke="none"/><path d="m22 9-6 6M16 9l6 6"/></svg>';
    const volumeControls = [];
    let playerVolume = 1;
    try {
      const saved = Number(localStorage.getItem("voiceDashboardPlayerVolume") ?? 1);
      if (Number.isFinite(saved)) playerVolume = Math.max(0, Math.min(1, saved));
    } catch (_) {}

    function setPlayerVolume(volume) {
      playerVolume = volume;
      volumeControls.forEach(control => {
        control.audio.volume = volume;
        control.range.value = String(Math.round(volume * 100));
        control.value.textContent = `${Math.round(volume * 100)}%`;
        control.button.innerHTML = volume ? VOLUME_ICON : MUTED_ICON;
      });
      try { localStorage.setItem("voiceDashboardPlayerVolume", String(volume)); } catch (_) {}
    }

    function addVolume(audio, bar, label) {
      bar.classList.add("has-volume");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "player-volume";
      button.setAttribute("aria-label", `Volume of the ${label}`);
      button.setAttribute("aria-expanded", "false");
      const panel = document.createElement("div");
      panel.className = "player-volume-panel";
      panel.hidden = true;
      const range = document.createElement("input");
      range.type = "range";
      range.min = "0";
      range.max = "100";
      range.setAttribute("aria-label", `Volume of the ${label}`);
      const value = document.createElement("span");
      value.className = "player-volume-value";
      panel.append(range, value);
      bar.append(button, panel);
      volumeControls.push({audio, range, value, button});
      const open = show => {
        panel.hidden = !show;
        button.setAttribute("aria-expanded", show ? "true" : "false");
      };
      button.onclick = () => {
        open(panel.hidden);
        if (!panel.hidden) range.focus();
      };
      range.addEventListener("input", () => setPlayerVolume(Number(range.value) / 100));
      // Esc closes the slider, not the window it is in.
      panel.addEventListener("keydown", event => {
        if (event.key !== "Escape") return;
        event.preventDefault();
        event.stopPropagation();
        open(false);
        button.focus();
      });
      document.addEventListener("pointerdown", event => {
        if (!panel.hidden && !bar.contains(event.target)) open(false);
      });
      setPlayerVolume(playerVolume);
    }
    document.querySelectorAll("audio").forEach(attachPlayer);
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
    // Mic to Voice starts its services once each time the tab opens. After Stop Everything they
    // stay off until the tab is opened again or Start is pressed.
    let autoStartPending = false;
    let systemStartBusy = false;
    // Stop Everything raises it, so a start that is still waiting, for example for the microphone
    // prompt, does not go on.
    let startToken = 0;
    let lastGpuBusy = false;
    // A dataset created on this page, kept selected until a dataset list includes it: a list that
    // was requested before the dataset existed can arrive after it.
    let newTrainingDataset = null;
    let welcomeChecked = false;
    let lastSystemMessage = "";
    let datasetUploadFiles = [];
    // Transcripts of the files waiting to be added, typed or written by speech recognition, so a
    // redraw of the list keeps them. A file without one is still being transcribed.
    const datasetUploadTexts = new WeakMap();
    const datasetUploadInputs = new WeakMap();
    const transcriptQueue = [];
    let transcriptBusy = false;
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
    // The generated line shown in the player.
    let generateShown = null;
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

    function datasetLabel(dataset) {
      return `${dataset.name} — ${dataset.wav_count} ${dataset.wav_count === 1 ? "clip" : "clips"}`;
    }

    function fillDatasetSelects(datasets, activeId) {
      ["voiceDataset", "relayDataset"].forEach(id => {
        const select = $(id);
        select.innerHTML = "";
        datasets.forEach(dataset => option(select, dataset.id, datasetLabel(dataset)));
        if (datasets.some(dataset => dataset.id === activeId)) select.value = activeId;
      });
      if (activeId !== voiceDataset) {
        voiceDataset = activeId;
        try { localStorage.setItem("voiceDashboardVoiceDataset", voiceDataset); } catch (_) {}
      }
    }

    function fillTrainingDatasets(datasets) {
      const select = $("trainingDataset");
      const old = (newTrainingDataset && newTrainingDataset.id) || select.value || localStorage.getItem("voiceDashboardTrainingDataset") || "";
      select.innerHTML = "";
      (datasets || []).forEach(dataset => option(select, dataset.id, datasetLabel(dataset)));
      if (newTrainingDataset && (datasets || []).some(dataset => dataset.id === newTrainingDataset.id)) newTrainingDataset = null;
      else if (newTrainingDataset) option(select, newTrainingDataset.id, newTrainingDataset.name);
      if (Array.from(select.options).some(item => item.value === old)) select.value = old;
      const selected = (datasets || []).find(dataset => dataset.id === select.value);
      if (selected) {
        $("trainingDatasetStatus").textContent = selected.ready ? "Ready" : (selected.wav_count || selected.list_rows ? "Clips and transcripts do not match" : "No clips yet");
        $("trainingDatasetStatus").className = selected.ready ? "status-line good" : "status-line warn";
        updateUploadButton();
      }
      renderDatasetList(datasets || [], select.value);
      loadClips();
    }

    // The clips already in the selected dataset: listen, correct a transcript, or delete a clip.
    // Rows are kept while the list grows, so typing in one is not interrupted.
    let clipsSignature = "";
    let clipsDataset = "";
    let clipsData = {clips: [], seconds: 0};
    const clipRows = new Map();
    const clipAudio = new Audio();
    let clipPlaying = "";
    const STOP_ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><rect x="6" y="6" width="12" height="12"/></svg>';

    function formatLength(seconds) {
      if (seconds === null || seconds === undefined) return "unreadable";
      if (seconds < 60) return `${Number(seconds).toFixed(1)} s`;
      const whole = Math.round(seconds);
      return `${Math.floor(whole / 60)} min ${whole % 60} s`;
    }

    function clipProblem(seconds) {
      if (seconds === null || seconds === undefined) return "The file cannot be read.";
      if (seconds <= 0.6) return "Training skips clips of 0.6 s or less.";
      if (seconds >= 54) return "Training skips clips of 54 s or more.";
      return "";
    }

    async function loadClips(force = false) {
      const dataset = ((state && state.datasets) || []).find(row => row.id === $("trainingDataset").value);
      const signature = dataset ? `${dataset.id}|${dataset.wav_count}|${dataset.list_rows}` : "";
      if (!force && signature === clipsSignature) return;
      clipsSignature = signature;
      if (!dataset) {
        clipsData = {clips: [], seconds: 0};
        renderClips("");
        return;
      }
      // Another dataset's clips, and the references made from them, go at once, not when the new list arrives.
      if (dataset.id !== clipsDataset) {
        clipsData = {clips: [], seconds: 0};
        renderClips(dataset.id);
      }
      try {
        const data = await (await fetch(`/api/dataset-clips?dataset=${encodeURIComponent(dataset.id)}`)).json();
        if (data.ok && $("trainingDataset").value === dataset.id) {
          clipsData = data;
          renderClips(dataset.id);
        }
      } catch (_) {
        clipsSignature = "";
      }
    }

    // The saved epochs of the model named in step 2, each used to say the same test line.
    function savedEpochs(kind) {
      const name = modelSlug($("trainingModelName").value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const pattern = new RegExp(kind === "gpt" ? `^${name}-e(\\d+)\\.ckpt$` : `^${name}_e(\\d+)_s\\d+\\.pth$`);
      return ((state && state.models && state.models[kind]) || []).filter(row => pattern.test(row.name)).length;
    }

    let compareRowsSignature = "";
    let compareChosen = "";
    function renderCompare() {
      const kind = $("compareKind").value;
      const count = savedEpochs(kind);
      const job = state && state.jobs && state.jobs.compare;
      const running = Boolean(job && job.status === "running");
      // Models are listed once WSL runs; until then the comparison itself finds the epochs.
      const listed = Boolean(state && state.models_listed);
      $("compareStop").hidden = !running;
      $("compareRun").disabled = running || (listed && !count) || !$("compareReference").value;
      $("compareInfo").textContent = !$("compareReference").options.length
        ? "Compare needs a clip of 3 to 10 seconds in this dataset, as the reference."
        : !listed
        ? "Generates one line with each saved epoch, so you can choose the best by ear."
        : (count
          ? `Generates one line with each of the ${count} saved ${kind === "gpt" ? "GPT" : "SoVITS"} epochs of ${modelSlug($("trainingModelName").value)}, so you can choose the best by ear.`
          : `${modelSlug($("trainingModelName").value)} has no saved ${kind === "gpt" ? "GPT" : "SoVITS"} epochs yet. Train it first.`);
      if (!job) return;
      const last = (job.lines || []).slice(-1)[0] || "Starting...";
      $("compareStatus").textContent = running ? last : (job.status === "failed" ? `${last} See Logs for details.` : (job.status === "stopped" ? "Stopped." : (compareChosen || "Done.")));
      $("compareStatus").className = job.status === "failed" ? "status-line bad" : (!running && compareChosen ? "status-line good" : "status-line");
      const results = job.results || [];
      const signature = results.map(row => row.url).join("|");
      if (signature === compareRowsSignature) return;
      compareRowsSignature = signature;
      const list = $("compareResults");
      list.innerHTML = "";
      results.forEach(row => {
        const line = document.createElement("div");
        line.className = "compare-row";
        const label = document.createElement("span");
        label.textContent = `${row.kind === "gpt" ? "GPT" : "SoVITS"} epoch ${row.epoch}`;
        const audio = document.createElement("audio");
        audio.src = row.url;
        audio.dataset.label = label.textContent;
        const use = document.createElement("button");
        use.className = "secondary";
        use.textContent = "Use this";
        use.onclick = () => {
          // Text to Voice and Mic to Voice speak with this epoch from now on.
          [["gptModel", row.gpt], ["sovitsModel", row.sovits], ["relayGptModel", row.gpt], ["relaySovitsModel", row.sovits]].forEach(([id, path]) => {
            if (Array.from($(id).options).some(option => option.value === path)) $(id).value = path;
          });
          saveRelaySettings();
          compareChosen = `Text to Voice and Mic to Voice now use ${label.textContent}.`;
          $("compareStatus").className = "status-line good";
          $("compareStatus").textContent = compareChosen;
        };
        line.append(label, audio, use);
        list.appendChild(line);
        attachPlayer(audio);
      });
    }

    function fillCompareReferences() {
      // Clips that can be a reference: 3 to 10 seconds long.
      const select = $("compareReference");
      const old = select.value;
      select.innerHTML = "";
      (clipsData.clips || []).filter(clip => clip.duration >= 3 && clip.duration <= 10).forEach(clip => option(select, clip.wav, clip.text));
      if (Array.from(select.options).some(item => item.value === old)) select.value = old;
      renderCompare();
    }

    // One player for the clip list and the files waiting to be added. A row's button plays its
    // audio; pressing it again, or another row's, stops it.
    let clipButton = null;
    let clipObjectUrl = "";
    function stopClip() {
      clipAudio.pause();
      clipAudio.removeAttribute("src");
      clipAudio.load();
      if (clipButton) clipButton.innerHTML = PLAY_ICON;
      if (clipObjectUrl) URL.revokeObjectURL(clipObjectUrl);
      clipButton = null;
      clipObjectUrl = "";
      clipPlaying = "";
    }
    clipAudio.onended = stopClip;

    function playRowAudio(key, button, source) {
      const playing = clipPlaying === key;
      stopClip();
      if (playing) return;
      clipPlaying = key;
      clipButton = button;
      button.innerHTML = STOP_ICON;
      // A file not yet added plays from the computer.
      if (source instanceof Blob) clipObjectUrl = URL.createObjectURL(source);
      clipAudio.src = clipObjectUrl || source;
      clipAudio.volume = playerVolume;
      clipAudio.play().catch(() => { if (clipPlaying === key) stopClip(); });
    }

    function rowPlayButton(label) {
      const play = document.createElement("button");
      play.type = "button";
      play.className = "secondary icon-button";
      play.innerHTML = PLAY_ICON;
      play.setAttribute("aria-label", `Play ${label}`);
      return play;
    }

    function clipRow(clip) {
      const row = document.createElement("div");
      row.className = "clip-row";
      const play = rowPlayButton(clip.wav);
      play.onclick = () => playRowAudio(clip.wav, play, clip.url);
      const text = document.createElement("input");
      text.value = clip.text;
      text.setAttribute("aria-label", `Transcript of ${clip.wav}`);
      text.title = clip.wav;
      let saved = clip.text;
      text.onchange = async () => {
        try {
          saved = (await post("/api/dataset-clip-text", {dataset_id: clipsDataset, wav: clip.wav, text: text.value})).text;
          text.value = saved;
        } catch (error) {
          text.value = saved;
          showError(error.message);
        }
      };
      const length = document.createElement("span");
      length.className = "clip-length";
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "secondary icon-button";
      remove.title = "Delete this clip";
      remove.setAttribute("aria-label", `Delete ${clip.wav}`);
      remove.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
      remove.onclick = async () => {
        if (clipPlaying === clip.wav) stopClip();
        const next = row.nextElementSibling || row.previousElementSibling;
        try {
          await post("/api/dataset-clip-delete", {dataset_id: clipsDataset, wav: clip.wav});
          await refresh();
          await loadClips(true);
          if (next && next.isConnected) next.querySelector("button:last-child").focus();
        } catch (error) {
          showError(error.message);
        }
      };
      row.append(play, text, length, remove);
      return {row, play, length};
    }

    function renderClips(datasetId) {
      if (datasetId !== clipsDataset) {
        stopClip();
        clipRows.clear();
        $("clipList").innerHTML = "";
        clipsDataset = datasetId;
      }
      const clips = clipsData.clips || [];
      const names = new Set(clips.map(clip => clip.wav));
      clipRows.forEach((parts, wav) => {
        if (names.has(wav)) return;
        if (clipPlaying === wav) stopClip();
        parts.row.remove();
        clipRows.delete(wav);
      });
      let problems = 0;
      clips.forEach(clip => {
        if (!clipRows.has(clip.wav)) {
          const parts = clipRow(clip);
          clipRows.set(clip.wav, parts);
          $("clipList").appendChild(parts.row);
        }
        const parts = clipRows.get(clip.wav);
        const problem = clipProblem(clip.duration);
        if (problem) problems++;
        parts.length.textContent = formatLength(clip.duration);
        parts.length.title = problem;
        parts.row.classList.toggle("warn", Boolean(problem));
      });
      const summary = $("clipSummary");
      summary.hidden = !clips.length;
      summary.textContent = `${clips.length} clip${clips.length === 1 ? "" : "s"}, ${formatLength(clipsData.seconds)} in total.`
        + (problems ? ` Training skips ${problems} of them: see the lengths marked in yellow.` : "")
        + (clips.length && clipsData.seconds < 60 ? " A minute or more of clips trains a better voice." : "");
      summary.classList.toggle("warn", problems > 0);
      fillCompareReferences();
    }

    // Redrawn only when a dataset or the choice changes, so keyboard focus stays in the list.
    let datasetListSignature = null;
    function renderDatasetList(datasets, selectedId) {
      const signature = selectedId + "|" + datasets.map(row => `${row.id}:${row.name}:${row.wav_count}:${row.ready}`).join("|");
      if (signature === datasetListSignature) return;
      datasetListSignature = signature;
      const list = $("datasetList");
      list.innerHTML = "";
      datasets.forEach(dataset => {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "history-item";
        row.setAttribute("role", "option");
        row.setAttribute("aria-selected", dataset.id === selectedId ? "true" : "false");
        const name = document.createElement("span");
        name.className = "history-text";
        name.textContent = dataset.name;
        const meta = document.createElement("span");
        meta.className = "history-meta";
        const clips = document.createElement("span");
        clips.textContent = `${dataset.wav_count} ${dataset.wav_count === 1 ? "clip" : "clips"}`;
        const ready = document.createElement("span");
        ready.textContent = dataset.ready ? "ready" : (dataset.wav_count ? "incomplete" : "no clips yet");
        meta.append(clips, ready);
        row.append(name, meta);
        row.onclick = () => {
          $("trainingDataset").value = dataset.id;
          $("trainingDataset").onchange();
        };
        list.appendChild(row);
      });
      $("datasetListCount").textContent = datasets.length ? `${datasets.length} ${datasets.length === 1 ? "dataset" : "datasets"}` : "No datasets yet";
      $("noDatasets").hidden = datasets.length > 0;
      $("trainingSteps").hidden = !datasets.length;
    }

    function openCreateDataset() {
      $("createDatasetDialog").showModal();
      $("newDatasetName").focus();
    }
    $("newDatasetButton").onclick = openCreateDataset;
    $("newDatasetEmpty").onclick = openCreateDataset;

    function renderDatasetUploads() {
      const list = $("datasetUploadRows");
      list.innerHTML = "";
      datasetUploadFiles.forEach((file, index) => {
        const row = document.createElement("div");
        row.className = "upload-row";
        const play = rowPlayButton(file.name);
        play.onclick = () => playRowAudio(file, play, file);
        // A file that plays while the list is redrawn keeps its stop button.
        if (clipPlaying === file) {
          play.innerHTML = STOP_ICON;
          clipButton = play;
        }
        const name = document.createElement("div");
        name.className = "file-name";
        name.title = file.name;
        name.textContent = file.name;
        const transcript = document.createElement("input");
        transcript.value = datasetUploadTexts.get(file) || "";
        transcript.placeholder = datasetUploadTexts.has(file) ? "Exact spoken transcript" : "Writing the transcript...";
        transcript.setAttribute("aria-label", `Transcript of ${file.name}`);
        transcript.oninput = () => {
          datasetUploadTexts.set(file, transcript.value);
          updateUploadButton();
        };
        datasetUploadInputs.set(file, transcript);
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "secondary icon-button";
        remove.title = "Remove from the list";
        remove.setAttribute("aria-label", `Remove ${file.name}`);
        remove.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
        remove.onclick = () => {
          if (clipPlaying === file) stopClip();
          datasetUploadFiles = datasetUploadFiles.filter(other => other !== file);
          renderDatasetUploads();
        };
        row.append(play, name, transcript, remove);
        list.appendChild(row);
      });
      $("datasetUploadHint").hidden = !datasetUploadFiles.length;
      updateUploadButton();
    }

    let uploadBusy = false;
    function updateUploadButton() {
      const selected = state && (state.datasets || []).find(dataset => dataset.id === $("trainingDataset").value);
      const waiting = datasetUploadFiles.some(file => !datasetUploadTexts.has(file));
      $("uploadDatasetAudio").hidden = !datasetUploadFiles.length;
      $("uploadDatasetAudio").disabled = uploadBusy || !datasetUploadFiles.length || !selected || waiting;
    }

    function setUploadTranscript(file, text) {
      datasetUploadTexts.set(file, text);
      const input = datasetUploadInputs.get(file);
      if (input) {
        input.value = text;
        input.placeholder = "Exact spoken transcript";
      }
      updateUploadButton();
    }

    // Speech recognition writes the transcripts, one file at a time, in the dataset's language.
    function queueTranscripts(files) {
      transcriptQueue.push(...files.filter(file => !datasetUploadTexts.has(file)));
      runTranscripts();
    }

    async function runTranscripts() {
      if (transcriptBusy) return;
      transcriptBusy = true;
      let failure = "";
      try {
        while (transcriptQueue.length) {
          const file = transcriptQueue.shift();
          if (!datasetUploadFiles.includes(file) || datasetUploadTexts.has(file)) continue;
          if (failure) {
            setUploadTranscript(file, suggestedTranscript(file.name));
            continue;
          }
          const left = 1 + transcriptQueue.filter(other => datasetUploadFiles.includes(other) && !datasetUploadTexts.has(other)).length;
          $("datasetUploadProgress").className = "status-line";
          $("datasetUploadProgress").textContent = `Writing transcripts, ${left} left...`;
          // The first transcript also loads, and the first time downloads, the speech model.
          const slow = setTimeout(() => {
            $("datasetUploadProgress").textContent = `Writing transcripts, ${left} left. The first time also downloads the speech model (about 480 MB)...`;
          }, 8000);
          try {
            const result = await post("/api/dataset-transcribe", {dataset_id: $("trainingDataset").value, filename: file.name, audio_base64: await fileToBase64(file)});
            if (!datasetUploadTexts.has(file)) setUploadTranscript(file, result.text);
          } catch (error) {
            failure = error.message;
            if (!datasetUploadTexts.has(file)) setUploadTranscript(file, suggestedTranscript(file.name));
          } finally {
            clearTimeout(slow);
          }
        }
        $("datasetUploadProgress").className = failure ? "status-line bad" : "status-line";
        $("datasetUploadProgress").textContent = failure
          ? `Automatic transcripts failed, so the file names were used. ${failure}`
          : (datasetUploadFiles.length ? "Transcripts written." : "");
      } finally {
        transcriptBusy = false;
      }
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
    function activateTab(name, autoStart = true) {
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
      try { localStorage.setItem("voiceDashboardTab", name); } catch (_) {}
      autoStartPending = name === "relay" && autoStart;
      autoStartSystem();
      renderStartButton();
      // Before the first status arrives, the page is still loading and has nothing to show.
      if (name === "relay" && lastSystemSnapshot) renderRelayCenter();
    }
    document.querySelectorAll(".tab-button, [data-go]").forEach(button => {
      button.onclick = () => {
        activateTab(button.dataset.tab || button.dataset.go);
        refresh();
      };
    });
    // The page opens on the tab used last, and the first time on Mic to Voice. Without a voice,
    // that page points to Voice Training.
    let savedTab = "";
    try { savedTab = localStorage.getItem("voiceDashboardTab") || ""; } catch (_) {}
    activateTab(location.hash.slice(1) || savedTab || "relay");
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

    // The Mic to Voice models. Their lists fill only once WSL runs, which can be after Mic to Voice
    // starts, so the saved choice stands in until then.
    function relayModels() {
      return {
        gpt: $("relayGptModel").value || relaySavedSettings.relayGptModel || "",
        sovits: $("relaySovitsModel").value || relaySavedSettings.relaySovitsModel || "",
      };
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
      // Only while the voice is starting or running; when it is off, why it is off matters more.
      if (!hint && needsPageClick() && (systemStartBusy || ["starting", "ready", "degraded"].includes(system.status))) {
        hint = "Click anywhere on this page once. Browsers play the voice only after that.";
      }
      if (!hint && relayCurrentItem && item && item !== relayCurrentItem) hint = "Another line is still being generated or played.";
      if (!hint) {
        if (relayListening && !relayAudioCtx) hint = "Speech recognition is starting.";
        else if (system.status === "stopped" && (systemStartBusy || autoStartPending)) hint = "Starting speech recognition and the voice engine.";
        else if (system.status === "stopped" && !relayListening && state && state.gpu_busy) hint = "Training is using the GPU, so nothing started by itself. Press Start to load the voice anyway.";
        else if (system.status === "stopped" && !relayListening) hint = "Everything is stopped. Press Start to turn it back on.";
        else if (system.status === "starting") hint = system.message || "Starting the system.";
        else if (system.status === "failed" || system.status === "degraded") hint = system.message || "";
      }
      return {key: "ready", icon: "", label: "Hold to talk", hint};
    }

    // Mic to Voice can start without a click now, but browsers refuse to play sound on a page that
    // was never clicked.
    function needsPageClick() {
      return Boolean(navigator.userActivation) && !navigator.userActivation.hasBeenActive;
    }
    ["pointerdown", "keydown"].forEach(name => document.addEventListener(name, () => setTimeout(renderRelayCenter), {capture: true, once: true}));

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
      await post("/api/start-api", relayModels());
      for (let i = 0; i < 80; i++) {
        await new Promise(resolve => setTimeout(resolve, 1500));
        await refresh();
        if (state && state.api_ready) return;
      }
      throw new Error("TTS engine did not become ready.");
    }

    async function generateRelayLine(text, generationId) {
      const ref = relayRef();
      if (!ref) throw new Error("Pick a reference clip in Settings.");
      const aux = relayAuxRefs().map(item => item.wsl_path);
      return await post("/api/generate", {
        ...relayModels(),
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
      if (status !== lastSystemStatus) {
        lastSystemStatus = status;
        renderRelayCenter();
      }
      autoStartSystem();
      renderStartButton();
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

    function hasVoice() {
      return Boolean(state && (state.datasets || []).some(dataset => dataset.wav_count > 0));
    }

    // Start shows only on Mic to Voice: as Retry after a failed start, and as Start after Stop Everything.
    function renderStartButton() {
      const status = (lastSystemSnapshot || {}).status || "";
      const failed = status === "failed" || status === "degraded";
      $("startSystem").hidden = document.body.dataset.tab !== "relay" || systemStartBusy || autoStartPending
        || !hasVoice() || !(failed || status === "stopped");
      $("startSystem").textContent = failed ? "Retry" : "Start";
    }

    function autoStartSystem() {
      if (!autoStartPending || systemStartBusy || document.body.dataset.tab !== "relay" || !lastSystemSnapshot || !hasVoice()) return;
      autoStartPending = false;
      // Training keeps the GPU. Start still loads the voice on purpose, for example to test a checkpoint.
      if (lastSystemSnapshot.status === "stopped" && !state.gpu_busy) startSystem(true);
      // The hint said it was starting while this was undecided.
      else renderRelayCenter();
    }

    async function startSystem(auto = false) {
      const token = ++startToken;
      systemStartBusy = true;
      renderStartButton();
      renderRelayCenter();
      $("globalError").classList.remove("visible");
      armRelayWhenReady = true;
      recentActivity("Requesting microphone access...");
      try {
        saveRelaySettings();
        await refreshRelayDevices();
        if (token !== startToken) return;
        const result = await post("/api/start-system", {...relayModels(), binding: $("relayPttKey").value, auto});
        if (result.started === false) {
          // A GPU job started meanwhile; the page learns it from the next state.
          armRelayWhenReady = false;
          await refresh();
          return;
        }
        recentActivity("System startup requested.");
        await refreshSystem();
      } catch (error) {
        armRelayWhenReady = false;
        showError(error.message);
      } finally {
        systemStartBusy = false;
        renderStartButton();
        renderRelayCenter();
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

    // A refresh asked for while another runs waits for it and runs once more, as the one running
    // asked with the choices from before.
    let refreshRunning = null;
    let refreshQueued = null;
    function refresh() {
      if (refreshRunning) {
        // Also after a refresh that failed, as when the dashboard restarts.
        if (!refreshQueued) refreshQueued = refreshRunning.catch(() => {}).then(() => { refreshQueued = null; return refresh(); });
        return refreshQueued;
      }
      refreshRunning = refreshOnce().finally(() => { refreshRunning = null; });
      return refreshRunning;
    }

    async function refreshOnce() {
      const res = await fetch(`/api/state?dataset=${encodeURIComponent(voiceDataset)}`);
      state = await res.json();
      if (state.config_error && !configErrorShown) {
        configErrorShown = true;
        showError(state.config_error);
      }
      fillTrainingDatasets(state.datasets || []);
      fillDatasetSelects(state.datasets || [], state.reference_dataset || "");
      document.body.dataset.voice = hasVoice() ? "yes" : "no";
      if (Boolean(state.gpu_busy) !== lastGpuBusy) {
        lastGpuBusy = Boolean(state.gpu_busy);
        renderRelayCenter();
      }
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
      const jobLine = (key, element, bar) => {
        const job = state.jobs && state.jobs[key];
        const progress = job && job.progress;
        $(bar).hidden = !(job && job.status === "running" && progress);
        if (!job) { $(element).textContent = ""; return; }
        let text = job.status === "running" ? "Running..." : `${job.status}${job.returncode !== null && job.returncode !== undefined ? ` (exit ${job.returncode})` : ""}`;
        if (job.status === "running" && progress) {
          text = progress.epoch ? `Epoch ${progress.epoch} of ${progress.epochs} done` : "Starting...";
          if (progress.seconds_left) text += `, about ${Math.max(1, Math.round(progress.seconds_left / 60))} min left`;
          $(bar).firstElementChild.style.width = `${Math.min(100, 100 * progress.epoch / progress.epochs)}%`;
        }
        $(element).textContent = text;
        $(element).className = job.status === "failed" ? "status-line bad" : (job.status === "done" ? "status-line good" : "status-line");
      };
      jobLine("train-sovits", "trainSovitsStatus", "trainSovitsBar");
      jobLine("train-gpt", "trainGptStatus", "trainGptBar");
      renderCompare();
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
      renderRestore();
      renderCutStatus();
      showWelcome();
      // Loaded data can move the highlighted element.
      placeTour();
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

    // Guided tour. The page darkens except for one highlighted element, and a pop-up beside it
    // explains it. The tour moves through the tabs; opening Mic to Voice starts nothing meanwhile.
    // Steps marked "voice" show controls that exist only once a voice does.
    const TOUR_STEPS = [
      {title: "Welcome", text: "This tour shows what each part of the dashboard does and how to use it. Use Next and Back, or the arrow keys. Esc ends the tour."},
      {targets: [".tablist"], title: "Tabs", text: "Mic to Voice and Text to Voice speak in a voice. Voice Training creates voices. Logs and System Check help when something does not work."},
      {tab: "training", targets: ["#newDatasetButton"], title: "Create a voice", text: "Every voice starts as a dataset. Press + New dataset, then give it a name and a language."},
      {tab: "training", targets: ["#clipsStep", "#noDatasets"], title: "Add clips", text: "Add clips, or long recordings, which are cut into clips at pauses. Speech recognition writes each transcript. Check them in the list, where every clip can be played, corrected or deleted."},
      {tab: "training", targets: ["#prepareStep", "#noDatasets"], title: "Sync the clips", text: "Sync Only copies the clips to WSL, which is all Text to Voice and Mic to Voice need. Sync again after adding clips."},
      {tab: "training", targets: ["#trainPanel", "#noDatasets"], title: "Train the voice", text: "Training makes the voice closer to its clips, after Sync & Prepare. It takes a while, and it stops the voice first to free the GPU."},
      {tab: "training", targets: ["#compareStep", "#noDatasets"], title: "Compare epochs", text: "Training saves the voice every few epochs. Compare says one test line with each saved epoch, so you can keep the one that sounds best."},
      {tab: "training", targets: ["#trainingModeRestore"], title: "Restore audio", text: "Clean up recordings before training: remove music, echo or noise, and restore missing high frequencies. Each result is added to the list, to restore again or add to a dataset."},
      {tab: "generate", targets: ["#targetText", "#tab-generate .no-voice-note"], title: "Text to Voice", text: "Type what the voice should say.", noVoiceText: "Here you type what a voice should say. It needs a voice first, made on Voice Training."},
      {tab: "generate", voice: true, targets: [".voice-line"], title: "Choose the voice", text: "Pick the voice and a reference clip. The clip's tone shapes how the line sounds, and you can play it here."},
      {tab: "generate", voice: true, targets: [".compose-actions"], title: "Generate", text: "Generate makes the line. The voice engine starts by itself the first time, which takes a moment. The gear opens the models, the text language and finer controls."},
      {tab: "generate", targets: ["#tab-generate .history"], title: "Your lines", text: "Every line you generate is listed here. Click one to play it again."},
      {tab: "relay", targets: ["#relayPttButton", "#relayCapture .no-voice-note"], title: "Mic to Voice", text: "Hold this button or your push-to-talk key, speak English, and release. Your words are spoken in the voice.", noVoiceText: "Here you speak, and your words are spoken in a voice. It needs a voice first, made on Voice Training."},
      {tab: "relay", voice: true, targets: ["#relaySetPttKey"], title: "Push-to-talk key", text: "Click to choose another key. It also works while another app, such as a game, is in front."},
      {tab: "relay", voice: true, targets: [".relay-options"], title: "Options", text: "With Hold to talk off, the microphone listens all the time instead. With Speak automatically off, your words are only written down. The gear sets the output device, the volume and the voice."},
      {tab: "relay", targets: ["#relayHistory"], title: "Recordings", text: "Everything you said in this session is listed here, to read or play again."},
      {targets: ["#systemIsland"], title: "Status bar", text: "Shows whether speech recognition, the voice engine and the global key are running. Mic to Voice starts them when you open it, and Stop Everything stops them and frees the GPU."},
      {targets: ["#openGuide"], title: "Take the tour again", text: "Press Guide to see this tour again at any time."},
    ];
    let tour = null;

    function startTour() {
      document.querySelectorAll("dialog[open]").forEach(dialog => dialog.close());
      endRelayPush();
      tour = {
        steps: TOUR_STEPS.filter(step => !step.voice || hasVoice()),
        index: 0,
        returnTab: document.body.dataset.tab,
        resumeAutoStart: autoStartPending,
        returnFocus: document.activeElement,
      };
      autoStartPending = false;
      $("tour").hidden = false;
      showTourStep();
    }

    function endTour() {
      if (!tour) return;
      const {returnTab, resumeAutoStart, returnFocus} = tour;
      tour = null;
      $("tour").hidden = true;
      activateTab(returnTab, resumeAutoStart);
      refresh();
      if (returnFocus && returnFocus.focus) returnFocus.focus();
    }

    function moveTour(change) {
      if (!tour) return;
      const index = tour.index + change;
      if (index < 0) return;
      if (index >= tour.steps.length) return endTour();
      tour.index = index;
      showTourStep();
    }

    function tourTarget(step) {
      for (const selector of step.targets || []) {
        const element = document.querySelector(selector);
        if (element && element.getClientRects().length) return element;
      }
      return null;
    }

    function showTourStep() {
      const step = tour.steps[tour.index];
      if (step.tab && document.body.dataset.tab !== step.tab) activateTab(step.tab, false);
      if (step.tab === "training") setTrainingMode("train");
      const target = tourTarget(step);
      if (target) target.scrollIntoView({block: "center", inline: "nearest"});
      $("tourCount").textContent = `${tour.index + 1} of ${tour.steps.length}`;
      $("tourTitle").textContent = step.title;
      $("tourText").textContent = !hasVoice() && step.noVoiceText ? step.noVoiceText : step.text;
      $("tourBack").disabled = tour.index === 0;
      $("tourNext").textContent = tour.index === tour.steps.length - 1 ? "Finish" : "Next";
      placeTour();
      $("tourNext").focus();
    }

    // The pop-up goes below the element, else above, right or left, else at the bottom of the window.
    function placeTour() {
      if (!tour) return;
      const target = tourTarget(tour.steps[tour.index]);
      const spot = $("tourSpot");
      const pop = $("tourPop");
      const gap = 12;
      const pad = 6;
      const width = window.innerWidth;
      const height = window.innerHeight;
      const popWidth = pop.offsetWidth;
      const popHeight = pop.offsetHeight;
      $("tour").classList.toggle("no-spot", !target);
      spot.hidden = !target;
      let top = (height - popHeight) / 2;
      let left = (width - popWidth) / 2;
      if (target) {
        const rect = target.getBoundingClientRect();
        const box = {top: rect.top - pad, left: rect.left - pad, right: rect.right + pad, bottom: rect.bottom + pad};
        Object.assign(spot.style, {top: `${box.top}px`, left: `${box.left}px`, width: `${box.right - box.left}px`, height: `${box.bottom - box.top}px`});
        // The highlight follows the element's corners, so a round button gets a round highlight.
        const radius = getComputedStyle(target).borderTopLeftRadius;
        spot.style.borderRadius = radius.endsWith("%") ? radius : `${parseFloat(radius) + pad}px`;
        const middleX = rect.left + rect.width / 2 - popWidth / 2;
        const middleY = rect.top + rect.height / 2 - popHeight / 2;
        if (box.bottom + gap + popHeight <= height - gap) [top, left] = [box.bottom + gap, middleX];
        else if (box.top - gap - popHeight >= gap) [top, left] = [box.top - gap - popHeight, middleX];
        else if (box.right + gap + popWidth <= width - gap) [top, left] = [middleY, box.right + gap];
        else if (box.left - gap - popWidth >= gap) [top, left] = [middleY, box.left - gap - popWidth];
        else [top, left] = [height - popHeight - gap, middleX];
      }
      pop.style.top = `${Math.round(Math.max(gap, Math.min(top, height - popHeight - gap)))}px`;
      pop.style.left = `${Math.round(Math.max(gap, Math.min(left, width - popWidth - gap)))}px`;
    }

    $("tourNext").onclick = () => moveTour(1);
    $("tourBack").onclick = () => moveTour(-1);
    $("tourEnd").onclick = endTour;
    // While the tour runs, keys belong to it, so the push-to-talk key does not start a recording.
    document.addEventListener("keydown", event => {
      if (!tour) return;
      event.stopPropagation();
      if (event.key === "Escape") { event.preventDefault(); endTour(); }
      else if (event.key === "ArrowRight") { event.preventDefault(); moveTour(1); }
      else if (event.key === "ArrowLeft") { event.preventDefault(); moveTour(-1); }
      else if (event.key === "Tab") {
        // Focus stays in the pop-up.
        event.preventDefault();
        const buttons = [...$("tourPop").querySelectorAll("button:not(:disabled)")];
        const index = buttons.indexOf(document.activeElement);
        buttons[(index + (event.shiftKey ? buttons.length - 1 : 1)) % buttons.length].focus();
      }
    }, true);
    ["mousedown", "pointerdown", "auxclick", "contextmenu"].forEach(name => $("tour").addEventListener(name, event => event.stopPropagation()));
    window.addEventListener("resize", placeTour);
    window.addEventListener("scroll", placeTour, true);

    // Offered once, to a dashboard without datasets. Closing it in any way counts as seen.
    function showWelcome() {
      if (welcomeChecked || !state) return;
      welcomeChecked = true;
      let seen = false;
      try { seen = localStorage.getItem("voiceDashboardWelcomeSeen") === "1"; } catch (_) {}
      if (!seen && !(state.datasets || []).length) $("welcomeDialog").showModal();
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
      const list = $("generateHistory");
      list.innerHTML = "";
      const items = generateOutputs.slice().reverse();
      if (!items.length) {
        const empty = document.createElement("div");
        empty.className = "history-empty";
        empty.textContent = "Lines you generate appear here.";
        list.appendChild(empty);
      }
      items.forEach(item => {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "history-item";
        row.setAttribute("role", "option");
        row.setAttribute("aria-selected", item === generateShown ? "true" : "false");
        row.title = item.name;
        const text = document.createElement("span");
        text.className = "history-text";
        text.textContent = item.text;
        const meta = document.createElement("span");
        meta.className = "history-meta";
        const time = document.createElement("span");
        time.textContent = formatClock(item.createdAt);
        meta.append(time);
        row.append(text, meta);
        row.onclick = () => showGeneratedLine(item, true);
        list.appendChild(row);
      });
      $("generateHistoryCount").textContent = items.length ? `${items.length} this session` : "Nothing generated yet";
    }

    function showGeneratedLine(item, play) {
      generateShown = item;
      $("outputAudio").src = item.url + "?t=" + Date.now();
      $("outputName").textContent = item.name;
      $("generateResult").hidden = false;
      if (play) $("outputAudio").play().catch(() => {});
      renderGenerateHistory();
    }

    $("generateNewLine").onclick = () => {
      generateShown = null;
      $("targetText").value = "";
      $("generateResult").hidden = true;
      $("generateStatus").textContent = "";
      renderGenerateHistory();
      $("targetText").focus();
    };
    $("openGenerateSettings").onclick = () => $("generateSettingsDialog").showModal();
    renderGenerateHistory();
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
    $("openGuide").onclick = startTour;
    $("welcomeDialog").addEventListener("close", () => {
      try { localStorage.setItem("voiceDashboardWelcomeSeen", "1"); } catch (_) {}
    });
    $("welcomeNo").onclick = () => $("welcomeDialog").close();
    $("welcomeYes").onclick = () => {
      $("welcomeDialog").close();
      startTour();
    };
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

    $("startSystem").onclick = () => startSystem(false);

    $("stopSystem").onclick = async () => {
      $("stopSystem").disabled = true;
      armRelayWhenReady = false;
      autoStartPending = false;
      startToken += 1;
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
        if (!ptt.held) endRelayPush();
        else if (!tour) beginRelayPush();
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
      loadClips(true);
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
        newTrainingDataset = {id: result.dataset.id, name: result.dataset.name};
        fillTrainingDatasets(state ? state.datasets : []);
        datasetUploadFiles = [];
        renderDatasetUploads();
        $("newDatasetName").value = "";
        $("createDatasetDialog").close();
        recentActivity(`Dataset ${result.dataset.name} created.`);
        await refresh();
      } catch (error) {
        showError(error.message);
      } finally {
        $("createDataset").disabled = false;
      }
    };

    // Recordings longer than this are cut into clips at pauses on the server; shorter ones are clips.
    const CUT_LONGER_THAN = 20;

    function audioSeconds(file) {
      // The length of an audio file, or 0 when the browser cannot read it.
      return new Promise(resolve => {
        const url = URL.createObjectURL(file);
        const audio = new Audio();
        const timer = setTimeout(() => done(0), 10000);
        function done(seconds) {
          clearTimeout(timer);
          audio.removeAttribute("src");
          URL.revokeObjectURL(url);
          resolve(Number.isFinite(seconds) ? seconds : 0);
        }
        audio.preload = "metadata";
        audio.onloadedmetadata = () => done(audio.duration);
        audio.onerror = () => done(0);
        audio.src = url;
      });
    }

    let cutSending = "";
    const UPLOAD_LIMIT = 4 * 1024 ** 3;
    async function sendToCut(file, datasetId) {
      if (file.size > UPLOAD_LIMIT) throw new Error("Files over 4 GB cannot be sent. Cut the recording into shorter files first.");
      const query = `dataset=${encodeURIComponent(datasetId)}&filename=${encodeURIComponent(file.name)}`;
      const response = await fetch(`/api/dataset-cut?${query}`, {method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: file});
      const result = await response.json().catch(() => ({}));
      if (!response.ok || result.ok === false) throw new Error(result.error || "The upload failed.");
    }

    // A voice as one file: both models and the reference clips.
    const importVoiceFile = document.createElement("input");
    importVoiceFile.type = "file";
    importVoiceFile.accept = ".zip";
    importVoiceFile.hidden = true;
    document.body.appendChild(importVoiceFile);
    function shareStatus(text, kind = "") {
      $("shareStatus").className = `status-line ${kind}`;
      $("shareStatus").textContent = text;
    }
    document.querySelectorAll("[data-import-voice]").forEach(button => { button.onclick = () => importVoiceFile.click(); });
    importVoiceFile.onchange = async () => {
      const file = importVoiceFile.files[0];
      importVoiceFile.value = "";
      if (!file) return;
      if (file.size > UPLOAD_LIMIT) return showError("This file is over 4 GB, far larger than a voice file.");
      const buttons = Array.from(document.querySelectorAll("[data-import-voice]"));
      buttons.forEach(button => { button.disabled = true; button.textContent = "Importing..."; });
      shareStatus(`Importing ${file.name}...`);
      recentActivity(`Importing the voice ${file.name}...`);
      try {
        const response = await fetch("/api/voice-import", {method: "POST", headers: {"Content-Type": "application/zip"}, body: file});
        const result = await response.json().catch(() => ({}));
        if (!response.ok || result.ok === false) throw new Error(result.error || "The import failed.");
        // The imported voice becomes the voice of Text to Voice and Mic to Voice.
        voiceDataset = result.dataset;
        try { localStorage.setItem("voiceDashboardVoiceDataset", voiceDataset); } catch (_) {}
        refsSignature = "";
        modelsSignature = "";
        await refresh();
        const pairs = [["gptModel", result.gpt], ["sovitsModel", result.sovits], ["relayGptModel", result.gpt], ["relaySovitsModel", result.sovits]];
        const found = pairs.every(([id, path]) => Array.from($(id).options).some(item => item.value === path));
        pairs.forEach(([id, path]) => {
          if (Array.from($(id).options).some(item => item.value === path)) $(id).value = path;
        });
        saveRelaySettings();
        shareStatus(found ? `Imported ${result.name}. Text to Voice and Mic to Voice now use it.` : `Imported ${result.name}. Choose its models in Settings.`, "good");
        recentActivity(`Imported the voice ${result.name}.`);
      } catch (error) {
        shareStatus("");
        showError(error.message);
      } finally {
        buttons.forEach(button => { button.disabled = false; button.textContent = "Import a voice"; });
      }
    };
    $("exportVoice").onclick = async () => {
      $("exportVoice").disabled = true;
      shareStatus("Packing the voice...");
      try {
        const result = await post("/api/voice-export", {gpt: $("gptModel").value, sovits: $("sovitsModel").value, dataset_id: voiceDataset,
          reference: $("reference").value, prompt_text: $("promptText").value, extra_references: Array.from(auxSelected)});
        const link = document.createElement("a");
        link.href = result.url;
        link.download = result.name;
        document.body.append(link);
        link.click();
        link.remove();
        shareStatus(`Saved as ${result.name}.`, "good");
      } catch (error) {
        shareStatus("");
        showError(error.message);
      } finally {
        $("exportVoice").disabled = false;
      }
    };

    $("compareKind").onchange = renderCompare;
    $("trainingModelName").addEventListener("input", renderCompare);
    $("compareRun").onclick = async () => {
      $("compareRun").disabled = true;
      try {
        await post("/api/compare-epochs", {model: $("trainingModelName").value, kind: $("compareKind").value, dataset_id: $("trainingDataset").value,
          reference: $("compareReference").value, text: $("compareText").value});
        compareRowsSignature = null;
        compareChosen = "";
        $("compareResults").innerHTML = "";
        $("compareStatus").className = "status-line";
        $("compareStatus").textContent = "Starting...";
        refresh();
      } catch (error) {
        showError(error.message);
        renderCompare();
      }
    };

    $("datasetAddFiles").onclick = () => $("datasetAudioFiles").click();
    $("datasetAudioFiles").onchange = async () => {
      // Clips join the list, which can also hold recordings from Restore audio; long recordings
      // are cut into clips on the server.
      const picked = Array.from($("datasetAudioFiles").files || []);
      $("datasetAudioFiles").value = "";
      const datasetId = $("trainingDataset").value;
      const clips = [];
      const long = [];
      // A length the browser cannot read is 0; ffmpeg on the server reads more formats.
      for (const file of picked) {
        const seconds = await audioSeconds(file);
        (seconds === 0 || seconds > CUT_LONGER_THAN ? long : clips).push(file);
      }
      if ($("trainingDataset").value !== datasetId) return;
      if (clips.length) {
        datasetUploadFiles.push(...clips);
        renderDatasetUploads();
        queueTranscripts(clips);
      }
      for (const file of long) {
        cutSending = file.name;
        $("datasetCutStatus").className = "status-line";
        $("datasetCutStatus").textContent = `Sending ${file.name}...`;
        try {
          await sendToCut(file, datasetId);
        } catch (error) {
          showError(`${file.name}: ${error.message}`);
        } finally {
          cutSending = "";
        }
      }
      if (long.length) refresh();
    };

    // Clips the server has cut join the list of files to check, with their transcripts, once each.
    const cutTaken = new Set();
    async function takeCutClips(job) {
      for (const clip of job.results || []) {
        if (cutTaken.has(clip.url) || job.dataset !== $("trainingDataset").value) continue;
        cutTaken.add(clip.url);
        try {
          const response = await fetch(clip.url);
          if (!response.ok) continue;
          const file = new File([await response.blob()], clip.name, {type: "audio/wav"});
          if (job.dataset !== $("trainingDataset").value) continue;
          datasetUploadTexts.set(file, clip.text);
          datasetUploadFiles.push(file);
          renderDatasetUploads();
        } catch (_) {
          cutTaken.delete(clip.url);
        }
      }
    }

    function renderCutStatus() {
      const job = state && state.jobs && state.jobs["dataset-cut"];
      if (job) takeCutClips(job);
      const mine = Boolean(job && job.dataset === $("trainingDataset").value);
      const running = Boolean(mine && job.status === "running");
      $("datasetCutStop").hidden = !running;
      // A recording on its way to the server says so until it has arrived.
      if (cutSending) return;
      if (!mine) {
        $("datasetCutStatus").textContent = "";
        return;
      }
      const last = (job.lines || []).slice(-1)[0] || "Starting...";
      $("datasetCutStatus").textContent = running ? last : (job.status === "stopped" ? "Cutting stopped." : (job.status === "failed" ? `${last} See Logs for details.` : last));
      $("datasetCutStatus").className = job.status === "failed" ? "status-line bad" : "status-line";
    }

    $("uploadDatasetAudio").onclick = async () => {
      if (uploadBusy || !datasetUploadFiles.length) return;
      if (datasetUploadFiles.some(file => !(datasetUploadTexts.get(file) || "").trim())) return showError("Every selected audio file needs a transcript.");
      const datasetId = $("trainingDataset").value;
      const total = datasetUploadFiles.length;
      uploadBusy = true;
      updateUploadButton();
      let added = 0;
      try {
        // The list as it is now: a file removed meanwhile is not sent, and a transcript edited
        // meanwhile is sent as it is. Switching datasets empties the list, which ends the upload.
        while (datasetUploadFiles.length && $("trainingDataset").value === datasetId) {
          const file = datasetUploadFiles[0];
          $("datasetUploadProgress").className = "status-line";
          $("datasetUploadProgress").textContent = `Adding ${added + 1} of ${total}: ${file.name}`;
          const audio = await fileToBase64(file);
          if (!datasetUploadFiles.includes(file)) continue;
          await post("/api/upload-dataset-audio", {
            dataset_id: datasetId,
            filename: file.name,
            audio_base64: audio,
            text: (datasetUploadTexts.get(file) || "").trim(),
            speaker: "",
            language: ""
          });
          added++;
          // An added file leaves the list, so a failure later does not add it twice. Only its row
          // goes, so typing in another row goes on.
          datasetUploadFiles = datasetUploadFiles.filter(other => other !== file);
          if (clipPlaying === file) stopClip();
          const input = datasetUploadInputs.get(file);
          if (input) input.parentElement.remove();
          $("datasetUploadHint").hidden = !datasetUploadFiles.length;
        }
        $("datasetUploadProgress").textContent = `${added} file${added === 1 ? "" : "s"} added.`;
      } catch (error) {
        $("datasetUploadProgress").className = "status-line bad";
        $("datasetUploadProgress").textContent = `${added} of ${total} added. The rest are still in the list.`;
        showError(error.message);
      } finally {
        uploadBusy = false;
        updateUploadButton();
        if (added) await refresh();
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
      btn.onclick = async () => {
        btn.disabled = true;
        try {
          await post("/api/stop-job", {key: btn.dataset.stop});
        } catch (error) {
          showError(error.message);
        } finally {
          btn.disabled = false;
          refresh();
        }
      };
    });
    $("generate").onclick = async () => {
      const ref = state.references.find(r => r.id === $("reference").value);
      if (!ref) return alert("Pick a reference clip.");
      $("generate").disabled = true;
      $("generateStatus").className = "status-line";
      if (!(state && state.api_ready)) {
        $("generateStatus").textContent = state && state.gpu_busy
          ? "Starting the voice engine first. Training is using the same GPU, so both can run slower or run out of memory."
          : "Starting the voice engine first...";
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
        const line = {text: $("targetText").value.trim(), url: result.url, name: result.name, createdAt: Date.now()};
        generateOutputs.push(line);
        showGeneratedLine(line, false);
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

    // Restore audio: LavaSR's noise remover and high-frequency restoration, for recordings that
    // can then be added to a dataset. Every result is a new recording in the list, so the steps can
    // be combined: for example noise removal first, then high frequencies.
    let restoreFocus = "";
    const restoreChecked = new Set();
    let restoreListSignature = "";
    // The result to show when it appears, if the view is still on the recording it is made from.
    let restoreAwait = null;
    const RESTORE_SETTING_IDS = ["restoreMusic", "restoreMusicModel", "restoreEcho", "restoreDenoise", "restoreEnhance", "restoreQuality", "restoreKeep", "restoreModel", "restorePieces"];
    // The options that need restore_env, which Setup-Restore-Env.bat creates.
    const LAVASR_IDS = ["restoreDenoise", "restoreEnhance", "restoreQuality", "restoreKeep", "restoreModel", "restorePieces"];

    function setTrainingMode(mode) {
      $("tab-training").dataset.mode = mode;
      document.querySelectorAll(".mode-tabs [data-mode]").forEach(button => {
        button.setAttribute("aria-selected", button.dataset.mode === mode ? "true" : "false");
      });
      try { localStorage.setItem("voiceDashboardTrainingMode", mode); } catch (_) {}
    }

    function restoreItems() {
      return (state && state.restore && state.restore.items) || [];
    }

    function restoreItem(id) {
      return restoreItems().find(item => item.id === id) || null;
    }

    function restoreTargets() {
      return restoreChecked.size ? [...restoreChecked] : (restoreFocus ? [restoreFocus] : []);
    }

    function describeRestore(item) {
      return item.steps && item.steps.length ? item.steps[item.steps.length - 1] : `original, ${Number(item.sample_rate) / 1000} kHz`;
    }

    function renderRestore() {
      const restore = (state && state.restore) || {installed: false, items: []};
      const items = restore.items || [];
      const job = state && state.jobs && state.jobs.restore;
      const status = job ? job.status : "";
      if (restoreAwait && restoreFocus !== restoreAwait.from) restoreAwait = null;
      if (restoreAwait && items.some(item => item.id === restoreAwait.to)) {
        restoreFocus = restoreAwait.to;
        restoreAwait = null;
      }
      if (!items.some(item => item.id === restoreFocus)) restoreFocus = items.length ? items[0].id : "";
      [...restoreChecked].forEach(id => { if (!items.some(item => item.id === id)) restoreChecked.delete(id); });
      // Music and echo removal run in WSL, so the tab works before restore_env is installed.
      $("restoreMissing").hidden = Boolean(restore.installed);
      LAVASR_IDS.forEach(id => { $(id).disabled = !restore.installed; });
      $("restoreEmpty").hidden = items.length > 0;
      $("restoreSteps").hidden = !items.length;
      // Redrawn only when the list or the recording in view changes, so focus stays put.
      const signature = restoreFocus + "|" + items.map(item => item.id).join(",");
      if (signature !== restoreListSignature) {
        restoreListSignature = signature;
        renderRestoreList(items);
      }
      renderRestoreFocus();
      const running = status === "running";
      $("restoreRun").disabled = running || !restoreFocus;
      if (job) {
        const progress = (job.lines || []).filter(line => /^(Loading|Downloading|Separated|Restored|Failed|Done|\d+ of \d+ file)/.test(line)).slice(-1)[0] || "Starting...";
        const failed = `${/ failed\.$/.test(progress) ? progress : "Failed."} See Logs for details.`;
        $("restoreStatus").textContent = running ? progress : (status === "done" ? "Done." : (status === "failed" ? failed : "Stopped."));
        $("restoreStatus").className = status === "failed" ? "status-line bad" : (status === "done" ? "status-line good" : "status-line");
      }
    }

    function renderRestoreList(items) {
      const list = $("restoreList");
      list.innerHTML = "";
      items.forEach(item => {
        const row = document.createElement("div");
        row.className = "restore-row";
        const check = document.createElement("input");
        check.type = "checkbox";
        check.checked = restoreChecked.has(item.id);
        check.setAttribute("aria-label", `Select ${item.name}`);
        check.onchange = () => {
          if (check.checked) restoreChecked.add(item.id);
          else restoreChecked.delete(item.id);
          renderRestoreFocus();
        };
        const button = document.createElement("button");
        button.type = "button";
        button.className = "history-item";
        button.setAttribute("aria-current", item.id === restoreFocus ? "true" : "false");
        button.setAttribute("aria-selected", item.id === restoreFocus ? "true" : "false");
        const name = document.createElement("span");
        name.className = "history-text";
        name.textContent = item.name;
        const meta = document.createElement("span");
        meta.className = "history-meta";
        const what = document.createElement("span");
        what.textContent = describeRestore(item);
        const time = document.createElement("span");
        time.textContent = formatClock((item.created || 0) * 1000);
        meta.append(what, time);
        button.append(name, meta);
        button.onclick = () => {
          restoreFocus = item.id;
          renderRestore();
        };
        row.append(check, button);
        list.appendChild(row);
      });
    }

    function renderRestoreFocus() {
      const items = restoreItems();
      const selected = restoreChecked.size;
      $("restoreCount").textContent = items.length
        ? `${items.length} recording${items.length === 1 ? "" : "s"}${selected ? `, ${selected} selected` : ""}`
        : "No audio yet";
      const item = restoreItem(restoreFocus);
      if (!item) return;
      const parent = item.made_from ? restoreItem(item.made_from) : null;
      $("restoreTitle").textContent = item.name;
      $("restoreInfo").textContent = `${Number(item.sample_rate) / 1000} kHz · ${item.duration} s`;
      $("restoreParentBlock").hidden = !parent;
      if (!parent) $("restoreParent").pause();
      $("restoreItemLabel").textContent = item.made_from ? "After" : "Recording";
      // Only a different recording is loaded, so a refresh does not stop playback.
      if (parent && $("restoreParent").dataset.item !== parent.id) {
        $("restoreParent").src = parent.url;
        $("restoreParent").dataset.item = parent.id;
      }
      if ($("restoreItem").dataset.item !== item.id) {
        $("restoreItem").src = item.url;
        $("restoreItem").dataset.item = item.id;
      }
      let history = item.steps && item.steps.length ? `Made by: ${item.steps.join(", then ")}.` : "Not restored yet.";
      if (item.made_from && !parent) history += " The recording it was made from was deleted.";
      $("restoreHistory").textContent = history;
      $("restoreRun").textContent = selected ? `Restore ${selected} selected` : "Restore";
      $("restoreToDataset").textContent = selected ? `Add ${selected} to dataset` : "Add to dataset";
      $("restoreDelete").textContent = selected ? `Delete ${selected}` : "Delete";
    }

    function restoreSettings() {
      return {
        music: $("restoreMusic").checked ? $("restoreMusicModel").value : "",
        echo: $("restoreEcho").checked,
        denoise: $("restoreDenoise").checked && !$("restoreDenoise").disabled,
        enhance: $("restoreEnhance").checked && !$("restoreEnhance").disabled,
        quality: $("restoreQuality").value,
        keep_below: $("restoreKeep").value ? Number($("restoreKeep").value) * 1000 : null,
        model: $("restoreModel").value,
        pieces: $("restorePieces").checked,
      };
    }

    document.querySelectorAll(".mode-tabs [data-mode]").forEach(button => {
      button.addEventListener("click", () => setTrainingMode(button.dataset.mode));
    });
    let savedTrainingMode = "train";
    try { savedTrainingMode = localStorage.getItem("voiceDashboardTrainingMode") || "train"; } catch (_) {}
    setTrainingMode(savedTrainingMode === "restore" ? "restore" : "train");

    try {
      const saved = JSON.parse(localStorage.getItem("voiceDashboardRestoreSettings") || "{}") || {};
      RESTORE_SETTING_IDS.forEach(id => {
        if (saved[id] === undefined) return;
        if ($(id).type === "checkbox") $(id).checked = Boolean(saved[id]);
        else $(id).value = saved[id];
      });
    } catch (_) {}
    RESTORE_SETTING_IDS.forEach(id => $(id).addEventListener("change", () => {
      const saved = {};
      RESTORE_SETTING_IDS.forEach(key => { saved[key] = $(key).type === "checkbox" ? $(key).checked : $(key).value; });
      try { localStorage.setItem("voiceDashboardRestoreSettings", JSON.stringify(saved)); } catch (_) {}
    }));

    $("restoreAddButton").onclick = () => $("restoreFiles").click();
    $("restoreAddEmpty").onclick = () => $("restoreFiles").click();
    $("restoreFiles").onchange = async () => {
      const files = Array.from($("restoreFiles").files || []);
      $("restoreFiles").value = "";
      for (let index = 0; index < files.length; index++) {
        $("restoreCount").textContent = `Adding ${index + 1} of ${files.length}...`;
        try {
          const result = await post("/api/restore-upload", {filename: files[index].name, audio_base64: await fileToBase64(files[index])});
          restoreFocus = result.item.id;
        } catch (error) {
          showError(`${files[index].name}: ${error.message}`);
        }
      }
      await refresh();
    };
    $("restoreRun").onclick = async () => {
      $("restoreRun").disabled = true;
      try {
        const result = await post("/api/restore-run", {item_ids: restoreTargets(), settings: restoreSettings()});
        const made = (result.results || {})[restoreFocus];
        restoreAwait = made ? {from: restoreFocus, to: made} : null;
        $("restoreStatus").className = "status-line";
        $("restoreStatus").textContent = "Starting...";
        refresh();
      } catch (error) {
        $("restoreRun").disabled = false;
        showError(error.message);
      }
    };
    $("restoreToDataset").onclick = async () => {
      if (!state || !(state.datasets || []).length) return showError("Create a dataset first, on the Train tab.");
      const chosen = restoreTargets().map(restoreItem).filter(Boolean);
      const long = chosen.filter(item => Number(item.duration) > CUT_LONGER_THAN);
      const items = chosen.filter(item => !long.includes(item));
      const files = [];
      $("restoreToDataset").disabled = true;
      try {
        if (long.length) await post("/api/dataset-cut-restored", {dataset_id: $("trainingDataset").value, item_ids: long.map(item => item.id)});
        for (const item of items) {
          const response = await fetch(item.url);
          if (!response.ok) throw new Error(`${item.name} could not be loaded. It may have been deleted.`);
          files.push(new File([await response.blob()], `${item.name}.wav`, {type: "audio/wav"}));
        }
      } catch (error) {
        return showError(error.message);
      } finally {
        $("restoreToDataset").disabled = false;
      }
      datasetUploadFiles.push(...files);
      renderDatasetUploads();
      queueTranscripts(files);
      setTrainingMode("train");
      if (long.length) refresh();
      $("clipsStep").scrollIntoView({block: "start"});

    };
    $("restoreDownload").onclick = () => {
      const item = restoreItem(restoreFocus);
      if (!item) return;
      const link = document.createElement("a");
      link.href = item.url;
      link.download = `${item.name}${item.made_from ? "-restored" : ""}.wav`;
      document.body.append(link);
      link.click();
      link.remove();
    };
    $("restoreDelete").onclick = async () => {
      const ids = restoreTargets();
      if (!ids.length || !confirm(`Delete ${ids.length === 1 ? "this recording" : `${ids.length} recordings`}? This cannot be undone.`)) return;
      ["restoreItem", "restoreParent"].forEach(id => {
        $(id).pause();
        $(id).removeAttribute("src");
        $(id).load();
        delete $(id).dataset.item;
      });
      for (const id of ids) {
        try { await post("/api/restore-delete", {item_id: id}); }
        catch (error) { showError(error.message); }
      }
      restoreChecked.clear();
      restoreFocus = "";
      refresh();
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
