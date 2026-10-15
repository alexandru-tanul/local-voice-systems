"""Settings from config.json, and the folders, ports and addresses that follow from them."""

import json
import os
from pathlib import Path


# The project folder, above this package.
ROOT = Path(__file__).resolve().parent.parent
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
