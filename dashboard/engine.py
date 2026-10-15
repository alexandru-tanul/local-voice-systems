"""The voice engine, GPT-SoVITS's api_v2.py in WSL: starting and stopping it, loading models into
it, and saving the lines it generates."""

import io
import json
import threading
import time
import urllib.parse
import urllib.request
import wave
from datetime import datetime

from . import config, datasets, jobs, system, wsl


API_PROCESS = None
API_GPT_PATH = None
API_SOVITS_PATH = None

# The voice engine handles one generation at a time, including its model switch.
GENERATION_LOCK = threading.Lock()
ENGINE_LOCK = threading.Lock()


def api_ready():
    return system.port_open(config.TTS_PORT)


def write_tts_config(gpt_path, sovits_path):
    runtime = wsl.voice_runtime()
    text = f"""custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: {runtime['device']}
  is_half: {str(runtime['is_half']).lower()}
  t2s_weights_path: {json.dumps(gpt_path)}
  version: v2Pro
  vits_weights_path: {json.dumps(sovits_path)}
"""
    path = config.WSL_GSV_WIN / "TEMP" / "dashboard_tts_infer.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "TEMP/dashboard_tts_infer.yaml"


def start_api(gpt_path, sovits_path):
    global API_PROCESS, API_GPT_PATH, API_SOVITS_PATH
    # Text to Voice and Mic to Voice can both start the engine; one check-and-start runs at a time.
    with ENGINE_LOCK:
        if API_PROCESS and API_PROCESS.running():
            jobs.log("Inference engine is already running.")
            return True
        if not api_ready():
            config_path = write_tts_config(gpt_path, sovits_path)
            script = (
                wsl.voice_environment() + "ulimit -l 2097152; "
                f"cd {wsl.shquote(config.WSL_GSV_ROOT)} && "
                f"{wsl.shquote(config.WSL_PYTHON)} api_v2.py -a 127.0.0.1 -p {config.TTS_PORT} -c {wsl.shquote(config_path)}"
            )
            API_PROCESS = jobs.ManagedProcess("inference engine", wsl.wsl_command(script), stop_script=wsl.WSL_STOP_SCRIPTS["api"])
            API_PROCESS.start()
            API_GPT_PATH = gpt_path
            API_SOVITS_PATH = sovits_path
            jobs.log(f"Starting inference engine on {config.API_URL}.")
            return True
    jobs.log("A voice engine is already accepting connections; applying the selected models.")
    set_api_weights(gpt_path, sovits_path)
    return True


def stop_api():
    global API_PROCESS, API_GPT_PATH, API_SOVITS_PATH
    try:
        urllib.request.urlopen(config.API_URL + "/control?command=exit", timeout=1)
    except Exception:
        pass
    if API_PROCESS:
        API_PROCESS.stop()
    else:
        # An engine left by an earlier dashboard run.
        wsl.stop_in_wsl(wsl.WSL_STOP_SCRIPTS["api"])
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
        url = config.API_URL + "/" + endpoint + "?" + urllib.parse.urlencode({"weights_path": path})
        with urllib.request.urlopen(url, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status >= 400:
                raise RuntimeError(body)
        if endpoint == "set_gpt_weights":
            API_GPT_PATH = path
        else:
            API_SOVITS_PATH = path
        jobs.log(f"Loaded {endpoint.replace('set_', '').replace('_weights', '')} weights in {time.time() - started:.1f}s.")


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


def new_output_path(folder, voice):
    # Call with GENERATION_LOCK held, so no other generation takes the same name meanwhile.
    folder.mkdir(parents=True, exist_ok=True)
    stem = datasets.slugify(voice, "voice") + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = folder / f"{stem}.wav"
    counter = 2
    while path.exists():
        path = folder / f"{stem}-{counter}.wav"
        counter += 1
    return path


def prune_relay_outputs(keep=config.RELAY_OUTPUT_LIMIT):
    files = sorted(config.RELAY_OUTPUT_DIR.glob("*.wav"), key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    for path in files[keep:]:
        try:
            path.unlink()
        except OSError:
            pass
