"""Speech recognition by faster-whisper, in relay_asr_server.py: starting it, and the transcripts it
writes for datasets."""

import base64
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, datasets, jobs, system


# Dataset transcripts come from faster-whisper in the speech recognition server, with a larger
# model than Mic to Voice uses, since a transcript is written once and should be exact.
DATASET_ASR_MODEL = str(config.APP_CONFIG.get("dataset_asr_model") or "")


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
        with jobs.STATE_LOCK:
            proc = jobs.JOBS.get("relay-asr")
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
            config.RELAY_ASR_URL + "/transcribe",
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
    return datasets.clean_transcript(result.get("text"))


def transcribe_dataset_upload(dataset_id, filename, audio_base64):
    descriptor = datasets.dataset_descriptor(dataset_id)
    audio = datasets.decode_audio_upload(audio_base64)
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, source_name = tempfile.mkstemp(prefix="transcribe-", suffix=Path(str(filename or "")).suffix.lower() or ".audio", dir=str(config.TMP_DIR))
    os.close(fd)
    source = Path(source_name)
    converted = source.with_name(source.stem + "-16k.wav")
    try:
        source.write_bytes(audio)
        datasets.convert_audio(source, converted, sample_rate=16000)
        return transcribe_dataset_wav(converted, descriptor["language"])
    finally:
        for path in (source, converted):
            path.unlink(missing_ok=True)


def relay_asr_ready():
    return system.port_open(config.RELAY_ASR_PORT)


def start_relay_asr():
    if relay_asr_ready():
        jobs.log("Speech recognition is already running.")
        return True
    if not config.RELAY_ASR_PYTHON.exists():
        raise RuntimeError("The speech recognition environment is missing. Run Setup-Relay-Env.bat.")
    config.RELAY_ASR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.RELAY_ASR_TMP.mkdir(parents=True, exist_ok=True)
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    relay_env = os.environ.copy()
    relay_env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "HF_HOME": str(config.CACHE_DIR / "huggingface"),
            "HF_HUB_CACHE": str(config.CACHE_DIR / "huggingface" / "hub"),
            "PIP_CACHE_DIR": str(config.CACHE_DIR / "pip"),
            "TEMP": str(config.TMP_DIR),
            "TMP": str(config.TMP_DIR),
            "RELAY_ASR_MODEL_DIR": str(config.RELAY_ASR_MODEL_DIR),
            "RELAY_ASR_TMP": str(config.RELAY_ASR_TMP),
            "RELAY_ASR_MODEL": config.RELAY_ASR_MODEL,
            "RELAY_ASR_PORT": str(config.RELAY_ASR_PORT),
        }
    )
    proc = jobs.ManagedProcess(
        "Speech recognition",
        [str(config.RELAY_ASR_PYTHON), str(config.ROOT / "relay_asr_server.py")],
        cwd=config.ROOT,
        env=relay_env,
    )
    with jobs.STATE_LOCK:
        current = jobs.JOBS.get("relay-asr")
        if current and current.running():
            # Started, but not listening yet. The caller waits for the port.
            return True
        jobs.JOBS["relay-asr"] = proc
    proc.start()
    jobs.log(f"Speech recognition starting on {config.RELAY_ASR_URL}.")
    return True


def warm_up_relay_asr():
    # The server answers a failed model load with JSON that names the cause.
    try:
        with urllib.request.urlopen(config.RELAY_ASR_URL + "/warmup", timeout=180) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
    try:
        warmup = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        warmup = {}
    if not warmup.get("ok"):
        raise RuntimeError(warmup.get("error") or "Speech recognition model failed to load.")
