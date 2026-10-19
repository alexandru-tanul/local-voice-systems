"""Speech recognition by faster-whisper, in relay_asr_server.py: starting it, the transcripts it
writes for datasets, and Mic to Voice's lines, which it also translates."""

import base64
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import config, datasets, jobs, system


# Dataset transcripts come from faster-whisper in the speech recognition server, with a larger
# model than Mic to Voice uses, since a transcript is written once and should be exact.
DATASET_ASR_MODEL = str(config.APP_CONFIG.get("dataset_asr_model") or "")


def dataset_asr_model(language):
    return DATASET_ASR_MODEL or ("small.en" if language == "en" else "small")


# A speech recognition server from before an update knows only what it knew then.
OLDER_VERSION = "Speech recognition is running from an older version. Press Stop Everything, then try again."
# The first download of a model goes on in the speech recognition server after the request stops waiting.
STILL_DOWNLOADING = ("Speech recognition did not answer within 15 minutes, perhaps because a model is still downloading. "
                     "The download goes on, so try again in a while.")


def timed_out(exc):
    return isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError)


def whisper_language(language, model):
    # Whisper models before large-v3 have no Cantonese; they write it as Chinese.
    return "zh" if language == "yue" and "large-v3" not in model else language


def relay_model(language):
    """Mic to Voice's Whisper model for speech in this language: asr_model, and for the other
    languages the same model without .en, as the .en models know only English. base stands in for
    the distil models, which know only English with or without .en, and for a folder or a Hugging
    Face repository ending in .en, whose version for every language is not known."""
    model = config.RELAY_ASR_MODEL
    if language == "en":
        return model
    if "distil" in model.lower() or (model.endswith(".en") and any(mark in model for mark in "/\\")):
        return "base"
    return model[:-3] if model.endswith(".en") else model


def written(language):
    """The language a text is written in. Chinese and Cantonese are both written in Chinese
    characters, which the voice reads in either, so no translation is needed between them."""
    return "zh" if language == "yue" else language


def ask_relay_asr(path, payload):
    """The speech recognition server's answer to a request. The first use of a model downloads
    it, which can take minutes."""
    request = urllib.request.Request(
        config.RELAY_ASR_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(OLDER_VERSION) from exc
        body = exc.read()
    except (TimeoutError, urllib.error.URLError) as exc:
        if timed_out(exc):
            raise RuntimeError(STILL_DOWNLOADING) from exc
        raise
    try:
        result = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        result = {}
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "Speech recognition did not answer.")
    return result


def translate_relay(text, spoken, speaks, source=None):
    """A Mic to Voice line, said or typed in the language spoken, in the language the voice speaks.
    source is the language the text is written in, when it is not the one spoken."""
    if not text or written(spoken) == written(speaks):
        return text
    try:
        ensure_relay_asr()
        return ask_relay_asr("/translate", {"text": text, "source": source or spoken, "target": speaks})["text"]
    except Exception as exc:
        # The page tells this from a failed transcription by these words.
        raise RuntimeError(f"Translation failed: {exc}") from exc


def transcribe_relay(audio_base64, spoken="en", speaks="en"):
    """The words of a Mic to Voice recording, said in the language spoken, as the voice speaks
    them. heard is what was said, when the voice says it in another language."""
    ensure_relay_asr()
    model = relay_model(spoken)
    language = whisper_language(spoken, model)
    result = ask_relay_asr("/transcribe", {"audio_base64": audio_base64, "model": model, "language": language})
    if result.get("model") != model:
        # A server from before the dataset transcripts uses its own model for every language.
        raise RuntimeError(OLDER_VERSION)
    heard = result["text"].strip()
    # Cantonese that Whisper wrote as Chinese is translated from Chinese.
    text = translate_relay(heard, spoken, speaks, source=language)
    return {"text": text, "heard": heard if text != heard else ""}


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
    payload = {
        "audio_base64": base64.b64encode(Path(wav_path).read_bytes()).decode("ascii"),
        "model": model,
        "language": whisper_language(language, model),
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
        raise RuntimeError(OLDER_VERSION)
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


def warm_up_relay_asr(spoken="en", speaks="en"):
    """Load Mic to Voice's model for the language spoken, and the translation model when the voice
    speaks another, so a first download happens while the system starts."""
    translation = written(spoken) != written(speaks)
    url = config.RELAY_ASR_URL + "/warmup"
    if spoken != "en" or translation:
        # Only then, so English also starts with a server from before languages were added.
        url += "?" + urllib.parse.urlencode({"model": relay_model(spoken), "translation": int(translation)})
    # The server answers a failed model load with JSON that names the cause.
    try:
        with urllib.request.urlopen(url, timeout=900) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(OLDER_VERSION) from exc
        body = exc.read()
    except (TimeoutError, urllib.error.URLError) as exc:
        if timed_out(exc):
            raise RuntimeError(STILL_DOWNLOADING) from exc
        raise
    try:
        warmup = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        warmup = {}
    if not warmup.get("ok"):
        raise RuntimeError(warmup.get("error") or "Speech recognition model failed to load.")
