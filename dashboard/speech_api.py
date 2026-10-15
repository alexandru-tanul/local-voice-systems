"""OpenAI-style speech for other programs. A voice speaks with the models, reference clip and
settings of its last line on Text to Voice."""

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config, datasets, engine, jobs, models


# OpenAI-style speech for other programs on this PC: POST /v1/audio/speech, and the lists that
# Open WebUI reads, GET /v1/audio/voices, /v1/audio/models and /v1/models. A voice speaks as it
# last did on Text to Voice, with the models, reference clip and settings kept in speech.json.
SPEECH_SETTINGS = "speech.json"
SPEECH_API_KEY = str(config.APP_CONFIG.get("api_key") or "")
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
        descriptor = datasets.dataset_descriptor(dataset_id)
    except (RuntimeError, ValueError):
        return
    if not (settings.get("gpt") and settings.get("sovits") and settings.get("reference")):
        return
    reference = next((ref for ref in datasets.parse_references(descriptor["id"], valid_only=False) if ref["id"] == settings["reference"]), None)
    if reference and datasets.clean_transcript(settings.get("prompt_text")) == datasets.clean_transcript(reference["text"]):
        # Only a reference text edited on Text to Voice is kept. Otherwise the clip's transcript is
        # read when the voice speaks, so a later correction of it counts.
        settings = {key: value for key, value in settings.items() if key != "prompt_text"}
    datasets.write_text_atomic(descriptor["root"] / SPEECH_SETTINGS, json.dumps(settings, indent=2))


def speech_settings_from_generation(body):
    return {
        "gpt": str(body.get("gpt") or ""),
        "sovits": str(body.get("sovits") or ""),
        # Clips by name, so a voice still finds them if the datasets folder in WSL moves.
        "reference": Path(str(body.get("ref_audio_path") or "")).stem,
        "extra_references": [Path(str(path)).stem for path in body.get("aux_ref_audio_paths") or []],
        "prompt_text": datasets.clean_transcript(body.get("prompt_text")),
        "text_lang": datasets.language_code(body.get("text_lang")),
        **{key: body[key] for key in SPEECH_TUNING if key in body},
    }


def speech_voices():
    """The voices other programs can use: those that have spoken a line on Text to Voice."""
    return [{"id": row["id"], "name": row["name"]} for row in datasets.list_datasets() if (config.DATASETS_DIR / row["id"] / SPEECH_SETTINGS).is_file()]


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
    if config.WSL_PATHS_ERROR:
        raise SpeechApiError(503, config.WSL_PATHS_ERROR, "server_error")
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
    descriptor = datasets.dataset_descriptor(voice["id"])
    settings = json.loads((descriptor["root"] / SPEECH_SETTINGS).read_text(encoding="utf-8"))
    references = {ref["id"]: ref for ref in datasets.parse_references(descriptor["id"], valid_only=False)}
    reference = references.get(settings.get("reference"))
    if not reference or not reference["valid_reference"]:
        raise SpeechApiError(400, f"The reference clip of {voice['name']} is no longer in its dataset. Speak a line with it on Text to Voice again.",
                             param="voice")
    extras = [references[ref_id]["wsl_path"] for ref_id in settings.get("extra_references") or []
              if ref_id in references and ref_id != reference["id"] and references[ref_id]["valid_aux_reference"]]
    # Without this, the engine would speak with its pretrained models instead of a missing one.
    available = models.model_files()
    if not available["gpt"] and not available["sovits"]:
        raise SpeechApiError(503, "The models of GPT-SoVITS cannot be read. Run System Check in the dashboard.", "server_error")
    for kind, label in (("gpt", "GPT"), ("sovits", "SoVITS")):
        if settings.get(kind) not in {row["path"] for row in available[kind]}:
            raise SpeechApiError(400, f"The {label} model of {voice['name']}, {settings.get(kind)}, is no longer in GPT-SoVITS. "
                                      "Speak a line with the voice on Text to Voice again.", param="voice")
    if not engine.api_ready():
        # While a job uses the GPU, a program cannot start the voice engine.
        if jobs.gpu_busy():
            raise SpeechApiError(503, "Training or another GPU job is running, and the voice engine starts when it ends.", "server_error")
        engine.start_api(settings["gpt"], settings["sovits"])
        for _ in range(720):
            if engine.api_ready():
                break
            if engine.API_PROCESS and not engine.API_PROCESS.running():
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
    request = urllib.request.Request(config.API_URL + "/tts", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.time()
    with engine.GENERATION_LOCK:
        try:
            engine.set_api_weights(settings["gpt"], settings["sovits"])
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
    jobs.log(f"Speech API: {voice['name']} said {len(text)} characters in {time.time() - started:.1f}s.")
    return audio, content_type
