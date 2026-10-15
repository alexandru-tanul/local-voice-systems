"""Compare epochs: one test line said with each saved epoch of a model."""

import json
import re
import time
import urllib.error
import urllib.request

from . import config, datasets, engine, jobs, models, training, wsl


COMPARE_DIR = config.OUTPUT_DIR / "compare"


def model_epochs(model_name, kind, version="v2Pro"):
    """The saved checkpoints of a trained model, as (epoch, path) pairs. GPT training saves
    name-eN.ckpt and SoVITS training name_eN_sSTEPS.pth, in the folder of its version."""
    name = re.escape(datasets.slugify(model_name, "voice-model"))
    pattern = re.compile(rf"^{name}-e(\d+)\.ckpt$" if kind == "gpt" else rf"^{name}_e(\d+)_s\d+\.pth$")
    epochs = {}
    # The list is newest first, so of two files for one epoch the newer is kept.
    for row in models.model_files()[kind]:
        if (match := pattern.match(row["name"])) and (kind == "gpt" or models.sovits_version(row["path"]) == version):
            epochs.setdefault(int(match.group(1)), row["path"])
    return sorted(epochs.items())


def start_compare(model_name, kind, dataset_id, reference_wav, text, version="v2Pro"):
    """Generate one line with each saved epoch of a model, the other model at its last epoch.
    The SoVITS model is of the version given."""
    if kind not in {"gpt", "sovits"}:
        raise RuntimeError("Choose GPT or SoVITS epochs to compare.")
    if version not in models.SOVITS_VERSIONS:
        raise RuntimeError("Choose v2Pro or v2ProPlus.")
    text = " ".join(str(text or "").split())
    if not text:
        raise RuntimeError("Type a test line first.")
    if any(jobs.job_running(key) for key in jobs.GPU_JOBS):
        raise RuntimeError("Training is using the GPU. Compare epochs when it ends.")
    if jobs.separation_running():
        raise RuntimeError("Removing music or echo is using the GPU. Compare epochs when it ends.")
    descriptor = datasets.dataset_descriptor(dataset_id)
    wsl.wake_wsl()
    reference = next((ref for ref in datasets.parse_references(descriptor["id"]) if f"{ref['id']}.wav" == reference_wav), None)
    if not reference:
        raise RuntimeError("Choose a reference clip of 3 to 10 seconds.")
    epochs = model_epochs(model_name, kind, version)
    if not epochs:
        label = "GPT" if kind == "gpt" else ("SoVITS" if version == "v2Pro" else f"{version} SoVITS")
        raise RuntimeError(f"{datasets.slugify(model_name, 'voice-model')} has no saved {label} epochs yet.")
    other_kind = "sovits" if kind == "gpt" else "gpt"
    others = model_epochs(model_name, other_kind, version)
    if not others and other_kind == "sovits":
        # A trained SoVITS of the other version still speaks closer to the voice than a pretrained one.
        others = [row for other_version in models.SOVITS_VERSIONS if other_version != version
                  for row in model_epochs(model_name, "sovits", other_version)]
    # Without saved epochs of the other model, the pretrained one GPT-SoVITS starts from.
    other = others[-1][1] if others else next((row["path"] for row in models.model_files()[other_kind] if "pretrained" in row["path"]
                                               and (other_kind == "gpt" or models.sovits_version(row["path"]) == version)), "")
    if not other:
        raise RuntimeError("The pretrained models of GPT-SoVITS are missing. Run System Check.")
    settings = {"name": datasets.slugify(model_name, "voice-model"), "kind": kind, "epochs": epochs, "other": other,
                "reference": reference, "text": text, "language": descriptor["language"], "dataset": descriptor}
    with jobs.STATE_LOCK:
        current = jobs.JOBS.get("compare")
        if current and current.running():
            raise RuntimeError("A comparison is already running.")
        job = jobs.ThreadJob(f"Compare {settings['name']} epochs", lambda job: compare_epochs(job, settings))
        job.results = []
        jobs.JOBS["compare"] = job
        job.start()
    jobs.log(f"Comparing {len(epochs)} {kind} epochs of {settings['name']}.")


def compare_epochs(job, settings):
    # The voice engine reads the reference clip in WSL, where a clip added since the last sync is not yet.
    job.say("Syncing the dataset to WSL...")
    result = job.run(wsl.wsl_command(training.dataset_sync_script(settings["dataset"])), timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"The dataset could not be synced: {(result.stderr or result.stdout).strip()[-300:]}")
    if not engine.api_ready():
        job.say("Starting the voice engine...")
        first = settings["epochs"][0][1]
        engine.start_api(first if settings["kind"] == "gpt" else settings["other"], first if settings["kind"] == "sovits" else settings["other"])
        for _ in range(720):
            job.check_stop()
            if engine.api_ready():
                break
            if engine.API_PROCESS and not engine.API_PROCESS.running():
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
        request = urllib.request.Request(config.API_URL + "/tts", data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with engine.GENERATION_LOCK:
            engine.set_api_weights(gpt, sovits)
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
