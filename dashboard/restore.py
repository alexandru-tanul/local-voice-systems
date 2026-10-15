"""Restore audio: music and echo removal with GPT-SoVITS's UVR5 models in WSL, then noise removal
and high frequencies with LavaSR in restore_env."""

import json
import os
import shutil
import subprocess
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

from . import config, datasets, jobs, system, wsl


# Restore audio. The LavaSR worker runs in restore_env, which Setup-Restore-Env.bat creates, so the
# dashboard itself needs no audio libraries. Each recording and each result is a WAV in RESTORE_DIR
# with a JSON file beside it, so results written by the worker appear without shared state.
RESTORE_DIR = config.OUTPUT_DIR / "restored"
RESTORE_PYTHON = Path(config.APP_CONFIG.get("restore_python") or config.ROOT / "restore_env" / "Scripts" / "python.exe")
RESTORE_QUALITIES = {"detect", "file", 8000, 16000, 22050, 24000, 32000, 44100, 48000}


def restore_item_id(name):
    # A result keeps its recording's name, and the clock can read the same twice in one run, so
    # random digits keep the ids of one run apart.
    return f"{datasets.slugify(name, 'audio')[:40]}-{datetime.now():%Y%m%d-%H%M%S}-{os.urandom(4).hex()}"


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
    audio = datasets.decode_audio_upload(audio_base64)
    filename = str(filename or "audio")
    name = Path(filename).stem or "audio"
    item_id = restore_item_id(name)
    RESTORE_DIR.mkdir(parents=True, exist_ok=True)
    fd, source_name = tempfile.mkstemp(prefix="restore-", suffix=Path(filename).suffix.lower() or ".audio", dir=str(config.OUTPUT_DIR))
    os.close(fd)
    source = Path(source_name)
    target = RESTORE_DIR / f"{item_id}.wav"
    try:
        source.write_bytes(audio)
        datasets.convert_audio(source, target, sample_rate=None)
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
    datasets.write_text_atomic(RESTORE_DIR / f"{item_id}.json", json.dumps(item, indent=2))
    jobs.log(f"Added {Path(filename).name} to Restore audio.")
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
        if config.WSL_PATHS_ERROR:
            raise RuntimeError(config.WSL_PATHS_ERROR)
        with jobs.STATE_LOCK:
            starting = jobs.GPU_RESERVATIONS > 0
        if starting or any(jobs.job_running(key) for key in jobs.GPU_JOBS):
            raise RuntimeError("Removing music or echo needs the GPU, which training is using. Wait until it ends.")
        if jobs.job_running("compare"):
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
    with jobs.STATE_LOCK:
        current = jobs.JOBS.get("restore")
        if current and current.running():
            raise RuntimeError("Restore audio is already running.")
        # A stopped run leaves its work folder, and a result it was still writing.
        for leftover in [*config.TMP_DIR.glob(f"restore-{config.DASHBOARD_PORT}-*"), *RESTORE_DIR.glob("*.part")]:
            if leftover.is_dir():
                shutil.rmtree(leftover, ignore_errors=True)
            else:
                leftover.unlink(missing_ok=True)
        job = jobs.ThreadJob("Restore audio", lambda job: run_restore(job, items, settings, separation, lavasr))
        # Counted as a GPU job from the start, so nothing else starts on the GPU meanwhile.
        job.uses_gpu = bool(separation)
        jobs.JOBS["restore"] = job
        job.start()
    jobs.log(f"Restoring {len(chosen)} recording(s).")
    return {item["id"]: item["out_id"] for item in items}


def run_restore(job, items, settings, separation, lavasr):
    """The steps of a Restore audio run, in order: music and echo in WSL, then LavaSR."""
    # The port in the name tells this dashboard's runs from those of another one with the same TMP_DIR.
    work = config.TMP_DIR / f"restore-{config.DASHBOARD_PORT}-{datetime.now():%Y%m%d-%H%M%S}-{os.urandom(4).hex()}"
    work.mkdir(parents=True)
    count = len(items)
    try:
        if separation:
            items = separate_for_restore(job, items, separation, work)
        if items and lavasr:
            job_path = work / "lavasr.json"
            datasets.write_text_atomic(job_path, json.dumps({"out_dir": str(RESTORE_DIR), "settings": settings, "items": items}, indent=2))
            env = os.environ.copy()
            env.update({"HF_HOME": str(config.CACHE_DIR / "huggingface"), "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
                        "PYTHONIOENCODING": "utf-8", "TEMP": str(config.TMP_DIR), "TMP": str(config.TMP_DIR)})
            job.stream([str(RESTORE_PYTHON), str(config.ROOT / "restore_audio.py"), str(job_path)], env=env)
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
    runtime = wsl.voice_runtime()
    datasets.write_text_atomic(job_path, json.dumps({"steps": steps, "device": runtime["device"], "is_half": runtime["is_half"], "items": items}, indent=2))
    jobs.free_gpu_for("music and echo removal")
    try:
        script = (
            f"cd {wsl.shquote(config.WSL_GSV_ROOT)} && "
            f"{wsl.shquote(config.WSL_PYTHON)} -s \"$(wslpath -u {wsl.shquote(str(config.ROOT / 'separate_audio.py'))})\" {wsl.shquote(str(job_path))}"
        )
        job.stream(wsl.wsl_command(wsl.voice_environment() + script), stop_script=wsl.WSL_STOP_SCRIPTS["separate"])
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
        "duration": datasets.wav_seconds(target),
        "made_from": item["id"],
        "steps": item["steps"],
        "created": time.time(),
    }
    datasets.write_text_atomic(RESTORE_DIR / f"{item['out_id']}.json", json.dumps(sidecar, indent=2))


def kill_orphan_restore_workers():
    """Stop Restore audio runs that an earlier run of this dashboard left, when it was closed or
    crashed. Only restore_audio.py with a job file of this dashboard, by TMP_DIR and port, matches."""
    marker = str(config.TMP_DIR / f"restore-{config.DASHBOARD_PORT}-").replace("'", "''")
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
        f"Where-Object {{ $_.CommandLine -like '*restore_audio.py*' -and $_.CommandLine.Contains('{marker}') }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    try:
        system.run_powershell(script, timeout=8)
    except (OSError, subprocess.SubprocessError) as exc:
        jobs.log(f"Could not stop an earlier Restore audio run: {exc}")
