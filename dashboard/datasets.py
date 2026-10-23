"""Datasets: folders of WAV clips with their transcripts, in datasets/. Also the reference clips
that Text to Voice offers, and the audio conversion that adding a clip uses."""

import base64
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import wave
from pathlib import Path

from . import config, jobs


MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 10.0
LANGUAGES = {"en", "zh", "ja", "ko", "yue"}


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
    root = (config.DATASETS_DIR / dataset_id).resolve()
    if root.parent != config.DATASETS_DIR.resolve() or not root.exists():
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
        "wsl_root": f"{config.WSL_DATASETS_ROOT}/{dataset_id}",
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
    for manifest_path in sorted(config.DATASETS_DIR.glob("*/dataset.json")):
        try:
            rows.append(dataset_summary(dataset_descriptor(manifest_path.parent.name)))
        except Exception as exc:
            jobs.log(f"Could not read dataset {manifest_path.parent.name}: {exc}")
    return rows


def create_dataset(name, speaker="speaker", language="en"):
    clean_name = str(name or "").strip()
    if not clean_name:
        raise RuntimeError("Enter a dataset name.")
    dataset_id = dataset_id_for_name(clean_name)
    root = config.DATASETS_DIR / dataset_id
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
    jobs.log(f"Created dataset {clean_name} ({dataset_id}).")
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
    # In TMP_DIR, where the dashboard removes what a crash leaves, and not in the served outputs/.
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    fd, source_name = tempfile.mkstemp(prefix="upload-", suffix=suffix, dir=str(config.TMP_DIR))
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
    jobs.log(f"Added {wav_name} to dataset {descriptor['name']}.")
    return dataset_summary(dataset_descriptor(dataset_id))


def wav_seconds(path):
    try:
        with wave.open(str(path), "rb") as audio:
            return round(audio.getnframes() / audio.getframerate(), 2)
    except (OSError, EOFError, ZeroDivisionError, wave.Error):
        return None


# SoVITS training keeps only clips longer than 0.6 s and shorter than 54 s
# (GPT_SoVITS/module/data_utils.py).
TRAINING_MIN_SECONDS = 0.6
TRAINING_MAX_SECONDS = 54.0


def training_skips(seconds):
    """Why training skips a clip of this length, or "" when it uses it. The page shows the reason
    at the clip's length, and Check clips lists the clip."""
    if seconds is None:
        return ""
    if seconds <= TRAINING_MIN_SECONDS:
        return f"Training skips clips of {TRAINING_MIN_SECONDS:g} s or less."
    if seconds >= TRAINING_MAX_SECONDS:
        return f"Training skips clips of {TRAINING_MAX_SECONDS:g} s or more."
    return ""


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
        duration = wav_seconds(path)
        clips.append(
            {
                "wav": wav,
                "text": entry.get("text", ""),
                "duration": duration,
                "skipped": training_skips(duration),
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
                jobs.log(f"{path.name} is out of the dataset, but its file is still in use, so it stays in wavs/.")
            time.sleep(0.1)
    jobs.log(f"Deleted {wav} from dataset {descriptor['name']}.")


def decode_audio_upload(audio_base64):
    try:
        audio = base64.b64decode(audio_base64, validate=True)
    except Exception as exc:
        raise RuntimeError("The uploaded audio data is invalid.") from exc
    if not audio or len(audio) > 100 * 1024 * 1024:
        raise RuntimeError("Audio files must be between 1 byte and 100 MB.")
    return audio
