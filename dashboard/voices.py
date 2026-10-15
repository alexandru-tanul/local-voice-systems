"""Export voice and Import a voice: one zip file with both models and the reference clips."""

import hashlib
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

from . import config, datasets, jobs, models, speech_api, training, wsl


# A voice file holds voice.json, both models in models/, and the reference clips. Its clips become a
# small dataset on import, as the references of every voice are dataset clips.
VOICE_FILE_FORMAT = 1
VOICE_EXPORT = re.compile(r"^voice-export-[0-9a-f]{8}\.zip$")


def export_voice(gpt, sovits, dataset_id, reference_id, prompt_text, extra_ids):
    """Pack a voice into a zip file in TMP_DIR. Returns the file and a name to save it as."""
    wsl.wake_wsl()
    available = models.model_files()
    if gpt not in {row["path"] for row in available["gpt"]} or sovits not in {row["path"] for row in available["sovits"]}:
        raise RuntimeError("Choose a GPT and a SoVITS model first.")
    descriptor = datasets.dataset_descriptor(dataset_id)
    references = {ref["id"]: ref for ref in datasets.parse_references(descriptor["id"], valid_only=False)}
    reference = references.get(reference_id)
    if not reference or not reference["valid_reference"]:
        raise RuntimeError("Choose a reference clip of 3 to 10 seconds first.")
    extras = [references[ref_id] for ref_id in extra_ids or [] if ref_id in references and ref_id != reference_id
              and references[ref_id]["valid_aux_reference"]]
    manifest = {
        "format": VOICE_FILE_FORMAT,
        "name": descriptor["name"],
        "language": descriptor["language"],
        "version": models.sovits_version(sovits),
        "gpt": Path(gpt).name,
        "sovits": Path(sovits).name,
        "reference": {"file": "reference.wav", "text": datasets.clean_transcript(prompt_text) or reference["text"], "language": reference["lang"]},
        "extra_references": [{"file": f"extra-{index}.wav", "text": ref["text"], "language": ref["lang"]} for index, ref in enumerate(extras, 1)],
    }
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TMP_DIR / f"voice-export-{os.urandom(4).hex()}.zip"
    try:
        # The models are already compressed, so the zip only stores them.
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as bundle:
            bundle.writestr("voice.json", json.dumps(manifest, indent=2))
            bundle.write(config.WSL_GSV_WIN / gpt, f"models/{Path(gpt).name}")
            bundle.write(config.WSL_GSV_WIN / sovits, f"models/{Path(sovits).name}")
            for ref, entry in [(reference, manifest["reference"]), *zip(extras, manifest["extra_references"])]:
                bundle.write(descriptor["root"] / "wavs" / f"{ref['id']}.wav", entry["file"])
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    jobs.log(f"Exported the voice {descriptor['name']}.")
    return path, f"{datasets.slugify(descriptor['name'], 'voice')}.voice.zip"


def install_weight(bundle, member, folder):
    """Copy a model from a voice file into GPT-SoVITS, under a name no other model has."""
    name = Path(member).name
    target_dir = config.WSL_GSV_WIN / folder
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
        if version not in models.SOVITS_VERSIONS:
            raise RuntimeError(f"This voice is for GPT-SoVITS {version}, and the dashboard uses v2Pro and v2ProPlus.")
        gpt = f"models/{Path(str(manifest.get('gpt'))).name}"
        sovits = f"models/{Path(str(manifest.get('sovits'))).name}"
        references = [manifest.get("reference") or {}, *(manifest.get("extra_references") or [])][:10]
        names = set(bundle.namelist())
        if not gpt.endswith(".ckpt") or not sovits.endswith(".pth") or gpt not in names or sovits not in names:
            raise RuntimeError("The voice file is missing its models.")
        wsl.wake_wsl()
        if any(Path(str(ref.get("file"))).name not in names or not datasets.clean_transcript(ref.get("text")) for ref in references):
            raise RuntimeError("The voice file is missing a reference clip or its text.")
        if (any(bundle.getinfo(Path(str(ref["file"])).name).file_size > 100 * 1024 ** 2 for ref in references)
                or max(bundle.getinfo(gpt).file_size, bundle.getinfo(sovits).file_size) > 4 * 1024 ** 3):
            raise RuntimeError("The voice file holds files far larger than a voice needs.")
        gpt_path = install_weight(bundle, gpt, "GPT_weights_v2Pro")
        sovits_path = install_weight(bundle, sovits, f"SoVITS_weights_{version}")
        name = str(manifest.get("name") or "Imported voice").strip()[:60] or "Imported voice"
        base, counter = name, 2
        while (config.DATASETS_DIR / datasets.dataset_id_for_name(name)).exists():
            name = f"{base} {counter}"
            counter += 1
        dataset_id = datasets.create_dataset(name, language=datasets.language_code(manifest.get("language")))["id"]
        root = config.DATASETS_DIR / dataset_id
        config.TMP_DIR.mkdir(parents=True, exist_ok=True)
        try:
            for index, ref in enumerate(references):
                member = Path(str(ref["file"])).name
                source = config.TMP_DIR / f"import-{os.urandom(4).hex()}.wav"
                converted = source.with_name(source.stem + "-44k.wav")
                try:
                    source.write_bytes(bundle.read(member))
                    datasets.convert_audio(source, converted)
                    datasets.add_dataset_clip(root, converted, "reference" if index == 0 else f"extra-{index}", member,
                                     datasets.clean_transcript(ref["text"]), language=datasets.language_code(ref.get("language")))
                finally:
                    source.unlink(missing_ok=True)
                    converted.unlink(missing_ok=True)
        except BaseException:
            # No half-made voice stays behind, and a second try gets the same name.
            shutil.rmtree(root, ignore_errors=True)
            raise
    # The voice engine reads the clips in WSL.
    result = wsl.run_wsl(training.dataset_sync_script(datasets.dataset_descriptor(dataset_id)), timeout=180)
    if result.returncode != 0:
        jobs.log(f"The imported voice's clips could not be synced: {result.stdout} {result.stderr}")
        raise RuntimeError("The voice was imported, but its clips could not be copied to WSL. Press Sync Only on Voice Training.")
    speech_api.save_speech_settings(dataset_id, {
        "gpt": gpt_path, "sovits": sovits_path, "reference": "reference",
        "extra_references": [f"extra-{index}" for index in range(1, len(references))],
        "prompt_text": datasets.clean_transcript(references[0]["text"]), "text_lang": datasets.language_code(manifest.get("language")),
    })
    models.model_files()
    jobs.log(f"Imported the voice {name}.")
    return {"dataset": dataset_id, "name": name, "gpt": gpt_path, "sovits": sovits_path, "reference": "reference"}
