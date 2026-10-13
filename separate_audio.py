"""Remove music and sound effects, or echo and reverb, from recordings with the UVR5 models that
GPT-SoVITS includes in tools/uvr5.

The dashboard runs this in WSL, in the GPT-SoVITS folder and with its Python, with a job file that
lists the recordings (44.1 kHz stereo WAV files, as the models need) and the steps. Each result is a
mono WAV of the voice alone. The first run downloads the models GPT-SoVITS's installer offers with
--download-uvr5 (about 520 MB) into tools/uvr5/uvr5_weights.
"""
import hashlib
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

WEIGHTS = Path("tools/uvr5/uvr5_weights")
REVISION = "0c47645e02a7bc3688d7b263b0042c81e3cd82cd"
ZIP_URL = f"https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/{REVISION}/uvr5_weights.zip"
ZIP_SHA256 = "1317c4117aed4f08f1780df08d0e6c8d63b3c34b987f8d9bef4ef3bc1f290c21"
# The page's choices, and the model files GPT-SoVITS names them by.
MODELS = {
    "best": "model_bs_roformer_ep_317_sdr_12.9755",
    "main": "HP5_only_main_vocal",
    "all": "HP2_all_vocals",
    "echo": "VR-DeEchoDeReverb",
}
# The models read a whole recording into memory, so a long one goes through in 5 minute pieces
# that overlap by 2 seconds and fade into each other.
PIECE_SECONDS = 300
OVERLAP_SECONDS = 2


def model_file(name):
    return WEIGHTS / (name + (".ckpt" if "roformer" in name else ".pth"))


def ensure_weights(names):
    # A download that was stopped leaves its folder; only this script makes tmp* folders here.
    for folder in WEIGHTS.parent.glob("tmp*"):
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
    if all(model_file(name).is_file() for name in names):
        return
    print("Downloading the separation models, about 520 MB, the first time...", flush=True)
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WEIGHTS.parent) as folder:
        archive = Path(folder) / "uvr5_weights.zip"
        digest = hashlib.sha256()
        with urllib.request.urlopen(ZIP_URL, timeout=60) as response, archive.open("wb") as handle:
            size = int(response.headers.get("Content-Length") or 0)
            done = shown = 0
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if size and done * 10 // size > shown:
                    shown = done * 10 // size
                    print(f"Downloading the separation models: {shown * 10}%", flush=True)
        if digest.hexdigest() != ZIP_SHA256:
            raise RuntimeError("The downloaded separation models are damaged. Try again.")
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = WEIGHTS.parent / member.filename
                # Only files in uvr5_weights/, and none that are there already.
                if member.is_dir() or not member.filename.startswith("uvr5_weights/") or target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Written under another name and renamed, so a stop leaves no half model behind.
                part = target.with_name(target.name + ".part")
                with bundle.open(member) as source, open(part, "wb") as copy:
                    shutil.copyfileobj(source, copy)
                part.replace(target)


def linux_path(path):
    """The dashboard writes Windows paths; wslpath maps them the way this distro mounts drives."""
    if ":" not in path[:3]:
        return Path(path)
    return Path(subprocess.run(["wslpath", "-u", path], capture_output=True, text=True, check=True).stdout.strip())


def load_model(name, device, is_half):
    sys.path.insert(0, "tools/uvr5")
    if "roformer" in name:
        from bsroformer import Roformer_Loader

        # Without its own .yaml, GPT-SoVITS uses the settings this model was trained with.
        return Roformer_Loader(model_path=str(model_file(name)), config_path=str(WEIGHTS / f"{name}.yaml"), device=device, is_half=is_half)
    from vr import AudioPre, AudioPreDeEcho

    # These small models run in full precision. In half precision HP2 kept less of the voice than on
    # the CPU, and the first run in a process could return a different, wrong result.
    return (AudioPreDeEcho if "DeEcho" in name else AudioPre)(agg=10, model_path=str(model_file(name)), device=device, is_half=False)


def separate_whole(model, source, target):
    """Write the voice of a 44.1 kHz stereo WAV to target, as GPT-SoVITS's UVR5 page does."""
    with tempfile.TemporaryDirectory() as folder:
        voice, rest = Path(folder, "voice"), Path(folder, "rest")
        # Arguments as tools/uvr5/webui.py passes them: the recording, the folder for the rest, and
        # the folder for the voice. Each model names its files its own way, so the one file in the
        # voice folder is taken.
        model._path_audio_(str(source), str(rest), str(voice), "wav", False)
        produced = sorted(voice.glob("*.wav"))
        if len(produced) != 1:
            raise RuntimeError(f"expected one voice file, found {len(produced)}")
        shutil.move(str(produced[0]), str(target))


def separate(model, source, target):
    import numpy as np
    import soundfile as sf

    info = sf.info(str(source))
    rate = info.samplerate
    piece, overlap = PIECE_SECONDS * rate, OVERLAP_SECONDS * rate
    if info.frames <= piece + overlap:
        separate_whole(model, source, target)
        return
    with tempfile.TemporaryDirectory() as folder, sf.SoundFile(str(target), "w", rate, info.channels, subtype="PCM_16") as out:
        start = 0
        tail = None
        while start < info.frames:
            stop = min(info.frames, start + piece + overlap)
            part, voice = Path(folder, "part.wav"), Path(folder, "voice.wav")
            sf.write(str(part), sf.read(str(source), start=start, stop=stop, always_2d=True)[0], rate, subtype="PCM_16")
            separate_whole(model, part, voice)
            result = sf.read(str(voice), always_2d=True)[0]
            voice.unlink()
            # Padded or cut to the length of the piece, in case a model returns a few samples more or less.
            result = np.pad(result, ((0, max(0, stop - start - len(result))), (0, 0)))[: stop - start]
            if tail is not None:
                fade = np.linspace(0, 1, len(tail))[:, None]
                result[: len(tail)] = tail * (1 - fade) + result[: len(tail)] * fade
            if stop < info.frames:
                out.write(result[:-overlap])
                tail = result[-overlap:]
            else:
                out.write(result)
            start += piece


def to_mono(source, target):
    """One voice: the two channels averaged, a block at a time, so any length fits in memory."""
    import soundfile as sf

    with sf.SoundFile(str(source)) as reader, sf.SoundFile(str(target), "w", reader.samplerate, 1, subtype="PCM_16") as writer:
        for block in reader.blocks(blocksize=reader.samplerate * 30, always_2d=True):
            writer.write(block.mean(axis=1))


def main(job_path):
    import torch

    job = json.loads(linux_path(job_path).read_text(encoding="utf-8"))
    steps = [MODELS[step] for step in job["steps"]]
    device = job.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    is_half = bool(job.get("is_half")) and device == "cuda"
    ensure_weights(steps)
    items = job["items"]
    failed = 0
    for step_index, name in enumerate(steps):
        print(f"Loading {name}...", flush=True)
        model = load_model(name, device, is_half)
        last = step_index == len(steps) - 1
        for index, item in enumerate(items, 1):
            if item.get("failed"):
                continue
            started = time.perf_counter()
            source = linux_path(item["work"])
            result = source.with_name(f"{source.stem}-{step_index}.wav")
            try:
                separate(model, source, result)
                if last:
                    to_mono(result, linux_path(item["output"]))
                    result.unlink()
                else:
                    item["work"] = str(result)
                print(f"Separated {index} of {len(items)} ({name}): {item['name']} ({time.perf_counter() - started:.1f} s)", flush=True)
            except Exception as exc:
                failed += 1
                item["failed"] = True
                print(f"Failed {index} of {len(items)}: {item['name']} ({type(exc).__name__}: {exc})", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"Separated {len(items) - failed} of {len(items)} file(s).", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    # The dashboard's Stop ends this with SIGTERM; as an exit, it still removes its temporary files.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    sys.exit(main(sys.argv[1]))
