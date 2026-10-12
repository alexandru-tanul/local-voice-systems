"""Restore speech recordings with LavaSR: remove background noise, rebuild the high frequencies a
recording lacks, or both.

The dashboard runs this in restore_env, which Setup-Restore-Env.bat creates, with a job file that
lists the recordings and the settings. Each result is written as a 48 kHz WAV next to a JSON file
that the dashboard lists.

LavaSR works on a 16 kHz copy of the audio and predicts everything above 8 kHz. Its own merge step
then keeps the 16 kHz copy below the cutoff, which leaves an empty band between 8 kHz and the cutoff
for recordings better than 16 kHz. Here the real recording is kept below the cutoff instead.
"""
import json
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

MODEL_REPO = "YatharthS/LavaSR"
MODEL_REVISION = "b98dc8be472da45ab7b6346ad7997e1dfeb5911d"
# Half of the common sample rates: where a recording made at that rate stops.
STANDARD_BANDS = (4000, 8000, 11025, 12000, 16000, 22050, 24000)
OUTPUT_RATE = 48000
# Work in short pieces: 10 s at a time at 16 kHz, with 1 s more on both sides that is dropped
# afterwards, and 0.1 s where neighbouring pieces fade into each other. Recordings longer than a
# minute always do, as the whole recording at once needs about 9 MB of memory per second of it.
PIECE = 160000
CONTEXT = 16000
FADE = 1600
LONG = 60 * 16000


def detect_band(wav, sr):
    """The frequency where the recording's real sound stops: half the rate it was really made at.

    An 8 kHz recording saved as a 44.1 kHz file still stops near 4 kHz. Such an edge, from the recording,
    a codec or resampling, leaves almost nothing above it and drops steeply, while speech itself
    fades gradually. This returns the lowest common edge that does both, else the file's own limit.
    Levels are in dB under the voice's strongest part, measured on ffmpeg, MP3 and soxr cuts."""
    if len(wav) < 2048:
        return sr / 2
    # Without center=False the frames at both ends are padded with silence, and a recording cut
    # mid-sound then jumps there, which reads as every frequency and hides where the sound stops.
    # Long recordings are measured in at most 20000 frames, spread over all of it.
    spec = np.abs(librosa.stft(wav, n_fft=2048, hop_length=max(512, len(wav) // 20000), center=False)) ** 2
    energy = spec.sum(axis=0)
    loud = energy > energy.max() * 1e-3
    if not loud.any():
        return sr / 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    level = 10 * np.log10(spec[:, loud].mean(axis=1) + 1e-20)
    level -= level[(freqs >= 300) & (freqs <= 3000)].max()

    # The lowest edge that passes wins. 11.025 and 12 kHz are too close to tell apart, and the lower
    # one is the safe mistake: the restored audio then rebuilds 1 kHz of real sound instead of
    # leaving 1 kHz empty.
    for band in STANDARD_BANDS:
        if band * 1.2 >= sr / 2:
            break
        # Resampling filters finish cutting a few percent past the edge, so the band just above
        # is measured from 6% on.
        below = level[(freqs >= band * 0.80) & (freqs <= band * 0.95)].mean()
        above = level[(freqs >= band * 1.06) & (freqs <= band * 1.20)].mean()
        rest = level[freqs >= band * 1.06].mean()
        if above <= -55 and rest <= -70 and below - above >= 15:
            return band
    return sr / 2


def high_quality_resample(wav, sr):
    """The audio at the output rate, as a 1 x samples tensor."""
    resampled = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=OUTPUT_RATE, res_type="soxr_hq")
    return torch.from_numpy(np.ascontiguousarray(resampled)).unsqueeze(0)


def load_model(version, model_dir=None):
    from huggingface_hub import snapshot_download
    from LavaSR.model import LavaEnhance, LavaEnhance2

    enhancer = "enhancer_v2/*" if version == "v2" else "enhancer/*"
    path = model_dir or snapshot_download(
        MODEL_REPO, revision=MODEL_REVISION, allow_patterns=["denoiser/*", enhancer, "config.json"]
    )
    return (LavaEnhance2 if version == "v2" else LavaEnhance)(path, "cpu")


def run_models(model, x16, denoise, enhance):
    """The cleaned audio at 16 kHz and the predicted audio at 48 kHz, for 1 x n audio at 16 kHz."""
    length16 = x16.shape[-1]
    # Silence after the end, cut off again below: the models need half a second or so, and the
    # enhancer's last frame would otherwise stop short of the end of the recording.
    work = torch.nn.functional.pad(x16, (0, max(1024, 8000 - length16)))
    with torch.inference_mode():
        if denoise:
            work = model.denoiser_model.infer(work)
        prediction = None
        if enhance:
            wav48 = AF.resample(work, 16000, OUTPUT_RATE)
            core = model.bwe_model.bwe_model
            prediction = core.head(core.backbone(core.feature_extractor(wav48)))
            prediction = prediction.reshape(1, -1)[:, : length16 * 3].float()
    return work.reshape(1, -1)[:, :length16], prediction


def restore_piece(model, x16, real, denoise, enhance, merge):
    """The 48 kHz result for 1 x n audio at 16 kHz, and real, the same audio at 48 kHz."""
    cleaned, prediction = run_models(model, x16, denoise, enhance)
    # The real sound: the cleaned 16 kHz audio, which stops at 8 kHz, or the recording itself.
    # soxr, unlike torchaudio's resampler, leaves nothing above the edge, so Detect reads a
    # cleaned file as 16 kHz quality when it is restored again.
    if denoise:
        real = high_quality_resample(cleaned.reshape(-1).numpy(), 16000)
    if prediction is None:
        return real
    n = min(prediction.shape[-1], real.shape[-1])
    return merge(prediction[:, :n], real[:, :n].float())


def restore_in_pieces(model, x16, real, denoise, enhance, merge):
    """restore_piece on one piece after another, so memory use stays low for any length. The audio
    around each piece lets the models start and stop away from its edges, and the fades between
    pieces leave no clicks."""
    n = x16.shape[-1]
    restored = torch.zeros(1, n * 3)
    start = 0
    while True:
        end = min(n, start + PIECE + FADE)
        first, last = max(0, start - CONTEXT), min(n, end + CONTEXT)
        around = None if real is None else real[:, first * 3 : last * 3]
        part = restore_piece(model, x16[:, first:last], around, denoise, enhance, merge)
        part = part[:, (start - first) * 3 : (end - first) * 3]
        weight = torch.ones(part.shape[-1])
        if start > 0:
            weight[: FADE * 3] = torch.linspace(0, 1, FADE * 3)
        if end < n:
            weight[-FADE * 3 :] = torch.linspace(1, 0, FADE * 3)
        restored[:, start * 3 : start * 3 + part.shape[-1]] += part * weight
        if end >= n:
            return restored
        start += PIECE


def restore(model, wav, sr, denoise, enhance, quality="detect", keep_below=None, pieces=False):
    """Returns the restored audio at 48 kHz and what was done, in plain words."""
    from LavaSR.enhancer.linkwitz_merge import FastLRMerge

    if len(wav) == 0:
        raise ValueError("the recording is empty")
    notes = []
    if quality == "detect":
        band = detect_band(wav, sr)
        if enhance:
            notes.append(f"detected {band * 2 / 1000:g} kHz quality")
    elif quality == "file":
        band = sr / 2
    else:
        band = min(int(quality) / 2, sr / 2)
    # The noise remover keeps up to 8 kHz, and a lower-quality recording has less to keep.
    real_band = min(8000, band) if denoise else band
    cutoff = min(float(keep_below) if keep_below else real_band, real_band)
    nothing_to_rebuild = enhance and cutoff >= OUTPUT_RATE / 2
    if nothing_to_rebuild:
        # The 48 kHz result holds nothing above 24 kHz.
        enhance = False
    audio = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    x16 = AF.resample(audio, sr, 16000).unsqueeze(0)
    real = None if denoise else high_quality_resample(wav, sr)
    merge = FastLRMerge(sample_rate=OUTPUT_RATE, cutoff=int(cutoff), transition_bins=1024) if enhance else None
    if pieces or x16.shape[-1] > LONG:
        restored = restore_in_pieces(model, x16, real, denoise, enhance, merge)
    else:
        restored = restore_piece(model, x16, real, denoise, enhance, merge)
    if denoise:
        notes.append("noise removed")
    if enhance:
        notes.append(f"rebuilt above {cutoff / 1000:g} kHz")
    if nothing_to_rebuild:
        notes.append("already full quality, nothing rebuilt")
    return restored[0, : round(len(wav) * OUTPUT_RATE / sr)].numpy(), notes


def main(job_path):
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    out_dir = Path(job["out_dir"])
    settings = job["settings"]
    print(f"Loading LavaSR {settings['model']}. The first run downloads it (about 55 MB).", flush=True)
    model = load_model(settings["model"])
    items = job["items"]
    failed = 0
    for index, item in enumerate(items, 1):
        started = time.perf_counter()
        # One recording that cannot be read or restored does not stop the others.
        try:
            wav, sr = librosa.load(item["path"], sr=None, mono=True)
            restored, notes = restore(
                model, wav, sr, settings["denoise"], settings["enhance"],
                settings["quality"], settings.get("keep_below"), settings.get("pieces", False),
            )
        except Exception as exc:
            failed += 1
            reason = "the recording was deleted" if not Path(item["path"]).is_file() else f"{type(exc).__name__}: {exc}"
            print(f"Failed {index} of {len(items)}: {item['name']} ({reason})", flush=True)
            continue
        peak = float(np.max(np.abs(restored))) if restored.size else 0.0
        if peak > 0.99:
            restored = restored * (0.99 / peak)
        part = out_dir / f"{item['out_id']}.wav.part"
        sf.write(part, restored, OUTPUT_RATE, subtype="PCM_16", format="WAV")
        os.replace(part, out_dir / f"{item['out_id']}.wav")
        sidecar = {
            "id": item["out_id"],
            "name": item["name"],
            "file": f"{item['out_id']}.wav",
            "sample_rate": OUTPUT_RATE,
            "duration": round(len(restored) / OUTPUT_RATE, 2),
            "made_from": item["id"],
            "steps": item.get("steps", []) + [", ".join(notes) + f" ({settings['model']})"],
            "created": time.time(),
        }
        part = out_dir / f"{item['out_id']}.json.part"
        part.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        os.replace(part, out_dir / f"{item['out_id']}.json")
        print(f"Restored {index} of {len(items)}: {item['name']} ({time.perf_counter() - started:.1f} s)", flush=True)
    if failed:
        print(f"{len(items) - failed} of {len(items)} file(s) restored; {failed} failed.", flush=True)
        return 1
    print(f"Done. {len(items)} file(s) restored.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main(sys.argv[1])
    finally:
        Path(sys.argv[1]).unlink(missing_ok=True)
    sys.exit(code)
