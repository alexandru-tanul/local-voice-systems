"""Prepare a dashboard dataset for RVC (Applio) voice conversion training.

Reads datasets/<id>/<id>.list, converts every clip within the duration limits
to mono 16-bit PCM at the requested sample rate, writes them under
voicechanger/rvc_dataset/<name>/ together with a CSV manifest, and optionally
copies the folder into Applio's assets/datasets/<name>/ directory.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
VOICECHANGER_DIR = ROOT / "voicechanger"


def slugify(value, fallback="dataset"):
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return (slug or fallback)[:64]


def rvc_dataset_name(dataset_id):
    return slugify(dataset_id, "voice").replace("-", "") or "voice"


def wav_duration(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def load_rows(dataset_id):
    root = DATASETS_DIR / dataset_id
    list_path = root / f"{dataset_id}.list"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing dataset list: {list_path}")
    rows = []
    for raw in list_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split("|", 3)
        if len(parts) != 4:
            continue
        rel_path, speaker, lang, text = parts
        src = root / rel_path.replace("/", "\\")
        if src.exists():
            rows.append({"src": src, "speaker": speaker, "lang": lang, "text": text})
    return rows


def convert_wav(src, dst, ffmpeg, sample_rate):
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-ac", "1", "-ar", str(sample_rate), "-sample_fmt", "s16", str(dst),
    ]
    subprocess.run(command, check=True)


def prepare_dataset(dataset_id, sample_rate, min_seconds, max_seconds, applio_root=None):
    dataset_id = slugify(dataset_id)
    name = rvc_dataset_name(dataset_id)
    rows = load_rows(dataset_id)
    dataset_dir = VOICECHANGER_DIR / "rvc_dataset" / name
    manifest_path = VOICECHANGER_DIR / f"{name}_rvc_manifest.csv"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH.")

    manifest = []
    skipped = []
    for row in rows:
        src = row["src"]
        try:
            duration = wav_duration(src)
        except Exception as exc:
            skipped.append((src.name, f"unreadable: {exc}"))
            continue
        if duration < min_seconds:
            skipped.append((src.name, f"too short: {duration:.2f}s"))
            continue
        if duration > max_seconds:
            skipped.append((src.name, f"too long: {duration:.2f}s"))
            continue
        dst = dataset_dir / src.name
        convert_wav(src, dst, ffmpeg, sample_rate)
        manifest.append(
            {
                "file": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "source_file": str(src.relative_to(ROOT)).replace("\\", "/"),
                "duration_seconds": f"{duration:.3f}",
                "speaker": row["speaker"],
                "language": row["lang"],
                "text": row["text"],
            }
        )

    VOICECHANGER_DIR.mkdir(exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["file", "source_file", "duration_seconds", "speaker", "language", "text"])
        writer.writeheader()
        writer.writerows(manifest)

    copied_to_applio = None
    if applio_root:
        target = Path(applio_root) / "assets" / "datasets" / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(dataset_dir, target)
        copied_to_applio = target

    total_seconds = sum(float(row["duration_seconds"]) for row in manifest)
    return {
        "kept": len(manifest),
        "skipped": skipped,
        "total_seconds": total_seconds,
        "dataset_dir": dataset_dir,
        "manifest": manifest_path,
        "copied_to_applio": copied_to_applio,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare a dashboard dataset for RVC/Applio voice conversion training.")
    parser.add_argument("--dataset", required=True, help="Dataset id, the folder name under datasets/.")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--min-seconds", type=float, default=0.45)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--applio-root", help="Optional Applio folder; copies the dataset into assets/datasets/<name>.")
    args = parser.parse_args()

    try:
        result = prepare_dataset(args.dataset, args.sample_rate, args.min_seconds, args.max_seconds, args.applio_root)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    minutes = result["total_seconds"] / 60.0
    print(f"Prepared {result['kept']} clips, {minutes:.2f} minutes total.")
    print(f"Dataset: {result['dataset_dir']}")
    print(f"Manifest: {result['manifest']}")
    if result["copied_to_applio"]:
        print(f"Copied to Applio: {result['copied_to_applio']}")
    if result["skipped"]:
        print(f"Skipped {len(result['skipped'])} clips:")
        for name, reason in result["skipped"][:30]:
            print(f"  {name}: {reason}")
        if len(result["skipped"]) > 30:
            print(f"  ... plus {len(result['skipped']) - 30} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
