"""Import a folder of audio clips as a dashboard dataset from the command line.

The Training page can do the same through the browser; this script suits
large folders. Every clip is converted with ffmpeg to mono 16-bit PCM WAV at
44.1 kHz, which is what GPT-SoVITS expects, and registered in
datasets/<id>/dataset.json, <id>.list, and metadata.csv.

Transcripts come from a CSV with two columns, file and text, matched by file
name. Without a CSV, each transcript is derived from the file name, so
"please_remain_calm.wav" becomes "Please remain calm." Check those by ear.

Examples:
  python import_dataset.py --name "Station Announcer" --audio-dir C:\\clips --transcripts C:\\clips\\lines.csv
  python import_dataset.py --name "Station Announcer" --audio-dir C:\\clips --speaker announcer --language en
"""

import argparse
import base64
import csv
import sys
from pathlib import Path

import voice_dashboard as dashboard

AUDIO_SUFFIXES = {".wav", ".ogg", ".mp3", ".flac", ".m4a", ".aac", ".opus", ".wma"}


def suggested_transcript(filename):
    stem = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    stem = " ".join(stem.split())
    if not stem:
        return ""
    text = stem[0].upper() + stem[1:]
    return text if text[-1] in ".!?" else text + "."


def read_transcripts(path):
    rows = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or "file" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise SystemExit("The transcript CSV needs a header with the columns: file,text")
        for row in reader:
            rows[Path(row["file"]).name.lower()] = (row["text"] or "").strip()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Import audio clips as a dashboard dataset.")
    parser.add_argument("--name", required=True, help="Dataset name shown in the dashboard.")
    parser.add_argument("--audio-dir", required=True, help="Folder containing the audio clips.")
    parser.add_argument("--transcripts", help="CSV with columns file,text. Omitted: transcripts are derived from file names.")
    parser.add_argument("--speaker", default="speaker")
    parser.add_argument("--language", default="en", choices=["en", "zh", "ja", "ko", "yue"])
    parser.add_argument("--existing", action="store_true", help="Add to an existing dataset with this name instead of creating one.")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    files = sorted(path for path in audio_dir.iterdir() if path.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise SystemExit(f"No audio files found in {audio_dir}")
    transcripts = read_transcripts(args.transcripts) if args.transcripts else {}

    dataset_id = dashboard.slugify(args.name)
    if args.existing:
        dashboard.dataset_descriptor(dataset_id)
    else:
        dashboard.create_dataset(args.name, args.speaker, args.language)

    added = 0
    missing = []
    for path in files:
        text = transcripts.get(path.name.lower()) if transcripts else suggested_transcript(path.name)
        if not text:
            missing.append(path.name)
            continue
        dashboard.add_dataset_audio(
            dataset_id,
            path.name,
            base64.b64encode(path.read_bytes()).decode("ascii"),
            text,
            args.speaker,
            args.language,
        )
        added += 1
        print(f"added {path.name}: {text}")

    print(f"Imported {added} clip(s) into datasets/{dataset_id}.")
    if missing:
        print(f"Skipped {len(missing)} file(s) without a transcript:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
