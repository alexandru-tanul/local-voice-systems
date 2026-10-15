"""Cutting long recordings into clips at their pauses, each with a transcript for the page to check."""

import os
import shutil
import wave
from pathlib import Path

from . import asr, config, datasets, jobs, restore


# Long recordings are cut into clips at pauses. ffmpeg measures the loudness of every 20 ms, so
# the dashboard needs no audio libraries; each clip is transcribed and added to the dataset. The
# page sends recordings longer than 20 seconds here.
CUT_STEP = 0.02
CUT_DIR = config.OUTPUT_DIR / "cut"


def frame_levels(job, wav_path):
    """The loudness of each 20 ms of a recording, in dB; silence is -inf."""
    result = job.run(
        [
            "ffmpeg", "-v", "error", "-i", str(wav_path), "-af",
            "aresample=16000,asetnsamples=n=320:p=0,"
            "astats=metadata=1:reset=1:measure_overall=RMS_level:measure_perchannel=none,"
            "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null", "-",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg could not measure the recording.")
    levels = []
    for line in result.stdout.splitlines():
        if line.startswith("lavfi.astats.Overall.RMS_level="):
            try:
                levels.append(float(line.split("=", 1)[1]))
            except ValueError:
                levels.append(float("-inf"))
    return levels


def plan_clips(levels, step=CUT_STEP, min_seconds=3.0, max_seconds=10.0):
    """Where to cut a recording, as (start, end) seconds, from the loudness of each step.

    Speech is what is louder than the quiet parts by a third of the way to the loud parts, so the
    level of the recording and of its background does not matter. The loud parts are measured
    among what stands out from the quiet ones, so a recording that is mostly silence still has its
    speech as its loud part; one where nothing stands out has no speech to cut. A clip ends at the
    first pause of 0.3 s once it is min_seconds long, always at a pause of a second, and before it
    would grow past max_seconds; speech that goes on longer is cut at its quietest moment. Each
    clip keeps up to 0.2 s of the pause on both sides."""
    # Digital silence measures -inf; it counts as -120 dB, so it is the quiet part of a recording.
    heard = sorted(max(level, -120.0) for level in levels)
    if not heard or heard[-1] <= -120:
        return []
    quiet = heard[len(heard) // 20]
    standing_out = [level for level in heard if level > quiet + 6]
    if not standing_out:
        return []
    loud = standing_out[len(standing_out) * 9 // 10]
    threshold = quiet + (loud - quiet) / 3
    # Runs of speech, joined across dips shorter than 0.1 s, such as the stop in "kept".
    runs = []
    for index, level in enumerate(levels):
        if level <= threshold:
            continue
        if runs and (index - runs[-1][1]) * step < 0.1:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1])
    if not runs:
        return []
    clips = []
    start, end = runs[0]
    for run_start, run_end in runs[1:]:
        pause = (run_start - end) * step
        if pause >= 1.0 or (run_end - start) * step > max_seconds or ((end - start) * step >= min_seconds and pause >= 0.3):
            clips.append((start, end))
            start = run_start
        end = run_end
    clips.append((start, end))

    def split(start, end):
        if (end - start) * step <= max_seconds:
            return [(start, end)]
        margin = int(min_seconds / step)
        middle = range(start + margin, end - margin)
        cut = min(middle, key=lambda index: levels[index]) if len(middle) else (start + end) // 2
        return split(start, cut) + split(cut, end)

    clips = [piece for start, end in clips for piece in split(start, end)]
    total = len(levels) * step
    padded = []
    for index, (start, end) in enumerate(clips):
        before = (start - clips[index - 1][1]) * step / 2 if index else 0.2
        after = (clips[index + 1][0] - end) * step / 2 if index + 1 < len(clips) else 0.2
        begin = max(0.0, start * step - min(0.2, before))
        finish = min(total, end * step + min(0.2, after))
        # Training skips clips of 0.6 s or less; shorter sounds are clicks and breaths.
        if finish - begin > 0.7:
            padded.append((round(begin, 3), round(finish, 3)))
    return padded


def write_wav_part(source, target, start, end):
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        reader.setpos(min(reader.getnframes(), int(start * rate)))
        frames = reader.readframes(int((end - start) * rate))
        with wave.open(str(target), "wb") as writer:
            writer.setparams(reader.getparams())
            writer.writeframes(frames)


def cut_recordings(job, dataset_id):
    """The work of the dataset-cut job: each recording in job.sources, in turn."""
    try:
        cut_each_recording(job, datasets.dataset_descriptor(dataset_id))
    finally:
        # Recordings it did not reach, because it failed or was stopped.
        with jobs.STATE_LOCK:
            job.accepting = False
            left, job.sources = job.sources, []
        for source, _ in left:
            source.unlink(missing_ok=True)


def cut_each_recording(job, descriptor):
    added = skipped = 0
    failed = []
    while True:
        with jobs.STATE_LOCK:
            if not job.sources:
                # From here on start_dataset_cut starts a new job instead of adding to this one.
                job.accepting = False
                break
            source, name = job.sources.pop(0)
        # One recording that cannot be read does not stop the ones after it.
        try:
            made, left_out = cut_recording(job, descriptor, source, name)
            added += made
            skipped += left_out
        except jobs.JobStopped:
            raise
        except Exception as exc:
            failed.append(name)
            job.say(f"Could not cut {name}: {exc}")
        finally:
            source.unlink(missing_ok=True)
            source.with_name(source.stem + "-44k.wav").unlink(missing_ok=True)
    job.say(f"Done. {added} clip{'' if added == 1 else 's'} to check in the list below."
            + (f" {skipped} had no words and were left out." if skipped else ""))
    if failed:
        raise RuntimeError(f"{len(failed)} recording{'' if len(failed) == 1 else 's'} could not be cut: {', '.join(failed)}.")


def cut_recording(job, descriptor, source, name):
    """Cut one recording into clips and transcribe them. The clips with words go to job.results,
    from where the page puts them in the list of files to check before they are added.
    Returns the clips made and those left out."""
    job.check_stop()
    stem = datasets.slugify(Path(name).stem, "clip")[:48]
    converted = source.with_name(source.stem + "-44k.wav")
    job.say(f"Converting {name}...")
    result = job.run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-ar", "44100", "-ac", "1", "-sample_fmt", "s16", str(converted)])
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg could not read it: {result.stderr.strip()}")
    job.say(f"Finding the pauses in {name}...")
    clips = plan_clips(frame_levels(job, converted))
    if not clips:
        job.say(f"Found no speech in {name}.")
        return 0, 0
    job.say(f"Cut {name} into {len(clips)} clips. Writing their transcripts...")
    added = skipped = 0
    CUT_DIR.mkdir(parents=True, exist_ok=True)
    for index, (start, end) in enumerate(clips, 1):
        job.check_stop()
        clip = f"{stem}-{index:03d}.wav"
        part = CUT_DIR / f"{os.urandom(4).hex()}-{clip}"
        write_wav_part(converted, part, start, end)
        text = asr.transcribe_dataset_wav(part, descriptor["language"])
        if not text:
            part.unlink(missing_ok=True)
            skipped += 1
            continue
        with jobs.STATE_LOCK:
            job.results.append({"name": clip, "url": f"/outputs/cut/{part.name}", "text": text})
        added += 1
        job.say(f"Transcribed {index} of {len(clips)} from {name}: {text}")
    return added, skipped


def start_dataset_cut(dataset_id, sources):
    """Cut recordings, given as (temporary file, name) pairs, into clips of a dataset. A run for the
    same dataset takes them on; the files are deleted when they are done."""
    try:
        descriptor = datasets.dataset_descriptor(dataset_id)
    except Exception:
        for source, _ in sources:
            source.unlink(missing_ok=True)
        raise
    with jobs.STATE_LOCK:
        current = jobs.JOBS.get("dataset-cut")
        busy = bool(current and current.running() and current.accepting and not current.stopping)
        if busy and current.dataset_id == descriptor["id"]:
            current.sources.extend(sources)
        elif not busy:
            # The clips of an earlier run were taken by the page then, or are no longer wanted.
            for old in CUT_DIR.glob("*.wav"):
                old.unlink(missing_ok=True)
            job = jobs.ThreadJob(f"Cut recordings for {descriptor['name']}", lambda job: cut_recordings(job, descriptor["id"]))
            job.dataset_id = descriptor["id"]
            job.sources = list(sources)
            job.results = []
            job.accepting = True
            jobs.JOBS["dataset-cut"] = job
            job.start()
    if busy and current.dataset_id != descriptor["id"]:
        for source, _ in sources:
            source.unlink(missing_ok=True)
        raise RuntimeError("Another dataset's recordings are being cut. Wait until that ends.")
    if busy:
        current.say(f"{len(sources)} more recording{'' if len(sources) == 1 else 's'} to cut.")
    else:
        jobs.log(f"Cutting {len(sources)} recording(s) into clips for {descriptor['name']}.")


def cut_restored(dataset_id, item_ids):
    """Cut recordings from Restore audio into clips of a dataset."""
    known = {row["id"]: row for row in restore.restore_items()}
    sources = []
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    for item_id in item_ids or []:
        row = known.get(item_id)
        if not row:
            continue
        copy = config.TMP_DIR / f"upload-{os.urandom(4).hex()}.wav"
        shutil.copyfile(restore.RESTORE_DIR / row["file"], copy)
        sources.append((copy, f"{row['name']}.wav"))
    if not sources:
        raise RuntimeError("Those recordings no longer exist.")
    start_dataset_cut(dataset_id, sources)
