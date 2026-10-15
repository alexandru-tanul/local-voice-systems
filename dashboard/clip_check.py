"""Check clips: finds clips with a wrong transcript, background noise, distortion, the same audio
twice, or a length that training skips."""

import array
import hashlib
import math
import re
import sys
import unicodedata
import wave

from . import asr, datasets, jobs


# Dataset check. Measured on clean game recordings, speech stood 20 dB or more above the quietest
# moments of each clip, and noise 10 dB below the speech brought that to about 17 dB.
CHECK_NOISE_DB = 16
# Less than this, and nothing stands out: a tone, a hum, or noise without speech.
CHECK_FLAT_DB = 6
CHECK_QUIET_DB = -30
# Places where three or more samples in a row are at full volume, where the sound was cut off.
CHECK_CLIPPED_PLACES = 10
# The share of a transcript's words that speech recognition hears differently.
CHECK_MISMATCH = 0.3


def clip_sound(path):
    """The loudest level of a 16-bit WAV clip in dB below full volume, how far its speech stands
    above its quietest moments, the places where it is clipped, and a digest of its samples.
    None when the file cannot be read."""
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getsampwidth() != 2:
                return None
            rate = audio.getframerate() * audio.getnchannels()
            frames = audio.readframes(audio.getnframes())
    except (OSError, EOFError, wave.Error):
        return None
    # A file cut short can end in half a sample.
    samples = array.array("h", frames[:len(frames) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return None
    step = max(1, int(rate * 0.02))
    levels = sorted(
        10 * math.log10(max(sum(x * x for x in samples[i:i + step]) / step, 1) / 32768 ** 2)
        for i in range(0, max(1, len(samples) - step + 1), step)
    )
    clipped = run = rail = 0
    for x in samples:
        side = 1 if x >= 32760 else (-1 if x <= -32760 else 0)
        if side and side == rail:
            run += 1
        else:
            clipped += run >= 3
            run, rail = (1, side) if side else (0, 0)
    clipped += run >= 3
    peak = max(max(samples), -min(samples))
    return {
        "peak_db": 20 * math.log10(max(peak, 1) / 32768),
        # The loud 5 % of a clip is speech, and its quietest 10 % the background between words.
        "speech_db": levels[int(len(levels) * 0.95)] - levels[int(len(levels) * 0.1)],
        "clipped": clipped,
        "digest": hashlib.sha1(samples.tobytes()).hexdigest(),
    }


def transcript_words(text, language):
    """A transcript as words to compare, without case or punctuation; Chinese, Japanese and
    Cantonese as characters, as they have no spaces between words, and Korean too, as its
    spacing varies."""
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    # 1,210 and 1210 are one number.
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = "".join(" " if unicodedata.category(char)[0] in "PS" else char for char in text)
    if language in {"zh", "ja", "yue", "ko"}:
        return [char for char in text if not char.isspace()]
    return text.split()


def transcript_difference(expected, heard, language):
    """The words that differ between two transcripts, as a share of the expected words."""
    expected, heard = transcript_words(expected, language), transcript_words(heard, language)
    if not expected:
        return 1.0 if heard else 0.0
    previous = list(range(len(heard) + 1))
    for i, word in enumerate(expected, 1):
        current = [i]
        for j, other in enumerate(heard, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != other)))
        previous = current
    return previous[-1] / len(expected)


def start_dataset_check(dataset_id):
    descriptor = datasets.dataset_descriptor(dataset_id)
    with jobs.STATE_LOCK:
        current = jobs.JOBS.get("dataset-check")
        if current and current.running():
            raise RuntimeError("A dataset check is already running.")
        job = jobs.ThreadJob(f"Check {descriptor['name']}", lambda job: check_dataset(job, descriptor["id"]))
        job.dataset_id = descriptor["id"]
        job.results = []
        jobs.JOBS["dataset-check"] = job
        job.start()
    jobs.log(f"Checking the clips of {descriptor['name']}.")


def check_dataset(job, dataset_id):
    """Find clips that can make a voice worse, and those that training skips. Each clip with
    problems goes to job.results with what was found, and, when its transcript differs, the words
    speech recognition heard."""
    descriptor = datasets.dataset_descriptor(dataset_id)
    clips = datasets.dataset_clips(dataset_id)["clips"]
    found = {}

    def flag(wav, problem, heard=None):
        with jobs.STATE_LOCK:
            if wav not in found:
                found[wav] = {"wav": wav, "problems": [], "heard": None}
                job.results.append(found[wav])
            found[wav]["problems"].append(problem)
            if heard is not None:
                found[wav]["heard"] = heard

    job.say(f"Checking the sound of {len(clips)} clips...")
    first = {}
    readable = []
    for clip in clips:
        job.check_stop()
        sound = clip_sound(descriptor["root"] / "wavs" / clip["wav"])
        if not sound:
            flag(clip["wav"], "The file cannot be read.")
            continue
        readable.append(clip)
        if clip["skipped"]:
            flag(clip["wav"], clip["skipped"])
        if sound["peak_db"] < CHECK_QUIET_DB:
            flag(clip["wav"], f"Very quiet: its loudest moment is {-round(sound['peak_db'])} dB below full volume.")
        elif sound["speech_db"] < CHECK_FLAT_DB:
            flag(clip["wav"], "No speech stands out: the sound is about as loud from start to end.")
        elif sound["speech_db"] < CHECK_NOISE_DB:
            flag(clip["wav"], f"Loud background: the speech is only {round(sound['speech_db'])} dB above it. Restore audio can remove noise and music.")
        if sound["clipped"] >= CHECK_CLIPPED_PLACES:
            flag(clip["wav"], f"Distorted: the sound is cut off at full volume in {sound['clipped']} places.")
        if sound["digest"] in first:
            flag(clip["wav"], f"The same audio as {first[sound['digest']]}.")
        else:
            first[sound["digest"]] = clip["wav"]
    language = descriptor["language"]
    skipped = ""
    if language == "yue" and "large-v3" not in asr.dataset_asr_model(language):
        # Whisper before large-v3 writes Cantonese as Mandarin, which would flag every transcript.
        skipped = " Cantonese transcripts are compared only with a large-v3 dataset_asr_model in config.json."
        readable = []
    else:
        job.say(f"Sound checked. Comparing transcripts with what speech recognition hears, 0 of {len(readable)}...")
    for index, clip in enumerate(readable, 1):
        job.check_stop()
        try:
            heard = asr.transcribe_dataset_wav(descriptor["root"] / "wavs" / clip["wav"], language)
        except FileNotFoundError:
            # Deleted while the check runs.
            continue
        if transcript_difference(clip["text"], heard, language) >= CHECK_MISMATCH:
            flag(clip["wav"], "The transcript differs from what speech recognition hears.", heard)
        if index % 10 == 0 or index == len(readable):
            job.say(f"Comparing transcripts with what speech recognition hears, {index} of {len(readable)}...")
    job.say((f"Done. {len(found)} of {len(clips)} clips need a look." if found else f"Done. No problems found in {len(clips)} clips.") + skipped)
