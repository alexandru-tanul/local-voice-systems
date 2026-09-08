# Live voice changer (experimental)

The dashboard's TTS pages turn typed or spoken text into a trained voice. The
Voice Changer page is different: it converts your microphone in real time, so
your words and timing stay yours and only the voice colour changes. It uses an
RVC model trained and served by [Applio](https://github.com/IAHispano/Applio).

## Folder contents

```text
voicechanger/rvc_dataset/<name>/     clips converted for RVC training (ignored by git)
voicechanger/<name>_rvc_manifest.csv what was kept, with durations (ignored by git)
```

`<name>` is the dataset id without hyphens, so the dataset `station-announcer`
becomes `stationannouncer`, and the default RVC model name is
`stationannouncer_rvc_32k`.

## Workflow

1. Build a dataset on the TTS Training page, or with `import_dataset.py`.
2. On the Voice Changer page, choose the source dataset and press
   **Prepare RVC Dataset**. The clips are converted and copied into Applio's
   `assets/datasets/<name>/` folder when Applio is installed.
3. Run Applio's preprocessing and feature extraction for the model name shown
   on the page, then press **Start RVC Training** and later **Build Index**.
4. Press **Start Live Engine**, choose the microphone, output, model, and
   index, and press **Start Live Voice**.

Route the output into a virtual microphone such as VB-CABLE to use the voice
in other applications.

## Starting points

- Model version RVC v2, F0 enabled, pitch extraction RMVPE, 32 kHz.
- Batch size 1 or 2 on consumer GPUs, 50 to 150 epochs, then listen.
- For real time, start with a larger chunk size for stability and lower it
  for latency once it runs cleanly. Use headphones to avoid feedback.
