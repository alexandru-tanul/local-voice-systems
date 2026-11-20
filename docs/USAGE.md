# Using the dashboard

## Voice Training

Create a dataset and press **+ Add audio**. Speech recognition writes each transcript; check them before adding.
Recordings over 20 seconds are cut into clips at pauses. The clip list plays, corrects and deletes clips.
**Check clips** finds clips with a wrong transcript, background noise, distortion, the same audio twice, or a length that training skips, and **Use heard text** puts in what speech recognition hears.
Press **Sync Only**. The other tabs read clips from the copy in WSL, so sync again after adding clips.

Datasets can be English, Chinese, Japanese, Korean, or Cantonese.
To train a voice, add more clips, press **Sync & Prepare**, then train SoVITS and GPT with batch size 1. Each shows its progress by epoch.
SoVITS can be v2Pro or v2ProPlus, GPT-SoVITS's larger model, which sounds better and needs more GPU memory. Both use the same preparation and GPT model.
**Compare epochs** then says one test line with each saved epoch; **Use this** gives the best one to Text to Voice and Mic to Voice.

To import a folder of clips at once, with transcripts from a CSV with the columns `file,text`:

```text
python import_dataset.py --name "My Voice" --audio-dir C:\clips --transcripts C:\clips\lines.csv
```

Without `--transcripts`, each transcript comes from the file name. Add `--language` for a dataset that is not English.

## Restore audio

**Restore audio**, a tab on **Voice Training**, cleans up recordings before training. In this order, it can remove music and sound effects, echo and reverb, and background noise, and restore missing high frequencies.
Music and echo removal use GPT-SoVITS's UVR5 models in WSL, on the GPU unless `tts_device` is `cpu`; the rest uses [LavaSR](https://github.com/ysharma3501/LavaSR) on the CPU and needs `Setup-Restore-Env.bat` once.
Each result is added to the list, so steps can be combined, compared, and added to a dataset, where long recordings are cut into clips.

## Text to Voice

Choose the voice and reference clip, enter text, and press **Generate**.
**Settings** holds the models, such as the pretrained `s1v3.ckpt` / `s2Gv2Pro.pth`, and **Text language**, which follows the dataset's language.

**Export voice**, in **Settings**, saves both models and the reference clips as one file. **Import a voice** installs such a file as a new voice.

## Mic to Voice

Press **Start**, then hold push-to-talk, speak, and release. Choose audio output in **Settings**.
**Start** turns on speech recognition, the voice engine, and the global push-to-talk key, and **Stop Everything** stops them.
**Sync & Prepare** and training stop them first to free GPU memory.

**Settings** sets the language you speak and the language the voice speaks: English, Chinese, Japanese, Korean or Cantonese, both English at first.
When they differ, your words are translated by [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) on the CPU. **Its license, CC-BY-NC 4.0, allows only non-commercial use.**
Chinese and Cantonese need no translation between them: both are written in Chinese characters, which the voice reads in either.
A voice sounds best in the language of its own clips.

Use [VB-CABLE](https://vb-audio.com/Cable/) to send the generated voice to another app.

## Output

Generated audio is saved in `outputs`. Mic to Voice keeps only its newest 200 lines, in `outputs\relay`.

## Use in other programs

While the dashboard runs, programs that work with OpenAI's speech service can speak with your voices at `http://127.0.0.1:8790/v1`.
A voice is offered there once it has spoken a line on Text to Voice, and it speaks with that line's models, reference clip and settings. An imported voice is offered at once.

- Open WebUI: in its audio settings, choose the OpenAI text-to-speech engine, enter `http://127.0.0.1:8790/v1` as the API base URL and any key, and pick a voice from its list.
- SillyTavern: choose the OpenAI Compatible TTS provider, enter `http://127.0.0.1:8790/v1/audio/speech` as the provider endpoint, and list voice ids, such as `my-voice`, as the available voices.
- Other programs: `POST /v1/audio/speech` with `input` and `voice`, and optionally `response_format` (`mp3`, the default, `opus`, `aac`, `flac`, `wav` or `pcm`) and `speed`. `GET /v1/audio/voices` lists the voice ids.

With `api_key` in `config.json`, a program must send it as `Authorization: Bearer <key>` to speak; the voice list stays open, as Open WebUI asks for it without the key.
