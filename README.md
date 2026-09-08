# Local Voice Systems

A local, browser-based dashboard for building a cloned voice and talking
through it. Record or type a line, have it transcribed on your own machine,
and hear it spoken in a voice you trained from your own clips. Everything runs
locally: speech recognition with faster-whisper, speech synthesis with
GPT-SoVITS, and an optional real-time voice changer through Applio.

The project ships no voice data. You bring your own clips and train the voice
yourself.

## What it does

- **Relay.** Hold a key or the big microphone button, speak, and the line is
  transcribed and spoken in the trained voice. Works while the page is not
  focused, so it can feed a game or a call through a virtual microphone.
- **Generate.** Type a line, choose a reference clip, and produce a WAV file.
- **TTS Training.** Create a dataset from audio files with transcripts,
  prepare its features, and train GPT and SoVITS models with names of your
  choosing.
- **Voice Changer.** Experimental real-time voice conversion with an RVC model
  trained from the same dataset.
- **Logs** and a read-only **Setup** check for the whole toolchain.

## Requirements

- Windows 10 or 11. The global push-to-talk helper uses Windows input hooks.
- Python 3.11 or newer for the dashboard.
- WSL2 with a Linux distribution that has [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
  installed together with its v2Pro pretrained models and a PyTorch build for
  your GPU. The dashboard runs training and synthesis inside WSL.
- ffmpeg on the Windows PATH, used to convert uploaded audio.
- Chrome, Edge, or Brave. Output device selection needs a Chromium browser.
- Optional: [VB-CABLE](https://vb-audio.com/Cable/) to route the generated
  voice into other applications, and [Applio](https://github.com/IAHispano/Applio)
  for the voice changer.

## Setup

1. Clone the repository and copy `config.example.json` to `config.json`.
   Fill in the WSL distribution name, the GPT-SoVITS folder and its Python
   interpreter inside WSL, and, if you use them, the Applio folder and the AMD
   HIP folder. Empty values fall back to the defaults shown in the example.
2. Run `Setup-Relay-Env.bat`. It creates `relay_env` and installs
   faster-whisper for speech recognition. The Whisper model downloads on
   first use.
3. Run `Launch-Voice-Dashboard.bat`. It finds Python, starts the dashboard in
   the background, and opens http://localhost:8790. `Stop-Voice-Dashboard.bat`
   stops it together with the services it started.
4. Open **Setup** in the dashboard and run the system check. It reports what
   is missing before you start anything.

## Train a voice

1. On **TTS Training**, create a dataset and add audio files. Each file gets
   an editable transcript; the file name is used as a starting suggestion.
   For large folders, use the command line instead:

   ```text
   python import_dataset.py --name "Station Announcer" --audio-dir C:\clips --transcripts C:\clips\lines.csv
   ```

   The CSV needs the columns `file,text`. Without a CSV, transcripts are
   derived from file names, so check them by ear.
2. Enter a model name and press **Sync & Prepare**. The dataset is copied into
   WSL and the text, HuBERT, and semantic features are extracted on the GPU.
3. Train SoVITS, then GPT. Checkpoints appear under the model name in the
   GPT-SoVITS weights folders and are listed at the bottom of the page.

Reference clips for synthesis come from the dataset you choose on the Generate
page or in the Relay settings, so a dataset has to be synced into WSL before
it can be used for synthesis.

## Use the voice

- **Relay.** Press **Start System** in the control island to load speech
  recognition, the voice engine, and the global push-to-talk helper. Then hold
  the microphone button or the push-to-talk key, speak, and release. The
  status under the button moves through Listening, Transcribing, Generating
  voice, and Playing. Each line is listed in the sidebar for the session.
  Without Start System, services start on first use, but the push-to-talk
  key only works while the page is focused.
- **Settings** in the island holds the output device, volume, voice dataset,
  models, reference clip, extra reference clips, and generation settings.
- **Generate** produces a WAV file from typed text and starts the voice engine
  on its own when it is off. Files are written to `outputs/`.

## Global push-to-talk

`ptt_helper.py` installs Windows keyboard and mouse hooks and forwards only the
configured key's pressed and released state to the dashboard on `127.0.0.1`.
It does not record text and does not talk to anything outside the machine.
Some games run elevated and only expose input to elevated applications; run
the launcher at the same privilege level if the key does not work in one game.

## Configuration keys

| Key | Meaning |
| --- | --- |
| `dashboard_python` | Python used by the launcher. Empty: `.venv`, then `python` or `py` on PATH. |
| `wsl_distro` | WSL distribution that has GPT-SoVITS. |
| `wsl_gpt_sovits_root` | GPT-SoVITS folder inside WSL. |
| `wsl_python` | Python interpreter inside WSL with GPT-SoVITS dependencies. |
| `wsl_datasets_root` | Where datasets are copied inside WSL. |
| `asr_model_dir` | Folder for the faster-whisper model. Empty: `cache_dir/faster-whisper`. |
| `cache_dir` | Cache folder for Hugging Face, pip, and RVC kernels. Empty: `.cache` in the project. |
| `tmp_dir` | Temporary folder for the services. Empty: `cache_dir/tmp`. |
| `applio_root` | Applio installation for the voice changer. |
| `amd_hip_bin` | AMD HIP runtime folder, needed by Applio on AMD GPUs. |

## Layout

```text
voice_dashboard.py               the dashboard: HTTP server, job control, and the web page
relay_asr_server.py              local speech recognition service (faster-whisper)
ptt_helper.py                    global push-to-talk hook for Windows
import_dataset.py                command-line dataset importer
prepare_voicechanger_dataset.py  converts a dataset for RVC training
rvc_monitor.py                   small status page for an RVC training run
datasets/                        your datasets (ignored by git)
outputs/                         generated audio (ignored by git)
voicechanger/                    RVC dataset folder and notes
```

## Credits

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) for few-shot voice
  synthesis and training.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for local speech
  recognition.
- [Applio](https://github.com/IAHispano/Applio) for RVC training and real-time
  conversion.

## License

MIT. See `LICENSE`. Voice data you train from remains subject to its own
rights; only use recordings you are allowed to use.
