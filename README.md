# Local Voice Systems

Speak or type, then hear the words in a reference voice. Everything runs locally after model downloads.
The dashboard can also train GPT-SoVITS voices and run an optional Applio voice changer.

```text
Microphone -> faster-whisper (Windows CPU) -> text
Text + reference clip -> GPT-SoVITS (WSL2 GPU or CPU) -> WAV / audio output
Microphone -> Applio (optional, Windows) -> live voice conversion
```

## Machines

The full app needs **64-bit Windows 10/11 with WSL2**. Windows 11 is the preferred setup.
Launchers, global push-to-talk, and file access use Windows APIs. Native Linux, macOS, and Windows ARM are not supported by this dashboard.

| Hardware | GPT-SoVITS setup | Automatic precision |
| --- | --- | --- |
| NVIDIA RTX 20/30/40/50 series | CUDA 12.8 | FP16 |
| NVIDIA GTX 16 series | CUDA 12.8 | FP32 |
| NVIDIA GTX 10 series, supported Pascal/Volta cards | CUDA 12.6 | FP32 on Pascal, FP16 on Volta |
| Older NVIDIA, Intel graphics, or no GPU | CPU | FP32 |
| AMD with a working WSL ROCm PyTorch install | Manual ROCm setup | FP16 |

GPU support also depends on the driver, PyTorch build, and available VRAM.
[NVIDIA supports WSL GPU compute on Pascal and newer cards in WDDM mode](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). Maxwell and older cards use the CPU route here.

The dashboard runs a small GPU calculation before choosing CUDA. `auto` falls back to CPU if that fails and shows the reason in **Setup**.
CPU use is slow, especially training. GPU memory can still run out when a full model loads.

Allow several GB of downloads. Plan for 16 GB RAM and 30 GB free disk as starting headroom, not measured minimums.
CPU installation and WAV synthesis were checked in Ubuntu 24.04. No Windows GPU run has been validated for this fork.

## Install

1. Install [64-bit Python 3.12](https://www.python.org/downloads/windows/), [Git](https://git-scm.com/downloads/win), and [FFmpeg](https://ffmpeg.org/download.html).
   Add Python and FFmpeg to the Windows PATH. Use Chrome, Edge, or Brave for audio output selection.
2. For NVIDIA, install the current [Windows driver](https://www.nvidia.com/Download/index.aspx).
   WSL uses that driver. Do not install a Linux NVIDIA driver inside WSL.
3. In an administrator PowerShell, install WSL2. Restart if prompted, then open Ubuntu once to create your Linux user.

   ```text
   wsl --install -d Ubuntu-24.04
   wsl --update
   wsl -l -v
   ```

   Ubuntu must show version `2`. For NVIDIA, confirm this command lists your card:

   ```text
   wsl -d Ubuntu-24.04 -- /usr/lib/wsl/lib/nvidia-smi
   ```

4. In a normal PowerShell, clone this fork and install the voice engine. Run this as your normal WSL user.

   ```text
   git clone https://github.com/scriptogre/local-voice-systems.git
   cd local-voice-systems
   Copy-Item config.example.json config.json
   wsl -d Ubuntu-24.04 -- bash ./setup_voice_engine.sh
   ```

   The installer asks for your Ubuntu password for system packages. It installs Miniforge, a separate Python 3.10 environment, a pinned GPT-SoVITS checkout, matching PyTorch packages, and pretrained models.
   It chooses CUDA 12.8, CUDA 12.6, or CPU from the detected NVIDIA card. AMD users get CPU unless they supply their own ROCm environment.

   To choose explicitly, add `--device cu128`, `--device cu126`, or `--device cpu`. Use `--help` for paths, model mirrors, and GPU selection.

5. Open `config.json`. Copy the paths printed by the installer into the matching keys.
   Keep `tts_device` and `tts_precision` set to `auto`. Set `wsl_distro` to the distribution shown by `wsl -l -v`.
6. Run `Setup-Relay-Env.bat`, then `Launch-Voice-Dashboard.bat`.
   Open **Setup** and run the check. It shows the selected GPU and precision, or the CPU fallback reason.

`Stop-Voice-Dashboard.bat` stops the dashboard and its services.
The first speech recognition request downloads the English Whisper model. Speech recognition uses CPU, leaving GPU memory for the voice engine.

## Try it

1. On **TTS Training**, create a dataset. Add a clear recording of 3 to 10 seconds and its exact transcript.
2. Press **Sync & Prepare** to copy it into WSL and build the features.
3. On **Generate**, select the dataset, reference clip, and pretrained `s1v3.ckpt` / `s2Gv2Pro.pth` models. Enter text and generate a WAV.

Training is optional. To fine-tune a voice, prepare more transcribed clips, choose a model name, then train SoVITS and GPT.
Start with batch size 1. Keep generated audio in `outputs/` and your recordings in `datasets/`.

For a folder of clips and a CSV with columns `file,text`:

```text
relay_env\Scripts\python.exe import_dataset.py --name "Station Announcer" --audio-dir C:\clips --transcripts C:\clips\lines.csv
```

On **Relay**, press **Start System**, hold push-to-talk, speak, then release.
Choose the voice, reference, models, and output device in **Settings**. [VB-CABLE](https://vb-audio.com/Cable/) can route output into another app.

Global push-to-talk uses Windows input hooks. If a game runs as administrator, the dashboard may need the same privilege level.
For direct microphone conversion, follow the separate [Applio setup](voicechanger/README.md).

## Fixes

| Problem | Action |
| --- | --- |
| Setup reports CPU on an NVIDIA machine | Check `nvidia-smi` in WSL, update the Windows driver, then rerun the installer with the correct `--device`. Run Setup again. |
| `no kernel image` or an unsupported GPU warning | Use `cu126` for Pascal/Volta or `cu128` for RTX 50. Set `tts_device` to `cuda` to make detection failures stop with an error. |
| NaNs, noise, or failed half precision | Set `tts_precision` to `fp32`, then restart the dashboard and engine. GTX 10/16 cards select FP32 automatically. |
| CUDA out of memory | Stop Applio and other GPU jobs. Use training batch size 1, shorter clips, or `tts_device: "cpu"`. |
| Missing WSL folder or Python | Use the exact paths printed by the installer. `/home/YOUR_WSL_USER` is a placeholder. Keep all paths under the same WSL user. |
| Missing training features | Run **Sync & Prepare**. v2Pro needs text, HuBERT, speaker, and semantic features. |
| Audio upload fails | Run `ffmpeg -version` in Windows PowerShell. Restart the dashboard after fixing PATH. |
| Existing relay environment uses Python 3.11 | Rename `relay_env`, install Python 3.12, and rerun `Setup-Relay-Env.bat`. |

## Config

Copy `config.example.json` to `config.json`. Restart the dashboard after edits.
Keep this file private to your machine. It is ignored by Git.

| Key | Value |
| --- | --- |
| `dashboard_python` | Launcher override. Empty: `relay_env`, `.venv`, then `python` or `py`. |
| `wsl_distro` | Distribution name, such as `Ubuntu-24.04`. |
| `wsl_gpt_sovits_root` | Absolute Linux path to GPT-SoVITS. |
| `wsl_python` | Its Python executable. The installer prints this path. |
| `wsl_datasets_root` | Linux folder for synced datasets. |
| `tts_device` | `auto`, `cuda` (GPU required, including ROCm), or `cpu`. |
| `tts_precision` | `auto`, `fp16`, or `fp32`. CPU requires FP32. |
| `tts_gpu_index` | GPU number, starting at `0`. Also pass `--gpu-index` to the installer when selecting another card. |
| `asr_model_dir` | Empty: `cache_dir/faster-whisper`. |
| `cache_dir` | Empty: `.cache` in this repository. |
| `tmp_dir` | Empty: `cache_dir/tmp`. |
| `applio_root` | Empty: an `Applio` folder beside this repository. |
| `applio_python` | Optional executable override. Auto-detects `env/python.exe` and `env/Scripts/python.exe`. |
| `applio_backend` | `auto` uses installed PyTorch, `cpu` hides GPUs, `zluda` uses Applio's ZLUDA wrapper. |
| `amd_hip_bin` | Optional AMD runtime folder added to Applio's PATH. Leave empty for NVIDIA/CPU. |

## Existing engines

You can keep an existing GPT-SoVITS install: set its paths in `config.json` and run **Setup**.
It needs v2Pro pretrained models, including the speaker checkpoint under `GPT_SoVITS/pretrained_models/sv/`.

The installer targets [GPT-SoVITS commit 48b1a01](https://github.com/RVC-Boss/GPT-SoVITS/tree/48b1a0169a28582a8984402f82cf438d3bfa6aca), PyTorch/torchaudio 2.11.0, and FFmpeg 6 to 8.
It refuses to replace another checkout. Use `--root /home/YOUR_WSL_USER/GPT-SoVITS-dashboard --env /home/YOUR_WSL_USER/gsv-dashboard` for a separate install.

For AMD acceleration, use [AMD's WSL compatibility guidance](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html) and [GPT-SoVITS's ROCm setup](https://github.com/RVC-Boss/GPT-SoVITS).
The dashboard recognizes ROCm through PyTorch's CUDA API. The installer does not set up AMD drivers.

## Checks

Run the dependency-free regression suite with Python 3.12:

```text
python -m unittest discover -s tests -v
```

GitHub Actions runs it on Windows and Linux. These checks do not load full voice models or prove GPU performance.

## Credits

[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and [Applio](https://github.com/IAHispano/Applio) provide the engines.
[Upstream dashboard](https://github.com/alexandru-tanul/local-voice-systems). [MIT license](LICENSE). Use recordings you have permission to use.
