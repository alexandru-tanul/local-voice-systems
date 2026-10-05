# Voice changer

Applio converts your microphone audio into an RVC voice in real time.
It runs separately from the GPT-SoVITS engine. Its Python and GPU packages are separate too.

## Install

1. Clone [Applio](https://github.com/IAHispano/Applio) beside this repository, then run its Windows installer:

   ```text
   cd ..
   git clone https://github.com/IAHispano/Applio.git
   cd Applio
   .\run-install.bat
   ```

2. Set `applio_root` in this dashboard's `config.json` if you chose another folder.
   The dashboard finds `env/python.exe` (Conda) or `env/Scripts/python.exe` (venv). Use `applio_python` for another layout.
3. Choose the route below, then restart the dashboard. Press **Start Live Engine** on **Voice Changer**.

## NVIDIA

Leave `applio_backend` as `auto` and `amd_hip_bin` empty. No AMD SDK is needed.
Applio's CUDA build drives the GPU. Its standard CUDA 12.8 install targets GTX 16 and RTX cards, including RTX 50.

For Pascal/Volta cards, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and replace PyTorch with matching CUDA 12.6 builds from the Applio folder:

```text
uv pip install --python .\env\python.exe "torch==2.11.0+cu126" "torchaudio==2.11.0+cu126" --index-url https://download.pytorch.org/whl/cu126
```

Verify in PowerShell from the Applio folder:

```text
.\env\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

This must print `True` and your GPU name. If it prints `False`, fix the Windows driver and Applio's PyTorch install first.
This dashboard's `tts_precision` setting applies only to GPT-SoVITS.

## CPU

Set `applio_backend` to `cpu` to disable GPU use in Applio.
Conversion may be too slow for live use. GPT-SoVITS has its own `tts_device` setting.

## AMD

Follow [Applio's AMD instructions](https://docs.applio.org/getting-started/installation/) for a supported card and runtime.
For its ZLUDA install, set `applio_backend` to `zluda` and `amd_hip_bin` to your installed HIP `bin` folder.

The dashboard launches `zluda/zluda.exe -- <python>`. It reports a missing wrapper instead of claiming GPU readiness.
For a working native ROCm PyTorch install, use `auto`. AMD support depends on the chosen Applio build and drivers.

## Use

1. Create a transcribed dataset on **TTS Training**.
2. On **Voice Changer**, select it and press **Prepare RVC Dataset**. Clips are copied to Applio's `assets/datasets/<name>/` folder.
3. In Applio, preprocess and extract features for the model name shown in the dashboard. Use RVC v2, F0 enabled, and 32 kHz.
4. Press **Start RVC Training** with batch size 1, then **Build Index** after training.
5. Press **Start Live Engine**, select the microphone, output, model, and index, then **Start Live Voice**.

Use headphones. Route output through [VB-CABLE](https://vb-audio.com/Cable/) to send it to another app.
Stop the GPT-SoVITS engine while using Applio if GPU memory is tight.

Converted clips live in `voicechanger/rvc_dataset/<name>/`; their manifest is `voicechanger/<name>_rvc_manifest.csv`.
Both are ignored by Git. No Windows GPU run has been validated for this fork.
