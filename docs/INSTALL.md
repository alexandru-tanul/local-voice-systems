# Installing the engine

`Setup-Engine.bat` is the usual way to install the engine; see the [README](../README.md#install).
It finds your graphics card and asks you to confirm the build, installs GPT-SoVITS in Ubuntu with Miniforge, and writes `config.json`.
It uses 15 GB of disk space for CPU and about 50 GB for AMD with ROCm.

For NVIDIA, `/usr/lib/wsl/lib/nvidia-smi` inside Ubuntu must list your card ([WSL requirements](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)).

## An existing GPT-SoVITS

Setup-Engine.bat leaves a `~/GPT-SoVITS` that it did not install as it is.
To use an existing GPT-SoVITS v2Pro install, copy `config.example.json` to `config.json`, set its three WSL paths to that install, and choose the device and precision from the table in the README.

## Installing by hand

Install [Miniforge for Linux](https://github.com/conda-forge/miniforge#install) inside Ubuntu, reopen Ubuntu, and run the commands below with `build` from the table in the README.
The upstream installer downloads the models and dependencies. Then write `config.json` as above, with full paths: `/home/<user>/GPT-SoVITS`, `/home/<user>/miniforge3/envs/gsv/bin/python`, and a folder for datasets.

```text
sudo apt update
sudo apt install -y build-essential git wget
conda create -y -n gsv python=3.10 uv
conda activate gsv
git clone https://github.com/RVC-Boss/GPT-SoVITS.git ~/GPT-SoVITS
cd ~/GPT-SoVITS
git checkout 48b1a0169a28582a8984402f82cf438d3bfa6aca
build=cu128
uv pip install "torch==2.11.0+$build" "torchaudio==2.11.0+$build" torchcodec==0.11.1 --index-url "https://download.pytorch.org/whl/$build"
printf 'torch==2.11.0\ntorchaudio==2.11.0\ntorchcodec==0.11.1\n' > voice-constraints.txt
CONDA_PINNED_PACKAGES='ffmpeg>=6,<9' PIP_CONSTRAINT="$PWD/voice-constraints.txt" WORKFLOW=true bash install.sh --device "${build^^}" --source HF
```

These pins keep PyTorch, torchaudio, and FFmpeg compatible.
For AMD, `scripts/setup_engine.sh` shows the ROCm steps: AMD's ROCm 7.2 for WSL, and AMD's PyTorch 2.8.0 on Python 3.12.

`Setup-Engine.bat` was tested with the CPU and AMD builds, each in a fresh Ubuntu 24.04.
