#!/usr/bin/env bash
# Run inside Ubuntu (WSL2), as the user who will run the dashboard.
set -eo pipefail

GSV_REPO=https://github.com/RVC-Boss/GPT-SoVITS.git
GSV_COMMIT=48b1a0169a28582a8984402f82cf438d3bfa6aca
device=auto
source_name=HF
gsv_root=$HOME/GPT-SoVITS
env_prefix=$HOME/gsv-venv
gpu_index=0

while [ $# -gt 0 ]; do
    case "$1" in
    --help|-h)
        echo 'Usage: bash setup_voice_engine.sh [--device auto|cu128|cu126|cpu] [--gpu-index 0]'
        echo '       [--source HF|HF-Mirror|ModelScope] [--root PATH] [--env PATH]'
        exit 0 ;;
    --device|--source|--root|--env|--gpu-index)
        if [ $# -lt 2 ] || [[ "$2" == --* ]]; then echo "Missing value for $1" >&2; exit 2; fi
        case "$1" in
        --device) device=$2 ;;
        --source) source_name=$2 ;;
        --root) gsv_root=$2 ;;
        --env) env_prefix=$2 ;;
        --gpu-index) gpu_index=$2 ;;
        esac
        shift 2 ;;
    *) echo "Unknown option: $1. Use --help." >&2; exit 2 ;;
    esac
done
case "$device" in auto|cu128|cu126|cpu) ;; *) echo 'Use auto, cu128, cu126, or cpu.' >&2; exit 2 ;; esac
case "$source_name" in HF|HF-Mirror|ModelScope) ;; *) echo 'Invalid model source.' >&2; exit 2 ;; esac
[[ "$gpu_index" =~ ^[0-9]+$ ]] || { echo 'GPU index must be zero or greater.' >&2; exit 2; }
[[ "$gsv_root" == /* && "$env_prefix" == /* ]] || { echo 'Use absolute paths for --root and --env.' >&2; exit 2; }
[ "$(uname -m)" = x86_64 ] || { echo 'This installer needs x86-64 Ubuntu.' >&2; exit 1; }

export PATH="$PATH:/usr/lib/wsl/lib"
if [ "$device" = auto ]; then
    if nvidia-smi -i "$gpu_index" -L >/dev/null 2>&1; then
        cap=$(nvidia-smi -i "$gpu_index" --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
        if [[ "$cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
            capability=$(( ${cap%%.*} * 10 + ${cap#*.} ))
            if [ "$capability" -ge 75 ]; then device=cu128
            elif [ "$capability" -ge 60 ]; then device=cu126
            else device=cpu; fi
        else
            echo 'Cannot detect GPU generation. Use --device cu126 or --device cu128.' >&2
            exit 1
        fi
    else
        device=cpu
    fi
fi
echo "Installing GPT-SoVITS with PyTorch $device. See README.md for GPU limits."

sudo_cmd=()
if [ "$(id -u)" != 0 ]; then sudo_cmd=(sudo); fi
"${sudo_cmd[@]}" apt-get update -q
"${sudo_cmd[@]}" apt-get install -y -q build-essential ca-certificates curl git unzip wget

conda_dir=$HOME/miniforge3
if [ ! -x "$conda_dir/bin/conda" ]; then
    installer=$(mktemp)
    curl -fsSL -o "$installer" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$installer" -b -p "$conda_dir"
    rm -f "$installer"
fi
source "$conda_dir/etc/profile.d/conda.sh"
if [ ! -d "$env_prefix/conda-meta" ]; then
    if [ -e "$env_prefix" ]; then echo "$env_prefix exists. Choose another --env path." >&2; exit 1; fi
    conda create -y -q -p "$env_prefix" python=3.10 uv
else
    conda install -y -q -p "$env_prefix" uv
fi
conda activate "$env_prefix"

if [ ! -e "$gsv_root" ]; then
    git init -q "$gsv_root"
    git -C "$gsv_root" fetch -q --depth 1 "$GSV_REPO" "$GSV_COMMIT"
    git -C "$gsv_root" checkout -q FETCH_HEAD
elif [ "$(git -C "$gsv_root" rev-parse HEAD 2>/dev/null)" != "$GSV_COMMIT" ]; then
    echo "$gsv_root has another checkout. Choose a new --root path or use the manual setup guide." >&2
    exit 1
fi

# Pin matching builds so dependency resolution cannot replace CUDA with a CPU wheel.
uv pip install --python "$env_prefix/bin/python" \
    "torch==2.11.0+$device" "torchaudio==2.11.0+$device" torchcodec==0.11.1 \
    --index-url "https://download.pytorch.org/whl/$device"
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
printf 'torch==2.11.0+%s\ntorchaudio==2.11.0+%s\ntorchcodec==0.11.1\n' "$device" "$device" > "$constraints"
cd "$gsv_root"
# WORKFLOW skips upstream's unpinned PyTorch install and terminal control commands.
CONDA_PINNED_PACKAGES='ffmpeg>=6,<9' PIP_CONSTRAINT="$constraints" WORKFLOW=true \
    bash install.sh --device "${device^^}" --source "$source_name"
mkdir -p TEMP
export LD_LIBRARY_PATH="$env_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

GSV_ROOT="$gsv_root" GSV_ENV="$env_prefix" GSV_GPU="$gpu_index" python - <<'PY'
import json
import os
import torch
import torchcodec

print('PyTorch:', torch.__version__)
print('GPU:', torch.cuda.get_device_name(int(os.environ['GSV_GPU'])) if torch.cuda.is_available() else 'CPU')
print('Copy these paths into config.json:')
print(json.dumps({
    'wsl_gpt_sovits_root': os.environ['GSV_ROOT'],
    'wsl_python': os.environ['GSV_ENV'] + '/bin/python',
    'wsl_datasets_root': os.path.expanduser('~/voice-datasets'),
    'tts_gpu_index': int(os.environ['GSV_GPU']),
}, indent=2))
PY
