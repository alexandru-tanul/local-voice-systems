#!/usr/bin/env bash
# Installs GPT-SoVITS for Local Voice Systems. Setup-Engine.bat runs it in WSL three times: as the
# distro's user to check for a GPT-SoVITS of the user's own ("check"), as root for system packages
# ("root"), and as the user for everything in the home folder ("user"). Each step is skipped when
# it is already done, so a run that failed can be started again.
#
#   bash setup_engine.sh check|root|user cu128|cu126|rocm|cpu
set -eo pipefail

phase=$1
build=$2
GSV_COMMIT=48b1a0169a28582a8984402f82cf438d3bfa6aca
GSV_DIR="$HOME/GPT-SoVITS"
FORGE="$HOME/miniforge3"
MARKER="$GSV_DIR/.local-voice-systems"
# AMD's PyTorch for ROCm 7.2 in WSL, which uses the ROCm libraries in /opt/rocm.
ROCM_WHEELS=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2
# Exit status for a GPT-SoVITS that this setup did not install. Setup-Engine.bat reads it.
OWN_INSTALL=87

step() { printf '\n== %s\n' "$*"; }

# Right after the distro starts, unattended-upgrades can hold apt's lock for a few minutes.
retry() {
    local attempt
    for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
        if "$@"; then
            return 0
        fi
        echo "$1 failed or is waiting for another install; trying again in 10 seconds ($attempt of 12)..."
        sleep 10
    done
    return 1
}

case "$build" in
cu128 | cu126 | rocm | cpu) ;;
*)
    echo "Unknown build: $build"
    exit 2
    ;;
esac

# A GPT-SoVITS that this setup did not install belongs to the user, and is left as it is.
if [ "$phase" = check ]; then
    if [ -d "$GSV_DIR" ] && [ ! -f "$MARKER" ]; then
        echo "$GSV_DIR exists and was not installed by this setup, so it is left as it is."
        exit "$OWN_INSTALL"
    fi
    exit 0
fi

if [ "$phase" = root ]; then
    # Without name lookup, apt and the downloads below retry for a long time and then fail.
    if ! getent hosts archive.ubuntu.com github.com >/dev/null; then
        echo "WSL cannot look up names on the internet, so nothing can be downloaded."
        echo "Fix the network in WSL, for example by setting a DNS server in /etc/resolv.conf, then run Setup-Engine.bat again."
        exit 1
    fi
    step "Installing system packages"
    export DEBIAN_FRONTEND=noninteractive
    # Completes an install that was cut short, which otherwise stops every apt-get below.
    retry dpkg --configure -a
    retry apt-get update -q
    retry apt-get install -y -q build-essential git wget unzip ca-certificates
    if [ "$build" = rocm ]; then
        if [ -e /opt/rocm/lib/libhsa-runtime64.so ]; then
            # AMD's PyTorch below needs ROCm 7.2, with the runtime that works in WSL.
            rocm_version=$(cat /opt/rocm/.info/version 2>/dev/null || true)
            if [[ "$rocm_version" != 7.2.* ]] || [ "$(dpkg-query -W -f='${Status}' hsa-runtime-rocr4wsl-amdgpu 2>/dev/null)" != "install ok installed" ]; then
                echo "ROCm ${rocm_version:-of an unknown version} is installed in this distro, and this setup needs AMD's ROCm 7.2 for WSL."
                echo "Update it with AMD's amdgpu-install 7.2 and the wsl,rocm use cases, or install into another distro, then run Setup-Engine.bat again."
                exit 1
            fi
        else
            . /etc/os-release
            case "$VERSION_CODENAME" in
            noble | jammy) ;;
            *)
                echo "ROCm for WSL needs Ubuntu 24.04 or 22.04, and this is $PRETTY_NAME."
                exit 1
                ;;
            esac
            step "Installing AMD ROCm 7.2 for WSL. This downloads about 7 GB."
            wget -q --show-progress -O /tmp/amdgpu-install.deb \
                "https://repo.radeon.com/amdgpu-install/7.2/ubuntu/$VERSION_CODENAME/amdgpu-install_7.2.70200-1_all.deb"
            retry apt-get install -y -q /tmp/amdgpu-install.deb
            rm -f /tmp/amdgpu-install.deb
            retry amdgpu-install -y --usecase=wsl,rocm --no-dkms
        fi
    fi
    exit 0
fi

if [ "$phase" != user ]; then
    echo "Unknown phase: $phase"
    exit 2
fi

if [ -d "$GSV_DIR" ] && [ ! -f "$MARKER" ]; then
    echo "$GSV_DIR exists and was not installed by this setup, so it is left as it is."
    exit "$OWN_INSTALL"
fi

if [ ! -x "$FORGE/bin/conda" ]; then
    step "Installing Miniforge"
    wget -q --show-progress -O /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    # -u completes a Miniforge that an earlier run left half installed.
    bash /tmp/miniforge.sh -b -u -p "$FORGE"
    rm -f /tmp/miniforge.sh
fi
# shellcheck disable=SC1091
source "$FORGE/etc/profile.d/conda.sh"

# AMD's PyTorch for WSL is tested here with Python 3.12; the other builds use GPT-SoVITS's 3.10.
python_version=3.10
if [ "$build" = rocm ]; then
    python_version=3.12
fi
if [ ! -x "$FORGE/envs/gsv/bin/python" ]; then
    step "Creating the Python environment"
    conda create -y -q -n gsv "python=$python_version" uv
fi
conda activate gsv
if [ "$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')" != "$python_version" ]; then
    echo "The conda environment gsv has another Python than the $python_version this build needs."
    echo "Remove it with: $FORGE/bin/conda env remove -y -n gsv, then run Setup-Engine.bat again."
    exit 1
fi
# An environment created by a run that was cut short can lack uv, while conda lists it as installed.
if ! command -v uv >/dev/null; then
    conda install -y -q --force-reinstall uv
    if ! command -v uv >/dev/null; then
        echo "uv is missing from the conda environment gsv. Remove it with: $FORGE/bin/conda env remove -y -n gsv, then run Setup-Engine.bat again."
        exit 1
    fi
fi

if [ ! -d "$GSV_DIR" ]; then
    step "Downloading GPT-SoVITS"
    # Cloned beside its place first, so a clone that stops halfway is not taken for a user's own.
    rm -rf "$GSV_DIR.part"
    git clone https://github.com/RVC-Boss/GPT-SoVITS.git "$GSV_DIR.part"
    printf '%s\n' "$build" >"$GSV_DIR.part/.local-voice-systems"
    mv "$GSV_DIR.part" "$GSV_DIR"
fi
cd "$GSV_DIR"
# A newer pin than the clone knows is fetched first.
if ! git cat-file -e "$GSV_COMMIT^{commit}" 2>/dev/null; then
    git fetch -q origin
fi
git -c advice.detachedHead=false checkout -q "$GSV_COMMIT"

# These pins keep PyTorch, torchaudio, and FFmpeg compatible. Each package is checked, so an
# install that was cut short is completed.
torch_ready() {
    python - "$build" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

build = sys.argv[1]
if build == "rocm":
    wanted = {"torch": "2.8.0+rocm7.2", "torchaudio": "2.8.0+rocm7.2"}
else:
    wanted = {"torch": f"2.11.0+{build}", "torchaudio": f"2.11.0+{build}", "torchcodec": "0.11.1"}
try:
    sys.exit(0 if all(version(name).startswith(prefix) for name, prefix in wanted.items()) else 1)
except PackageNotFoundError:
    sys.exit(1)
PY
}
if [ "$build" = rocm ]; then
    if ! torch_ready; then
        step "Installing PyTorch for ROCm. This downloads about 2 GB."
        uv pip install \
            "$ROCM_WHEELS/torch-2.8.0%2Brocm7.2.0.lw.gitbf943426-cp312-cp312-linux_x86_64.whl" \
            "$ROCM_WHEELS/torchaudio-2.8.0%2Brocm7.2.0.git6e1c7fe9-cp312-cp312-linux_x86_64.whl" \
            "$ROCM_WHEELS/triton-3.4.0%2Brocm7.2.0.git0cace8d2-cp312-cp312-linux_x86_64.whl"
    fi
    printf 'torch==2.8.0\ntorchaudio==2.8.0\n' >voice-constraints.txt
else
    if ! torch_ready; then
        step "Installing PyTorch ($build)"
        uv pip install "torch==2.11.0+$build" "torchaudio==2.11.0+$build" torchcodec==0.11.1 \
            --index-url "https://download.pytorch.org/whl/$build"
    fi
    printf 'torch==2.11.0\ntorchaudio==2.11.0\ntorchcodec==0.11.1\n' >voice-constraints.txt
fi

# install.sh skips the model downloads once these folders exist, so an unzip that was cut short
# would never be completed. Without their key files, the folders go and the download is repeated.
MODEL_FILES="GPT_SoVITS/pretrained_models/s1v3.ckpt GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth
GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth
GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
for file in $MODEL_FILES; do
    if [ ! -s "$file" ]; then
        rm -rf GPT_SoVITS/pretrained_models/sv
        break
    fi
done
if [ ! -s GPT_SoVITS/text/G2PWModel/g2pW.onnx ]; then
    rm -rf GPT_SoVITS/text/G2PWModel
fi

step "Installing GPT-SoVITS's packages and models. This downloads about 6 GB and is quiet for long stretches."
CONDA_PINNED_PACKAGES='ffmpeg>=6,<9' PIP_CONSTRAINT="$PWD/voice-constraints.txt" WORKFLOW=true \
    bash install.sh --device "$(printf '%s' "$build" | tr '[:lower:]' '[:upper:]')" --source HF

for file in $MODEL_FILES GPT_SoVITS/text/G2PWModel/g2pW.onnx; do
    if [ ! -s "$file" ]; then
        echo "GPT-SoVITS's model $file is missing after its install. Run Setup-Engine.bat again."
        exit 1
    fi
done
mkdir -p "$HOME/voice-datasets"
printf '%s\ndone\n' "$build" >"$MARKER"
step "GPT-SoVITS is installed."
