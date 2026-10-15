"""Commands in WSL, where GPT-SoVITS runs, and the GPU settings its programs read."""

import subprocess

from . import config, jobs


# pkill patterns for jobs that run inside WSL. They also work after a dashboard restart, when the
# job is no longer known. The brackets keep a pattern from matching the stop command's own shell.
WSL_STOP_SCRIPTS = {
    "dataset-sync": "pkill -f '[d]ataset-sync; ' || true",
    "dataset-prepare": "pkill -f '[G]PT_SoVITS/prepare_datasets/' || true",
    "train-gpt": "pkill -f '[G]PT_SoVITS/s1_train.py' || true",
    "train-sovits": "pkill -f '[G]PT_SoVITS/s2_train.py' || true",
    "api": f"pkill -f '[a]pi_v2.py.*-p {config.TTS_PORT} ' || true",
    "separate": "pkill -f '[s]eparate_audio.py' || true",
}


def shquote(value):
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def wsl_bash(script):
    # --exec hands the script to bash unchanged. With "--", wsl.exe first runs the command line
    # through the default shell, which expands the script's own variables before bash sets them.
    return ["wsl", "-d", config.WSL_DISTRO, "--exec", "bash", "-lc", script]


def wsl_command(script):
    """Run script in WSL with the voice engine's Python first on PATH."""
    if config.WSL_PATHS_ERROR:
        raise ValueError(config.WSL_PATHS_ERROR)
    python_dir = config.WSL_PYTHON.rsplit("/", 1)[0]
    lib_dir = python_dir.rsplit("/", 1)[0] + "/lib"
    script = (
        f'export PATH={shquote(python_dir)}:"$PATH":/usr/lib/wsl/lib; '
        f'export LD_LIBRARY_PATH={shquote(lib_dir)}"${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"; '
        + script
    )
    return wsl_bash(script)


def run_wsl(script, **kwargs):
    return subprocess.run(
        wsl_command(script),
        text=True,
        capture_output=True,
        **kwargs,
    )


def stop_in_wsl(script):
    """Run a pkill stop script in WSL. A stopped distro runs nothing to stop, and starting it
    only for this took longer than the timeout, so it is skipped. pkill needs none of the paths
    from config.json, so this also works while they are unset."""
    if not wsl_distro_running():
        return
    try:
        subprocess.run(wsl_bash(script), text=True, capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        jobs.log(f"Could not run the stop command in WSL: {exc}")


def voice_runtime():
    # The engine's config files are written into its folder in WSL, so its paths come first.
    if config.WSL_PATHS_ERROR:
        raise ValueError(config.WSL_PATHS_ERROR)
    device = config.APP_CONFIG.get("tts_device", "cuda")
    precision = config.APP_CONFIG.get("tts_precision", "fp32")
    if device not in {"cuda", "cpu"} or precision not in {"fp16", "fp32"}:
        raise ValueError("Set tts_device to cuda or cpu, and tts_precision to fp16 or fp32.")
    return {
        "device": device,
        "is_half": device == "cuda" and precision == "fp16",
        "gpu_index": "0" if device == "cuda" else "",
    }


def voice_environment():
    runtime = voice_runtime()
    return (
        f"export CUDA_VISIBLE_DEVICES={shquote(runtime['gpu_index'])}; "
        f"export _CUDA_VISIBLE_DEVICES={shquote(runtime['gpu_index'])}; "
        f"export is_half={runtime['is_half']}; "
    )


def wsl_distro_running():
    try:
        result = subprocess.run(
            ["wsl", "--list", "--running", "--quiet"],
            capture_output=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    # wsl.exe prints UTF-16 unless WSL_UTF8 is set.
    text = result.stdout.decode("utf-16-le" if b"\x00" in result.stdout else "utf-8", errors="ignore")
    return config.WSL_DISTRO.lower() in {line.strip().lower() for line in text.splitlines()}


def wake_wsl():
    """Start the distro before its files are read through \\wsl.localhost."""
    if run_wsl(":", timeout=90).returncode != 0:
        raise RuntimeError("WSL did not start. Run System Check.")
