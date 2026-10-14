"""Install the voice engine, GPT-SoVITS, in WSL and point config.json at it. Setup-Engine.bat runs this."""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve().parent / "setup_engine.sh"
DEFAULT_DISTRO = "Ubuntu-24.04"
# setup_engine.sh exits with this when GPT-SoVITS is there from an install of the user's own.
OWN_INSTALL = 87

# The choices, best first. Each names the PyTorch build and the config.json settings it needs.
BUILDS = {
    "rtx": {"label": "NVIDIA RTX 20, 30, 40 or 50 series", "build": "cu128", "device": "cuda", "precision": "fp16", "download": 11},
    "amd": {"label": "AMD Radeon RX 7000 or 9000 series, with ROCm", "build": "rocm", "device": "cuda", "precision": "fp16", "download": 15},
    "gtx16": {"label": "NVIDIA GTX 16 series", "build": "cu128", "device": "cuda", "precision": "fp32", "download": 11},
    "pascal": {"label": "NVIDIA GTX 10 series, Pascal or Volta", "build": "cu126", "device": "cuda", "precision": "fp32", "download": 11},
    "cpu": {"label": "No supported graphics card: CPU only, which is slow", "build": "cpu", "device": "cpu", "precision": "fp32", "download": 7},
}


def choice_for_card(name):
    """The build for one graphics card, by its name as Windows lists it."""
    name = name.upper()
    if "NVIDIA" in name or "GEFORCE" in name or "QUADRO" in name:
        if "GTX 16" in name:
            return "gtx16"
        # Laptop MX150 to MX350 are Pascal; MX110, MX130 and 9xxMX are older Maxwell.
        if re.search(r"GTX 10\d0|TITAN XP?\b|TITAN V\b|QUADRO (GP|GV|P)\d|TESLA [PV]\d|\bMX(15\d|[23]\d\d)\b", name):
            return "pascal"
        if re.search(r"GTX|GT \d|QUADRO [KM]\d|TESLA [KM]\d|\d{3}MX\b|\bMX1[0-4]\d\b", name):
            return "cpu"
        return "rtx"
    if re.search(r"RADEON.*\b(RX [79]\d{3}|PRO W[79]\d{3}|AI PRO R9\d{3})", name):
        return "amd"
    return "cpu"


def detect_choice(names):
    """The best build among the graphics cards in this PC."""
    choices = {choice_for_card(name) for name in names}
    return next((key for key in BUILDS if key in choices), "cpu")


def graphics_cards():
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def wsl_text(data):
    # wsl.exe writes its own messages in UTF-16 unless WSL_UTF8 is set.
    return data.decode("utf-16-le" if b"\x00" in data else "utf-8", errors="replace")


def installed_distros():
    try:
        result = subprocess.run(["wsl.exe", "--list", "--quiet"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return [line.strip() for line in wsl_text(result.stdout).splitlines() if line.strip()]


def linux_identity(distro):
    """The distro's default user and its home folder."""
    result = subprocess.run(["wsl.exe", "-d", distro, "--exec", "sh", "-c", 'printf "%s\\n%s\\n" "$(id -un)" "$HOME"'],
                            capture_output=True, timeout=180)
    lines = wsl_text(result.stdout).split()
    if result.returncode != 0 or len(lines) != 2 or not lines[1].startswith("/"):
        raise RuntimeError(f"WSL did not start {distro}: {wsl_text(result.stdout + result.stderr).strip()}")
    return lines[0], lines[1]


def read_config(path):
    source = path if path.exists() else ROOT / "config.example.json"
    try:
        return json.loads(source.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        raise RuntimeError(f"{source.name} is not valid JSON ({exc}). Fix it, or delete it to start from config.example.json.") from exc


def engine_config(config, distro, home, choice):
    """config.json's settings for the engine this setup installs, keeping the others."""
    build = BUILDS[choice]
    return {
        **config,
        "wsl_distro": distro,
        "wsl_gpt_sovits_root": f"{home}/GPT-SoVITS",
        "wsl_python": f"{home}/miniforge3/envs/gsv/bin/python",
        "wsl_datasets_root": f"{home}/voice-datasets",
        "tts_device": build["device"],
        "tts_precision": build["precision"],
    }


def write_config(path, config):
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(part, path)


def ask_choice(detected, cards):
    print(f"Graphics card: {', '.join(cards) or 'none found'}")
    keys = list(BUILDS)
    print("\nChoose the build:")
    for number, key in enumerate(keys, 1):
        print(f"  {number}  {BUILDS[key]['label']}")
    while True:
        answer = input(f"Press Enter for {keys.index(detected) + 1}, or type a number: ").strip()
        if not answer:
            return detected
        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            return keys[int(answer) - 1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distro", help=f"WSL distro, by default config.json's wsl_distro or {DEFAULT_DISTRO}")
    parser.add_argument("--build", choices=list(BUILDS), help="Build to install, instead of asking")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json", help="config.json to write")
    parser.add_argument("--yes", action="store_true", help="Start without asking")
    args = parser.parse_args(argv)

    print("Local Voice Systems: install the voice engine, GPT-SoVITS, in WSL\n")
    try:
        config = read_config(args.config)
    except RuntimeError as exc:
        print(exc)
        return 1
    distro = args.distro or config.get("wsl_distro") or DEFAULT_DISTRO
    distros = installed_distros()
    if distros is None:
        print("WSL is not installed. In an administrator PowerShell, run:\n\n  wsl --install -d Ubuntu-24.04\n  wsl --update\n\n"
              "Restart if asked, open Ubuntu once to create your Linux user, then run Setup-Engine.bat again.")
        return 1
    if distro.lower() not in {name.lower() for name in distros}:
        print(f"WSL has no distro named {distro}. In an administrator PowerShell, run:\n\n  wsl --install -d {distro}\n\n"
              "Open it once to create your Linux user, then run Setup-Engine.bat again.")
        return 1
    try:
        user, home = linux_identity(distro)
    except (RuntimeError, subprocess.SubprocessError) as exc:
        print(exc)
        return 1
    print(f"WSL distro: {distro}, user {user}, home {home}")
    if user == "root" and not args.yes:
        print("\nThe distro's default user is root. If you have not opened Ubuntu yet to create your Linux user, close this window,"
              " open Ubuntu once, then run Setup-Engine.bat again. The dashboard then runs as that user, who cannot read root's files.")
        input("To install for root anyway, press Enter. ")

    cards = graphics_cards()
    choice = args.build or ask_choice(detect_choice(cards), cards)
    build = BUILDS[choice]
    print(f"\nInstalling for: {build['label']}")
    print(f"This downloads about {build['download']} GB into {distro} and takes 20 to 60 minutes.")
    if build["build"] == "rocm":
        print("ROCm in WSL needs a recent AMD Software: Adrenalin Edition driver in Windows.")
    if not args.yes:
        input("Press Enter to start, or close this window to stop. ")

    linux_script = wsl_text(subprocess.run(["wsl.exe", "-d", distro, "--exec", "wslpath", "-a", str(SCRIPT)],
                                           capture_output=True, timeout=60).stdout).strip()
    if not linux_script.startswith("/"):
        print(f"WSL cannot read {SCRIPT}. Check that Windows drives are mounted in {distro}.")
        return 1
    # First a check for a GPT-SoVITS of the user's own, before anything is installed. System
    # packages need root, and wsl -u root needs no password.
    for phase, user_args in (("check", []), ("root", ["-u", "root"]), ("user", [])):
        result = subprocess.run(["wsl.exe", "-d", distro, *user_args, "--exec", "bash", linux_script, phase, build["build"]])
        if result.returncode == OWN_INSTALL and phase != "root":
            print(f"\nGPT-SoVITS is already in {home}/GPT-SoVITS, from an install of your own, and setup leaves it as it is."
                  "\nSet its paths in config.json, as the README describes.")
            return 1
        if result.returncode != 0:
            print("\nSetup stopped with an error, shown above. Run Setup-Engine.bat again to continue from where it stopped.")
            return 1

    python = f"{home}/miniforge3/envs/gsv/bin/python"
    try:
        check = subprocess.run(["wsl.exe", "-d", distro, "--exec", python, "-c", "import torch; print(torch.cuda.is_available())"],
                               capture_output=True, timeout=600)
    except subprocess.TimeoutExpired:
        check = None
    if check is None or check.returncode != 0:
        detail = "" if check is None else wsl_text(check.stdout + check.stderr).strip()[-600:]
        print(f"\nPyTorch does not load in {distro}, so config.json is left as it is.\n{detail or 'It did not answer within 10 minutes.'}")
        return 1
    gpu = wsl_text(check.stdout).strip().endswith("True")
    write_config(args.config, engine_config(read_config(args.config), distro, home, choice))
    print(f"\nThe voice engine is installed, and {args.config.name} points at it.")
    if build["device"] == "cuda" and not gpu:
        print("PyTorch does not see the graphics card yet. "
              + ("Update the AMD graphics driver in Windows." if build["build"] == "rocm"
                 else "Update the NVIDIA driver in Windows, and check that /usr/lib/wsl/lib/nvidia-smi lists the card in Ubuntu.")
              + ' Until then, the engine can run on the CPU: set "tts_device" to "cpu" in config.json.')
    print("Next, run Setup-Relay-Env.bat, then Launch-Voice-Dashboard.bat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
