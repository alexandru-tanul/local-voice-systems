"""The GPT and SoVITS models in the GPT-SoVITS folder: trained ones, and the pretrained ones they
start from."""

import time

from . import config, wsl


# SoVITS model versions. v2ProPlus has a larger decoder. Both share the GPT model and the features
# that preparation extracts, as their pretrained models have the same quantizer.
SOVITS_VERSIONS = ("v2Pro", "v2ProPlus")
MODEL_LIST = {"checked_at": 0.0, "listed": False, "models": {"gpt": [], "sovits": []}}


def sovits_version(path):
    """The version of a SoVITS model, from its folder or the name of a pretrained one."""
    return "v2ProPlus" if "v2ProPlus" in str(path) else "v2Pro"


def model_files():
    def collect(folders, pattern):
        rows = []
        for folder in folders:
            try:
                for path in (config.WSL_GSV_WIN / folder).glob(pattern):
                    if path.is_file():
                        rows.append(
                            {
                                "name": path.name,
                                "path": f"{folder}/{path.name}",
                                "version": sovits_version(f"{folder}/{path.name}"),
                                "mtime": path.stat().st_mtime,
                                "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
                            }
                        )
            except OSError:
                pass
        rows.sort(key=lambda row: row["mtime"], reverse=True)
        return rows

    # Trained models first, newest first, then the pretrained ones they start from.
    models = {
        "gpt": collect([f"GPT_weights_{version}" for version in SOVITS_VERSIONS], "*.ckpt")
        + collect(["GPT_SoVITS/pretrained_models"], "s1v3.ckpt"),
        "sovits": collect([f"SoVITS_weights_{version}" for version in SOVITS_VERSIONS], "*.pth")
        + collect(["GPT_SoVITS/pretrained_models/v2Pro"], "s2Gv2Pro.pth")
        + collect(["GPT_SoVITS/pretrained_models/v2Pro"], "s2Gv2ProPlus.pth"),
    }
    MODEL_LIST.update(models=models, checked_at=time.time(), listed=True)
    return models


def polled_model_files(max_age=15):
    """Model list for the page's 5-second poll.

    It rescans at most every max_age seconds, and only while the distro runs, because reading
    \\\\wsl.localhost starts a stopped distro. Starting an engine or a check rescans at once."""
    if time.time() - MODEL_LIST["checked_at"] >= max_age:
        MODEL_LIST["checked_at"] = time.time()
        if wsl.wsl_distro_running():
            model_files()
    return MODEL_LIST["models"]


def model_pair(gpt_path, sovits_path):
    """The requested models, or the first listed ones for a request that names none or a model
    that no longer exists. The page can only send the choice it saved while its list is empty."""
    models = model_files()
    gpt_paths = [row["path"] for row in models["gpt"]]
    sovits_paths = [row["path"] for row in models["sovits"]]
    gpt = gpt_path if gpt_path in gpt_paths else (gpt_paths[0] if gpt_paths else "")
    sovits = sovits_path if sovits_path in sovits_paths else (sovits_paths[0] if sovits_paths else "")
    return gpt, sovits
