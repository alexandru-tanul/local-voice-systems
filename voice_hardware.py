"""Probe PyTorch in the voice engine's environment, without importing it in the dashboard."""

import argparse
import json


def detect_runtime(torch, device="auto", precision="auto", gpu_index=0):
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("tts_device must be auto, cuda, or cpu.")
    if precision not in {"auto", "fp16", "fp32"}:
        raise ValueError("tts_precision must be auto, fp16, or fp32.")
    if gpu_index < 0:
        raise ValueError("tts_gpu_index must be zero or greater.")
    cpu = {"device": "cpu", "is_half": False, "gpu_index": "", "detail": "CPU (FP32). Synthesis and training will be slow."}
    if device == "cpu":
        if precision == "fp16":
            raise ValueError("CPU mode needs tts_precision auto or fp32.")
        return cpu
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access a GPU. Check the driver and PyTorch build.")
        if gpu_index >= torch.cuda.device_count():
            raise ValueError(f"GPU {gpu_index} does not exist. Check tts_gpu_index.")
        name = torch.cuda.get_device_name(gpu_index)
        capability = torch.cuda.get_device_capability(gpu_index)
        # Pascal and GTX 16 cards need FP32 in GPT-SoVITS, even with CUDA available.
        half = bool(torch.version.hip) or (capability >= (7, 0) and "GTX 16" not in name.upper())
        if precision != "auto":
            half = precision == "fp16"
        tensor = torch.ones((8, 8), device=f"cuda:{gpu_index}", dtype=torch.float16 if half else torch.float32)
        if not (tensor @ tensor).isfinite().all().item():
            raise RuntimeError("The GPU failed a small matrix calculation.")
        torch.cuda.synchronize(gpu_index)
        return {
            "device": "cuda",
            "is_half": half,
            "gpu_index": str(gpu_index),
            "detail": f"{name} ({'ROCm' if torch.version.hip else 'CUDA'}, {'FP16' if half else 'FP32'})",
        }
    except ValueError:
        raise
    except Exception as exc:
        if device == "cuda" or precision == "fp16":
            raise RuntimeError(f"GPU check failed: {exc}") from exc
        cpu["detail"] += f" GPU check: {exc}"
        return cpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args()
    import torch

    print(json.dumps(detect_runtime(torch, args.device, args.precision, args.gpu_index)))
