# Voice changer

[Clone Applio](https://github.com/IAHispano/Applio), run `run-install.bat` on Windows, then set `applio_root` in `config.json` to its folder.
Restart the dashboard. Applio uses its own Python and GPU packages.

| Hardware | Setup |
| --- | --- |
| NVIDIA GTX 16 / RTX | Use Applio's CUDA install. No AMD SDK needed. |
| NVIDIA Pascal / Volta | Replace its PyTorch packages with CUDA 12.6 using the command below. |
| CPU | Set `applio_backend` to `cpu`. Live conversion may be slow. |
| AMD | Follow [Applio's AMD guide](https://docs.applio.org/getting-started/installation/#amd-gpu-support-windows). For ZLUDA, set `applio_backend` to `zluda` and `amd_hip_bin` to the HIP `bin` folder. |

For Pascal/Volta, run from Applio's folder after its installer:

```text
.\env\python.exe -m uv pip install --python .\env\python.exe "torch==2.11.0+cu126" "torchaudio==2.11.0+cu126" --index-url https://download.pytorch.org/whl/cu126
```

On the dashboard's **Voice Changer** page:

1. Choose a dataset and press **Prepare RVC Dataset**.
2. In Applio, preprocess and extract features for the shown model name. Use RVC v2, F0, and 32 kHz.
3. Press **Start RVC Training** with batch size 1, then **Build Index**.
4. Press **Start Live Engine**, choose the microphone, output, model, and index, then **Start Live Voice**.

Use headphones. Stop GPT-SoVITS if GPU memory is tight.
