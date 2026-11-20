# Optional settings

These keys in `config.json` are optional, for when a port is taken or you want a different setting. Restart the dashboard after editing `config.json`.

| Key | Default | Use |
| --- | --- | --- |
| `dashboard_port` | `8790` | Port of the dashboard page |
| `asr_port` | `8792` | Port of speech recognition |
| `tts_port` | `9880` | Port of the voice engine in WSL |
| `asr_model` | `base.en` | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) model of Mic to Voice, for example `small.en` for better accuracy; for other languages it uses the same model without `.en`, or `base` for the English-only `distil` models and for a folder or repository ending in `.en` |
| `dataset_asr_model` | `small.en`, or `small` for other languages | faster-whisper model that writes dataset transcripts |
| `asr_model_dir` | `cache_dir/faster-whisper` | Download folder of the speech and translation models |
| `cache_dir`, `tmp_dir` | `.cache`, `cache_dir/tmp` | Download caches and temporary files |
| `dashboard_python` | `relay_env` | Python that runs the dashboard |
| `restore_python` | `restore_env` | Python that runs Restore audio |
| `api_key` | none | Key that other programs must send to speak with your voices; see [Use in other programs](USAGE.md#use-in-other-programs) |

## Downloads on first use

- Speech recognition uses CPU and downloads its model on first use; dataset transcripts use a larger one, about 480 MB.
- Speech in languages other than English uses one more speech recognition model: about 145 MB with the standard `asr_model`.
- Translation in Mic to Voice downloads NLLB-200, about 650 MB.
- `Setup-Restore-Env.bat` downloads about 1 GB, and Restore audio a model of about 55 MB on first use.
- Music and echo removal download GPT-SoVITS's separation models, about 520 MB.
