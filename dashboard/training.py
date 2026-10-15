"""Sync & Prepare and training: copying a dataset to WSL, extracting its features, and training
the GPT and SoVITS models."""

import json
import re
import time

from . import config, datasets, jobs, models, wsl


class TrainingProgress:
    """Follows a training job's output: the last epoch it finished, and the time an epoch takes.

    SoVITS training logs "====> Epoch: N" after epoch N. GPT training shows Lightning's progress
    bar, "Epoch N: 100%", counting from 0. Both continue from an earlier run's checkpoint, so the
    first epoch seen need not be 1."""

    PATTERNS = {"sovits": re.compile(r"====> Epoch: (\d+)"), "gpt": re.compile(r"\bEpoch (\d+):\s*100%")}

    def __init__(self, kind, epochs):
        self.kind = kind
        self.epochs = int(epochs)
        self.epoch = 0
        self.finished = []

    def __call__(self, line):
        match = self.PATTERNS[self.kind].search(line)
        if not match:
            return
        epoch = int(match.group(1)) + (1 if self.kind == "gpt" else 0)
        # The bar shows 100% more than once while it updates its numbers.
        if epoch != self.epoch:
            self.epoch = epoch
            self.finished.append(time.time())

    def state(self):
        left = None
        if len(self.finished) >= 2:
            per_epoch = (self.finished[-1] - self.finished[0]) / (len(self.finished) - 1)
            left = round(per_epoch * max(0, self.epochs - self.epoch))
        return {"epoch": self.epoch, "epochs": self.epochs, "seconds_left": left}


def write_s1_config(epochs, batch_size, save_every, model_name="voice-model"):
    runtime = wsl.voice_runtime()
    model_name = datasets.slugify(model_name, "voice-model")
    text = f"""data:
  max_eval_sample: 8
  max_sec: 54
  num_workers: 4
  pad_val: 1024
inference:
  top_k: 15
model:
  EOS: 1024
  dropout: 0
  embedding_dim: 512
  head: 16
  hidden_dim: 512
  linear_units: 2048
  n_layer: 24
  phoneme_vocab_size: 732
  random_bert: 0
  vocab_size: 1025
optimizer:
  decay_steps: 40000
  lr: 0.01
  lr_end: 0.0001
  lr_init: 1.0e-05
  warmup_steps: 2000
output_dir: logs/{model_name}/logs_s1_v2Pro
pretrained_s1: GPT_SoVITS/pretrained_models/s1v3.ckpt
train:
  batch_size: {int(batch_size)}
  epochs: {int(epochs)}
  exp_name: {model_name}
  gradient_clip: 1.0
  half_weights_save_dir: GPT_weights_v2Pro
  if_dpo: false
  if_save_every_weights: true
  if_save_latest: true
  precision: {"16-mixed" if runtime['is_half'] else '"32"'}
  save_every_n_epoch: {int(save_every)}
  seed: 1234
train_phoneme_path: logs/{model_name}/2-name2text.txt
train_semantic_path: logs/{model_name}/6-name2semantic.tsv
"""
    path = config.WSL_GSV_WIN / "TEMP" / f"{model_name}_s1.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"{config.WSL_GSV_ROOT}/TEMP/{model_name}_s1.yaml"


def write_s2_config(epochs, batch_size, save_every, model_name="voice-model", include_model_version=True, version="v2Pro"):
    runtime = wsl.voice_runtime()
    model_name = datasets.slugify(model_name, "voice-model")
    plus = version == "v2ProPlus"
    settings = {
        "train": {
            "log_interval": 100,
            "eval_interval": 500,
            "seed": 1234,
            "epochs": int(epochs),
            "learning_rate": 0.0001,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": int(batch_size),
            "fp16_run": runtime["is_half"],
            "lr_decay": 0.999875,
            "segment_size": 20480,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
            "text_low_lr_rate": 0.4,
            "grad_ckpt": False,
            "pretrained_s2G": f"GPT_SoVITS/pretrained_models/v2Pro/s2G{version}.pth",
            "pretrained_s2D": f"GPT_SoVITS/pretrained_models/v2Pro/s2D{version}.pth",
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": int(save_every),
            "gpu_numbers": runtime["gpu_index"],
            "lora_rank": "32",
        },
        "data": {
            "max_wav_value": 32768.0,
            "sampling_rate": 32000,
            "filter_length": 2048,
            "hop_length": 640,
            "win_length": 2048,
            "n_mel_channels": 128,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "n_speakers": 300,
            "cleaned_text": True,
            "exp_dir": f"logs/{model_name}",
        },
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 8, 2, 2, 2],
            # GPT-SoVITS's configs/s2v2ProPlus.json differs from s2v2Pro.json only in these two.
            "upsample_initial_channel": 768 if plus else 512,
            "upsample_kernel_sizes": [20 if plus else 16, 16, 8, 2, 2],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "gin_channels": 1024,
            "semantic_frame_rate": "25hz",
            "freeze_quantizer": True,
            "version": version,
        },
        "s2_ckpt_dir": f"logs/{model_name}",
        "content_module": "cnhubert",
        "save_weight_dir": f"SoVITS_weights_{version}",
        "name": model_name,
        "version": version,
    }
    if not include_model_version:
        settings["model"].pop("version", None)
    path = config.WSL_GSV_WIN / "TEMP" / f"{model_name}_s2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return f"{config.WSL_GSV_ROOT}/TEMP/{model_name}_s2.json"


def dataset_sync_script(descriptor):
    target = descriptor["wsl_root"]
    list_name = descriptor["list_path"].name
    # "&", "|" and "\" have a meaning in a sed replacement, so they are escaped.
    sed_target = target.replace("\\", "\\\\").replace("&", "\\&").replace("|", "\\|")
    return (
        "set -e; "
        # wslpath maps the Windows folder the way this distro mounts drives.
        f"source_dir=$(wslpath -u {wsl.shquote(str(descriptor['root']))}); "
        f"mkdir -p {wsl.shquote(target)}/wavs; "
        # -u copies only clips that are new or changed since the last sync.
        f'cp -ru "$source_dir"/wavs/. {wsl.shquote(target)}/wavs/; '
        f'cp "$source_dir"/{wsl.shquote(list_name)} {wsl.shquote(target)}/source.list; '
        f"sed {wsl.shquote(f's|^wavs/|{sed_target}/wavs/|')} {wsl.shquote(target)}/source.list > {wsl.shquote(target)}/training.list; "
        f"echo Dataset synced; wc -l {wsl.shquote(target + '/training.list')}"
    )


def sync_dataset_job(dataset_id):
    descriptor = datasets.dataset_descriptor(dataset_id)
    summary = datasets.dataset_summary(descriptor)
    if not summary["ready"]:
        raise RuntimeError("The selected dataset needs matching audio files and transcript rows before it can be synced.")
    # ": dataset-sync" is a no-op that puts a name for pkill into this shell's command line.
    proc = jobs.ManagedProcess(
        f"Sync {descriptor['name']}",
        wsl.wsl_command(": dataset-sync; " + dataset_sync_script(descriptor)),
        stop_script=wsl.WSL_STOP_SCRIPTS["dataset-sync"],
    )
    with jobs.STATE_LOCK:
        current = jobs.JOBS.get("dataset-sync")
        if current and current.process and current.process.poll() is None:
            raise RuntimeError("A dataset sync is already running.")
        jobs.JOBS["dataset-sync"] = proc
    proc.start()
    jobs.log(f"Syncing dataset {descriptor['name']} to WSL.")


def prepare_tts_dataset_job(dataset_id, model_name):
    descriptor = datasets.dataset_descriptor(dataset_id)
    summary = datasets.dataset_summary(descriptor)
    if not summary["ready"]:
        raise RuntimeError("The selected dataset is incomplete. Add matching audio and transcripts first.")
    if jobs.job_running("dataset-prepare"):
        raise RuntimeError("Dataset preparation is already running.")
    jobs.refuse_gpu_conflicts("preparation")
    jobs.free_gpu_for("dataset preparation")
    model_name = datasets.slugify(model_name, "voice-model")
    s2_config = write_s2_config(1, 1, 1, model_name, include_model_version=False)
    target = descriptor["wsl_root"]
    opt_dir = f"{config.WSL_GSV_ROOT}/logs/{model_name}"
    common = (
        f"export inp_text={wsl.shquote(target + '/training.list')}; "
        f"export inp_wav_dir={wsl.shquote(target + '/wavs')}; "
        f"export exp_name={wsl.shquote(model_name)}; export i_part=0; export all_parts=1; "
        + wsl.voice_environment() + "export version=v2Pro; "
        f"export opt_dir={wsl.shquote(opt_dir)}; "
        f"export bert_pretrained_dir={wsl.shquote(config.WSL_GSV_ROOT + '/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large')}; "
        f"export cnhubert_base_dir={wsl.shquote(config.WSL_GSV_ROOT + '/GPT_SoVITS/pretrained_models/chinese-hubert-base')}; "
        f"export pretrained_s2G={wsl.shquote(config.WSL_GSV_ROOT + '/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth')}; "
        f"export sv_path={wsl.shquote(config.WSL_GSV_ROOT + '/GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt')}; "
        f"export s2config_path={wsl.shquote(s2_config)}; "
    )
    script = (
        "set -e; "
        + dataset_sync_script(descriptor)
        + f"; mkdir -p {wsl.shquote(opt_dir)}; "
        + f"rm -f {wsl.shquote(opt_dir)}/2-name2text-0.txt {wsl.shquote(opt_dir)}/6-name2semantic-0.tsv; "
        + common
        + f"cd {wsl.shquote(config.WSL_GSV_ROOT)}; "
        + f"{wsl.shquote(config.WSL_PYTHON)} -s GPT_SoVITS/prepare_datasets/1-get-text.py; "
        + f"cp {wsl.shquote(opt_dir)}/2-name2text-0.txt {wsl.shquote(opt_dir)}/2-name2text.txt; "
        + f"{wsl.shquote(config.WSL_PYTHON)} -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py; "
        + f"{wsl.shquote(config.WSL_PYTHON)} -s GPT_SoVITS/prepare_datasets/2-get-sv.py; "
        + f"{wsl.shquote(config.WSL_PYTHON)} -s GPT_SoVITS/prepare_datasets/3-get-semantic.py; "
        + f"printf 'item_name\\tsemantic_audio\\n' > {wsl.shquote(opt_dir)}/6-name2semantic.tsv; "
        + f"cat {wsl.shquote(opt_dir)}/6-name2semantic-0.tsv >> {wsl.shquote(opt_dir)}/6-name2semantic.tsv; "
        + f"echo Training features ready for {wsl.shquote(model_name)}."
    )
    proc = jobs.ManagedProcess(
        f"Prepare {descriptor['name']} for {model_name}",
        wsl.wsl_command(script),
        # Matches this job's shell, whose command line names the scripts, and each running script.
        stop_script=wsl.WSL_STOP_SCRIPTS["dataset-prepare"],
    )
    with jobs.STATE_LOCK:
        current = jobs.JOBS.get("dataset-prepare")
        if current and current.process and current.process.poll() is None:
            raise RuntimeError("Dataset preparation is already running.")
        jobs.JOBS["dataset-prepare"] = proc
    proc.start()
    jobs.log(f"Preparing dataset {descriptor['name']} for model {model_name}.")


def start_train(kind, epochs, batch_size, save_every, dataset_id="", model_name="voice-model", version="v2Pro"):
    descriptor = datasets.dataset_descriptor(dataset_id)
    if version not in models.SOVITS_VERSIONS:
        raise RuntimeError("Choose v2Pro or v2ProPlus.")
    if jobs.job_running(f"train-{kind}"):
        raise RuntimeError(f"{'GPT' if kind == 'gpt' else 'SoVITS'} training is already running.")
    jobs.refuse_gpu_conflicts("training")
    model_name = datasets.slugify(model_name, "voice-model")
    feature_root = f"{config.WSL_GSV_ROOT}/logs/{model_name}"
    check = wsl.run_wsl(
        f"test -s {wsl.shquote(feature_root + '/2-name2text.txt')} -a -s {wsl.shquote(feature_root + '/6-name2semantic.tsv')}",
        timeout=15,
    )
    if check.returncode != 0:
        raise RuntimeError("Training features are missing. Run Sync & Prepare before starting training.")
    if kind == "sovits" and version != "v2Pro":
        pretrained = f"{config.WSL_GSV_ROOT}/GPT_SoVITS/pretrained_models/v2Pro"
        check = wsl.run_wsl(f"test -s {wsl.shquote(f'{pretrained}/s2G{version}.pth')} -a -s {wsl.shquote(f'{pretrained}/s2D{version}.pth')}", timeout=15)
        if check.returncode != 0:
            raise RuntimeError(f"GPT-SoVITS is missing the pretrained {version} models, s2G{version}.pth and s2D{version}.pth in "
                               "GPT_SoVITS/pretrained_models/v2Pro. Update GPT-SoVITS, or train v2Pro.")
    jobs.free_gpu_for("training")
    if kind == "gpt":
        config_path = write_s1_config(epochs, batch_size, save_every, model_name)
        script = (
            wsl.voice_environment() + "ulimit -l 2097152; "
            f"cd {wsl.shquote(config.WSL_GSV_ROOT)} && "
            f"{wsl.shquote(config.WSL_PYTHON)} -s GPT_SoVITS/s1_train.py --config_file {wsl.shquote(config_path)}"
        )
        name = f"GPT training: {model_name}"
    elif kind == "sovits":
        config_path = write_s2_config(epochs, batch_size, save_every, model_name, version=version)
        script = (
            wsl.voice_environment() + "ulimit -l 2097152; "
            f"cd {wsl.shquote(config.WSL_GSV_ROOT)} && "
            # s2_train.py saves its checkpoints in this folder but does not create it.
            f"mkdir -p {wsl.shquote(f'logs/{model_name}/logs_s2_{version}')} && "
            f"{wsl.shquote(config.WSL_PYTHON)} -s GPT_SoVITS/s2_train.py --config {wsl.shquote(config_path)}"
        )
        name = f"SoVITS training: {model_name}" + (f" ({version})" if version != "v2Pro" else "")
    else:
        raise ValueError("Unknown training kind.")
    key = f"train-{kind}"
    proc = jobs.ManagedProcess(name, wsl.wsl_command(script), stop_script=wsl.WSL_STOP_SCRIPTS[key])
    proc.watch = TrainingProgress(kind, epochs)
    with jobs.STATE_LOCK:
        current = jobs.JOBS.get(key)
        if current and current.process and current.process.poll() is None:
            raise RuntimeError(f"{name} is already running.")
        jobs.JOBS[key] = proc
    proc.start()
    jobs.log(f"{name} started from {descriptor['name']}: epochs={epochs}, batch={batch_size}, save_every={save_every}.")
