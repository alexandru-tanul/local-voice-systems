import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import voice_dashboard as dashboard


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="voice test ")))
        for name, value in {
            "WSL_GSV_WIN": self.root, "APPLIO_ROOT": self.root,
            "APPLIO_ENV_PYTHON": self.root / "python.exe", "TMP_DIR": self.root / "tmp",
            "AMD_HIP_BIN": None, "APP_CONFIG": {"tts_device": "cpu"}, "JOBS": {},
        }.items():
            self.enterContext(patch.object(dashboard, name, value))

    def test_configs_match_device_and_precision(self):
        for device, precision, half in [("cpu", "fp16", False), ("cuda", "fp16", True), ("cuda", "fp32", False)]:
            with self.subTest(device=device, precision=precision):
                dashboard.APP_CONFIG.update(tts_device=device, tts_precision=precision)
                dashboard.write_tts_config("base.ckpt", "base.pth")
                dashboard.write_s1_config(1, 1, 1)
                dashboard.write_s2_config(1, 1, 1)
                tts = (self.root / "TEMP/dashboard_tts_infer.yaml").read_text()
                s1 = (self.root / "TEMP/voice-model_s1.yaml").read_text()
                s2 = json.loads((self.root / "TEMP/voice-model_s2.json").read_text())
                self.assertIn(f"device: {device}", tts)
                self.assertIn(f"is_half: {str(half).lower()}", tts)
                self.assertIn('precision: 16-mixed' if half else 'precision: "32"', s1)
                self.assertEqual(s2["train"]["fp16_run"], half)
                self.assertEqual(s2["train"]["gpu_numbers"], "0" if device == "cuda" else "")

    def test_invalid_settings_explain_allowed_values(self):
        dashboard.APP_CONFIG["tts_device"] = "auto"
        with self.assertRaisesRegex(ValueError, "cuda or cpu"):
            dashboard.voice_runtime()

    def test_setup_reports_engine_errors(self):
        result = subprocess.CompletedProcess([], 1, "", "CUDA driver missing")
        with patch.object(dashboard, "run_wsl", return_value=result):
            check = dashboard.setup_status()["gpu"]
        self.assertEqual(check, {"status": "missing", "detail": "CUDA driver missing"})

    def test_setup_reports_wsl_start_errors(self):
        message = "WSL2 is unable to start.\n\nEnable virtualization.\n"
        result = subprocess.CompletedProcess([], 4294967295, "\x00".join(message) + "\x00", "")
        with patch.object(dashboard, "run_wsl", return_value=result):
            check = dashboard.setup_status()["gpu"]
        self.assertEqual(check, {"status": "missing", "detail": "WSL2 is unable to start."})

    def test_pretrained_models_need_no_training(self):
        root = self.root / "GPT_SoVITS/pretrained_models"
        (root / "v2Pro").mkdir(parents=True)
        (root / "s1v3.ckpt").touch()
        (root / "v2Pro/s2Gv2Pro.pth").touch()
        models = dashboard.model_files()
        self.assertEqual(models["gpt"][0]["path"], "GPT_SoVITS/pretrained_models/s1v3.ckpt")
        self.assertEqual(models["sovits"][0]["path"], "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth")

    def test_prepare_includes_v2pro_speaker_features(self):
        descriptor = {"id": "test", "name": "Test", "wsl_root": "/home/test/clips"}
        with (
            patch.object(dashboard, "dataset_descriptor", return_value=descriptor),
            patch.object(dashboard, "dataset_summary", return_value={"ready": True}),
            patch.object(dashboard, "dataset_sync_script", return_value="true"),
            patch.object(dashboard.ManagedProcess, "start"),
        ):
            dashboard.prepare_tts_dataset_job("test", "voice")
        script = dashboard.JOBS["dataset-prepare"].command[-1]
        self.assertIn("2-get-sv.py", script)
        self.assertIn("export sv_path=", script)
        self.assertIn("export is_half=False", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=''", script)

    def test_applio_launch_accepts_spaces_without_amd(self):
        dashboard.APPLIO_ENV_PYTHON.touch()
        with patch.object(dashboard, "realtime_ready", return_value=False), patch.object(dashboard.ManagedProcess, "start"):
            dashboard.start_applio()
        job = dashboard.JOBS["applio"]
        self.assertEqual(job.command[0], str(dashboard.APPLIO_ENV_PYTHON))
        self.assertIn("--client", job.command)
        self.assertEqual(job.env["TMP"], str(dashboard.TMP_DIR))

    def test_applio_cpu_hides_gpu(self):
        dashboard.APP_CONFIG["applio_backend"] = "cpu"
        self.assertEqual(dashboard.applio_environment()["CUDA_VISIBLE_DEVICES"], "-1")

    def test_zluda_launch_uses_wrapper(self):
        dashboard.APP_CONFIG["applio_backend"] = "zluda"
        dashboard.APPLIO_ENV_PYTHON.touch()
        (self.root / "zluda").mkdir()
        (self.root / "zluda/zluda.exe").touch()
        self.assertEqual(dashboard.applio_command("app.py")[1:], ["--", str(dashboard.APPLIO_ENV_PYTHON), "app.py"])

    @unittest.skipIf(os.name == "nt", "Checks the WSL environment in a local bash")
    def test_wsl_preserves_paths_with_spaces_and_quotes(self):
        with patch.object(dashboard, "WSL_PYTHON", "/tmp/user's env/bin/python"):
            script = dashboard.wsl_command('printf "%s\\n" "$PATH" "$LD_LIBRARY_PATH"')[-1]
        result = subprocess.run(["bash", "-c", script], env={"LD_LIBRARY_PATH": "/existing/lib"}, text=True, capture_output=True, check=True)
        paths = result.stdout.splitlines()
        self.assertTrue(paths[0].startswith("/tmp/user's env/bin:"))
        self.assertEqual(paths[1], "/tmp/user's env/lib:/existing/lib")
