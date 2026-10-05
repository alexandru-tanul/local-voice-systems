import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import voice_dashboard as dashboard


CPU = {"device": "cpu", "is_half": False, "gpu_index": "", "detail": "CPU"}
GPU = {"device": "cuda", "is_half": True, "gpu_index": "1", "detail": "RTX"}


class DashboardTests(unittest.TestCase):
    def test_new_install_can_write_cpu_configs(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(dashboard, "WSL_GSV_WIN", Path(folder)), patch.object(dashboard, "voice_runtime", return_value=CPU):
            dashboard.write_tts_config("base.ckpt", "base.pth")
            dashboard.write_s1_config(1, 1, 1)
            dashboard.write_s2_config(1, 1, 1)
            root = Path(folder) / "TEMP"
            tts = (root / "dashboard_tts_infer.yaml").read_text()
            self.assertIn("device: cpu", tts)
            self.assertIn("is_half: false", tts)
            self.assertIn('precision: "32"', (root / "voice-model_s1.yaml").read_text())
            s2 = json.loads((root / "voice-model_s2.json").read_text())
            self.assertFalse(s2["train"]["fp16_run"])
            self.assertEqual(s2["train"]["gpu_numbers"], "")

    def test_gpu_training_uses_selected_card(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(dashboard, "WSL_GSV_WIN", Path(folder)), patch.object(dashboard, "voice_runtime", return_value=GPU):
            dashboard.write_s2_config(1, 1, 1)
            config = json.loads((Path(folder) / "TEMP/voice-model_s2.json").read_text())
            self.assertTrue(config["train"]["fp16_run"])
            self.assertEqual(config["train"]["gpu_numbers"], "1")
            self.assertIn("CUDA_VISIBLE_DEVICES='1'", dashboard.voice_environment())

    def test_pretrained_models_are_available_without_training(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(dashboard, "WSL_GSV_WIN", Path(folder)):
            root = Path(folder) / "GPT_SoVITS/pretrained_models"
            (root / "v2Pro").mkdir(parents=True)
            (root / "s1v3.ckpt").write_bytes(b"gpt")
            (root / "v2Pro/s2Gv2Pro.pth").write_bytes(b"sovits")
            models = dashboard.model_files()
            self.assertEqual(models["gpt"][0]["path"], "GPT_SoVITS/pretrained_models/s1v3.ckpt")
            self.assertEqual(models["sovits"][0]["path"], "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth")

    def test_prepare_includes_v2pro_speaker_features(self):
        descriptor = {"id": "test", "name": "Test", "wsl_root": "/home/test/clips"}
        with patch.object(dashboard, "dataset_descriptor", return_value=descriptor), patch.object(dashboard, "dataset_summary", return_value={"ready": True}), patch.object(dashboard, "write_s2_config", return_value="/tmp/s2.json"), patch.object(dashboard, "voice_runtime", return_value=CPU), patch.object(dashboard, "dataset_sync_script", return_value="true"), patch.object(dashboard.ManagedProcess, "start"), patch.object(dashboard, "JOBS", {}):
            dashboard.prepare_tts_dataset_job("test", "test-voice")
            script = dashboard.JOBS["dataset-prepare"].command[-1]
            self.assertIn("2-get-sv.py", script)
            self.assertIn("export sv_path=", script)
            self.assertIn("export is_half=False", script)
            self.assertIn("export CUDA_VISIBLE_DEVICES=''", script)

    def test_applio_launch_accepts_spaces_without_amd(self):
        with tempfile.TemporaryDirectory(prefix="voice test ") as folder:
            python = Path(folder) / "env/python.exe"
            python.parent.mkdir()
            python.touch()
            with patch.object(dashboard, "APPLIO_ENV_PYTHON", python), patch.object(dashboard, "APPLIO_ROOT", Path(folder)), patch.object(dashboard, "TMP_DIR", Path(folder) / "tmp"), patch.object(dashboard, "APP_CONFIG", {}), patch.object(dashboard, "AMD_HIP_BIN", None), patch.object(dashboard, "realtime_ready", return_value=False), patch.object(dashboard.ManagedProcess, "start"), patch.object(dashboard, "JOBS", {}), patch.dict(os.environ, {}, clear=True):
                dashboard.start_applio()
                job = dashboard.JOBS["applio"]
                self.assertEqual(job.command[0], str(python))
                self.assertIn("--client", job.command)
                self.assertNotIn("HIP_VISIBLE_DEVICES", job.env)
                self.assertEqual(job.env["TMP"], str(Path(folder) / "tmp"))

    def test_applio_cpu_hides_gpu(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(dashboard, "TMP_DIR", Path(folder)), patch.object(dashboard, "APP_CONFIG", {"applio_backend": "cpu"}):
            self.assertEqual(dashboard.applio_environment()["CUDA_VISIBLE_DEVICES"], "-1")

    def test_zluda_launch_uses_wrapper(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(dashboard, "APPLIO_ROOT", Path(folder)), patch.object(dashboard, "APP_CONFIG", {"applio_backend": "zluda"}):
            python = Path(folder) / "python.exe"
            python.touch()
            (Path(folder) / "zluda").mkdir()
            (Path(folder) / "zluda/zluda.exe").touch()
            with patch.object(dashboard, "APPLIO_ENV_PYTHON", python):
                command = dashboard.applio_command("app.py")
                self.assertEqual(command[1:], ["--", str(python), "app.py"])

    @unittest.skipIf(os.name == "nt", "Uses a local bash to check WSL shell quoting")
    def test_wsl_path_handles_spaces_and_quotes(self):
        with patch.object(dashboard, "WSL_PYTHON", "/tmp/voice user's env/bin/python"):
            script = dashboard.wsl_command('printf "%s" "$PATH"')[-1]
            result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=True)
            self.assertTrue(result.stdout.startswith("/tmp/voice user's env/bin:"))

    @unittest.skipIf(os.name == "nt", "Uses a local bash to check the library search path")
    def test_wsl_finds_conda_audio_libraries(self):
        with patch.object(dashboard, "WSL_PYTHON", "/tmp/voice env/bin/python"):
            script = dashboard.wsl_command('printf "%s" "$LD_LIBRARY_PATH"')[-1]
            result = subprocess.run(["bash", "-c", script], env={"LD_LIBRARY_PATH": "/existing/lib"}, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout, "/tmp/voice env/lib:/existing/lib")


if __name__ == "__main__":
    unittest.main()
