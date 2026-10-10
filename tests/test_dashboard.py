import http.client
import json
import os
import re
import socket
import subprocess
import tempfile
import threading
import time
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

    def start_server(self):
        server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def request(self, port, method, path, **headers):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request(method, path, body="{}" if method == "POST" else None, headers=headers)
            response = connection.getresponse()
            return response.status, response.getheader("Content-Range"), response.read()
        finally:
            connection.close()

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

    def test_dataset_stop_scripts_match_their_job_but_not_themselves(self):
        descriptor = {"id": "test", "name": "Test", "wsl_root": "/home/test/clips"}
        with (
            patch.object(dashboard, "dataset_descriptor", return_value=descriptor),
            patch.object(dashboard, "dataset_summary", return_value={"ready": True}),
            patch.object(dashboard, "dataset_sync_script", return_value="true"),
            patch.object(dashboard.ManagedProcess, "start"),
        ):
            dashboard.sync_dataset_job("test")
            dashboard.prepare_tts_dataset_job("test", "voice")
        for key in ("dataset-sync", "dataset-prepare"):
            with self.subTest(job=key):
                job = dashboard.JOBS[key]
                pattern = re.search(r"pkill -f '([^']+)'", job.stop_script).group(1)
                self.assertRegex(job.command[-1], pattern)
                self.assertNotRegex(dashboard.wsl_command(job.stop_script)[-1], pattern)

    def test_rvc_process_pattern_ignores_other_programs(self):
        # PowerShell's -match ignores case, so the test does too.
        pattern = re.compile(dashboard.rvc_process_pattern("voice"), re.IGNORECASE)
        for command in [
            r'"C:\Python312\python.exe" voice_dashboard.py',
            r'"C:\local-voice-systems\relay_env\Scripts\python.exe" C:\local-voice-systems\relay_asr_server.py',
            r"C:\Applio\env\python.exe rvc\train\extract\extract.py C:\Applio\logs\voice2 rmvpe",
            r"C:\Applio\zluda\zluda.exe -- C:\Applio\env\python.exe app.py --client",
        ]:
            self.assertIsNone(pattern.search(command), command)
        for command in [
            r"C:\Applio\env\python.exe -u rvc/train/train.py voice 10 50",
            r'"C:\My Apps\Applio\env\python.exe" rvc\train\preprocess\preprocess.py "C:\My Apps\Applio\logs\voice" 32000',
            r"C:\Applio\env\python.exe -u rvc/train/process/extract_index.py logs/voice Auto",
        ]:
            self.assertIsNotNone(pattern.search(command), command)

    def test_dataset_names_without_latin_letters_get_distinct_ids(self):
        first = dashboard.dataset_id_for_name("駅員")
        self.assertEqual(first, dashboard.dataset_id_for_name(" 駅員 "))
        self.assertNotEqual(first, dashboard.dataset_id_for_name("車掌"))
        self.assertEqual(dashboard.slugify(first), first)
        self.assertEqual(dashboard.dataset_id_for_name("Station Announcer"), "station-announcer")

    def test_config_errors_are_reported(self):
        path = self.root / "config.json"
        path.write_text('{"applio_root": "C:\\Applio"}', encoding="utf-8")
        config, error = dashboard.load_config(path)
        self.assertEqual(config, {})
        self.assertIn("default paths", error)
        path.write_text('\ufeff{"applio_root": "C:/Applio"}', encoding="utf-8")
        self.assertEqual(dashboard.load_config(path), ({"applio_root": "C:/Applio"}, ""))

    def test_server_rejects_requests_from_other_sites(self):
        port = self.start_server()

        def status(method, path, **headers):
            return self.request(port, method, path, **headers)[0]

        self.assertEqual(status("GET", "/api/ptt-state", Host=f"127.0.0.1:{port}"), 200)
        self.assertEqual(status("GET", "/api/ptt-state", Host=f"rebound.example:{port}"), 403)
        self.assertEqual(status("POST", "/api/cancel-generation", Host=f"localhost:{port}", Origin="http://other.example"), 403)
        self.assertEqual(status("POST", "/api/cancel-generation", Host=f"localhost:{port}", Origin=f"http://localhost:{port}"), 200)

    def test_byte_ranges_follow_http(self):
        outputs = self.root / "outputs"
        outputs.mkdir()
        (outputs / "clip.wav").write_bytes(b"0123456789")
        self.enterContext(patch.object(dashboard, "OUTPUT_DIR", outputs))
        port = self.start_server()
        for header, expected in [
            ("bytes=2-4", (206, "bytes 2-4/10", b"234")),
            ("bytes=6-", (206, "bytes 6-9/10", b"6789")),
            ("bytes=-3", (206, "bytes 7-9/10", b"789")),
            ("bytes=-50", (206, "bytes 0-9/10", b"0123456789")),
        ]:
            with self.subTest(range=header):
                self.assertEqual(self.request(port, "GET", "/outputs/clip.wav", Host=f"127.0.0.1:{port}", Range=header), expected)

    def test_rvc_new_run_trains_the_requested_epochs(self):
        model_dir = self.root / "logs/voice"
        model_dir.mkdir(parents=True)
        (model_dir / "filelist.txt").write_text("row\n")
        (model_dir / "voice_50e_1000s.pth").touch()
        pretrained = self.root / "rvc/models/pretraineds/hifi-gan"
        pretrained.mkdir(parents=True)
        (pretrained / "f0G32k.pth").touch()
        (pretrained / "f0D32k.pth").touch()
        dashboard.APPLIO_ENV_PYTHON.touch()

        def target_epoch(fresh):
            with patch.object(dashboard.ManagedProcess, "start"):
                dashboard.start_rvc_training("voice", epochs=50, save_every=10, fresh=fresh)
            command = dashboard.JOBS["train-rvc"].command
            return command[command.index("rvc/train/train.py") + 3]

        self.assertEqual(target_epoch(fresh=True), "50")
        # Without G_/D_ checkpoints Applio cannot resume and starts at epoch 1.
        self.assertEqual(target_epoch(fresh=False), "50")
        (model_dir / "G_2333333.pth").touch()
        self.assertEqual(target_epoch(fresh=False), "100")

    def test_port_checks_return_at_once(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertTrue(dashboard.port_open(port))
        started = time.perf_counter()
        self.assertFalse(dashboard.port_open(port))
        self.assertLess(time.perf_counter() - started, 0.1)

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
