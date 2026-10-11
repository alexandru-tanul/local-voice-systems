import base64
import contextlib
import http.client
import io
import json
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import relay_asr_server
import voice_dashboard as dashboard


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="voice test ")))
        for name, value in {
            "WSL_GSV_WIN": self.root, "TMP_DIR": self.root / "tmp",
            "APP_CONFIG": {"tts_device": "cpu"}, "JOBS": {}, "WSL_PATHS_ERROR": "",
        }.items():
            self.enterContext(patch.object(dashboard, name, value))
        self.enterContext(patch.dict(dashboard.SYSTEM_STATE))

    def start_server(self):
        server = dashboard.DashboardServer(("127.0.0.1", 0), dashboard.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def request(self, port, method, path, body="{}", **headers):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request(method, path, body=body if method == "POST" else None, headers=headers)
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
                self.assertNotRegex(dashboard.wsl_bash(job.stop_script)[-1], pattern)

    def test_ports_come_from_config(self):
        self.assertEqual(dashboard.config_port({"asr_port": 9001}, "asr_port", 8792), (9001, ""))
        self.assertEqual(dashboard.config_port({}, "asr_port", 8792), (8792, ""))
        for value in ["port", 0, 70000, True, None]:
            with self.subTest(value=value):
                port, error = dashboard.config_port({"asr_port": value}, "asr_port", 8792)
                self.assertEqual(port, 8792)
                self.assertIn("asr_port must be a port number", error)

    def test_wsl_scripts_reach_bash_unchanged(self):
        # Without --exec, wsl.exe expands a script's own variables before bash sets them.
        command = dashboard.wsl_command("x=1; echo $x")
        self.assertEqual(command[3:6], ["--exec", "bash", "-lc"])

    def test_sync_maps_paths_in_wsl_and_escapes_the_target(self):
        descriptor = {
            "root": Path(r"C:\voice data\clips"),
            "list_path": Path(r"C:\voice data\clips\clips.list"),
            "wsl_root": "/home/me/a&b|c",
        }
        script = dashboard.dataset_sync_script(descriptor)
        self.assertIn("source_dir=$(wslpath -u 'C:\\voice data\\clips')", script)
        self.assertIn("cp -ru", script)
        self.assertIn("sed 's|^wavs/|/home/me/a\\&b\\|c/wavs/|'", script)

    def test_wsl_stop_patterns_do_not_match_the_stop_shell(self):
        for key, script in dashboard.WSL_STOP_SCRIPTS.items():
            with self.subTest(job=key):
                pattern = re.search(r"pkill -f '([^']+)'", script).group(1)
                self.assertNotRegex(dashboard.wsl_bash(script)[-1], pattern)

    def test_stop_buttons_work_after_a_restart(self):
        port = self.start_server()
        with patch.object(dashboard, "stop_in_wsl") as stop_in_wsl:
            status = self.request(port, "POST", "/api/stop-job", body=json.dumps({"key": "train-gpt"}), Host=f"127.0.0.1:{port}")[0]
        self.assertEqual(status, 200)
        stop_in_wsl.assert_called_once_with(dashboard.WSL_STOP_SCRIPTS["train-gpt"])

    @unittest.skipUnless(os.name == "nt", "Stops Windows processes")
    def test_orphan_cleanup_stops_only_this_dashboards_helper(self):
        sleeper = "import time; time.sleep(60)"
        marker = ["ptt_helper.py", "--dashboard", "http://127.0.0.1:18799"]
        quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        processes = {
            "helper": subprocess.Popen([sys.executable, "-c", sleeper, *marker], **quiet),
            "names the file": subprocess.Popen([sys.executable, "-c", sleeper, "ptt_helper.py"], **quiet),
            "other dashboard": subprocess.Popen([sys.executable, "-c", sleeper, *marker[:2], "http://127.0.0.1:187990"], **quiet),
            # PowerShell joins its arguments into one command, and "#" makes the rest a comment.
            "not python": subprocess.Popen(["powershell", "-NoProfile", "-Command", "Start-Sleep 60 #", *marker], **quiet),
        }
        for proc in processes.values():
            self.addCleanup(proc.kill)
        with patch.object(dashboard, "DASHBOARD_PORT", 18799):
            dashboard.kill_orphan_ptt_helpers()
        self.assertIsNotNone(processes["helper"].wait(timeout=10))
        for name in ("names the file", "other dashboard", "not python"):
            with self.subTest(process=name):
                self.assertIsNone(processes[name].poll())

    def test_saved_models_are_used_when_they_exist(self):
        models = {
            "gpt": [{"path": "GPT_weights_v2Pro/joe.ckpt"}, {"path": "GPT_weights_v2Pro/other.ckpt"}],
            "sovits": [{"path": "SoVITS_weights_v2Pro/joe.pth"}],
        }
        with patch.object(dashboard, "model_files", return_value=models):
            self.assertEqual(dashboard.model_pair("GPT_weights_v2Pro/other.ckpt", "SoVITS_weights_v2Pro/joe.pth"),
                             ("GPT_weights_v2Pro/other.ckpt", "SoVITS_weights_v2Pro/joe.pth"))
            # No choice, or a model that was deleted, falls back to the first listed one.
            self.assertEqual(dashboard.model_pair("", "SoVITS_weights_v2Pro/gone.pth"),
                             ("GPT_weights_v2Pro/joe.ckpt", "SoVITS_weights_v2Pro/joe.pth"))
        with patch.object(dashboard, "model_files", return_value={"gpt": [], "sovits": []}):
            self.assertEqual(dashboard.model_pair("GPT_weights_v2Pro/joe.ckpt", ""), ("", ""))

    def test_gpu_jobs_stop_the_voice_first(self):
        running = MagicMock(**{"running.return_value": True})
        with (
            patch.object(dashboard, "stop_system") as stop_system,
            patch.object(dashboard, "stop_api") as stop_api,
            patch.object(dashboard, "api_ready", return_value=True),
            patch.object(dashboard, "API_PROCESS", None),
        ):
            # In CPU mode the voice and training do not share a GPU.
            dashboard.free_gpu_for("training")
            dashboard.JOBS["train-gpt"] = running
            self.assertFalse(dashboard.gpu_busy())
            stop_system.assert_not_called()
            stop_api.assert_not_called()
            dashboard.APP_CONFIG["tts_device"] = "cuda"
            self.assertTrue(dashboard.gpu_busy())
            # A running Mic to Voice stops as a whole, so its status stays true.
            dashboard.SYSTEM_STATE["status"] = "ready"
            dashboard.free_gpu_for("training")
            stop_system.assert_called_once()
            # An engine started from Text to Voice stops alone.
            dashboard.SYSTEM_STATE["status"] = "stopped"
            dashboard.free_gpu_for("training")
            stop_api.assert_called_once()

    def test_a_starting_gpu_job_keeps_the_voice_from_starting_itself(self):
        dashboard.APP_CONFIG["tts_device"] = "cuda"
        port = self.start_server()
        start = lambda auto: json.loads(self.request(port, "POST", "/api/start-system", body=json.dumps({"auto": auto}), Host=f"127.0.0.1:{port}")[2])
        with patch.object(dashboard, "start_system") as start_system:
            with dashboard.gpu_reserved():
                self.assertTrue(dashboard.gpu_busy())
                self.assertEqual(start(True), {"ok": True, "started": False})
                start_system.assert_not_called()
                # Start pressed on purpose still loads the voice.
                self.assertEqual(start(False), {"ok": True, "started": True})
                start_system.assert_called_once()
        self.assertFalse(dashboard.gpu_busy())

    def test_a_running_job_fails_before_it_stops_the_voice(self):
        running = MagicMock(**{"running.return_value": True})
        dashboard.JOBS.update({"train-gpt": running, "dataset-prepare": running})
        with (
            patch.object(dashboard, "dataset_descriptor", return_value={"name": "Test"}),
            patch.object(dashboard, "dataset_summary", return_value={"ready": True}),
            patch.object(dashboard, "free_gpu_for") as free_gpu_for,
        ):
            with self.assertRaisesRegex(RuntimeError, "GPT training is already running"):
                dashboard.start_train("gpt", 1, 1, 1, "test")
            with self.assertRaisesRegex(RuntimeError, "Dataset preparation is already running"):
                dashboard.prepare_tts_dataset_job("test", "voice")
        free_gpu_for.assert_not_called()

    def test_training_frees_the_gpu_only_when_it_can_start(self):
        missing = subprocess.CompletedProcess([], 1, "", "")
        with (
            patch.object(dashboard, "dataset_descriptor", return_value={"name": "Test"}),
            patch.object(dashboard, "run_wsl", return_value=missing),
            patch.object(dashboard, "free_gpu_for") as free_gpu_for,
        ):
            with self.assertRaisesRegex(RuntimeError, "Training features are missing"):
                dashboard.start_train("gpt", 1, 1, 1, "test")
            free_gpu_for.assert_not_called()
            with (
                patch.object(dashboard, "run_wsl", return_value=subprocess.CompletedProcess([], 0, "", "")),
                patch.object(dashboard.ManagedProcess, "start"),
            ):
                dashboard.start_train("gpt", 1, 1, 1, "test")
            free_gpu_for.assert_called_once_with("training")

    def test_stop_skips_wsl_when_the_distro_is_stopped(self):
        with (
            patch.object(dashboard, "wsl_distro_running", return_value=False),
            patch.object(dashboard, "wsl_bash") as wsl_bash,
            patch.object(dashboard, "relay_asr_ready", return_value=False),
            patch.object(dashboard, "kill_orphan_ptt_helpers"),
            patch.object(dashboard, "API_PROCESS", None),
            patch.object(dashboard, "API_URL", "http://127.0.0.1:9"),
        ):
            dashboard.stop_system()
        wsl_bash.assert_not_called()
        self.assertEqual(dashboard.SYSTEM_STATE["status"], "stopped")

    def test_stop_during_startup_cancels_it(self):
        def slow_wsl(script, **kwargs):
            time.sleep(0.5)
            return subprocess.CompletedProcess([], 0, "", "")

        with (
            patch.object(dashboard, "run_wsl", slow_wsl),
            patch.object(dashboard, "model_files", return_value={"gpt": [{"path": "g"}], "sovits": [{"path": "s"}]}),
            patch.object(dashboard, "start_relay_asr") as start_asr,
            patch.object(dashboard, "relay_asr_ready", return_value=False),
            patch.object(dashboard, "stop_api"),
            patch.object(dashboard, "kill_orphan_ptt_helpers"),
        ):
            dashboard.start_system("", "")
            with self.assertRaisesRegex(RuntimeError, "already starting"):
                dashboard.start_system("", "")
            dashboard.stop_system()
            dashboard.SYSTEM_THREAD.join(timeout=5)
            self.assertEqual(dashboard.SYSTEM_STATE["status"], "stopped")
            start_asr.assert_not_called()
            # Start is accepted again; stop that run too before the patches end.
            dashboard.start_system("", "")
            dashboard.stop_system()
            dashboard.SYSTEM_THREAD.join(timeout=5)

    def test_startup_stops_waiting_when_the_process_exits(self):
        proc = dashboard.ManagedProcess("test engine", [sys.executable, "-c", "print('CUDA driver missing'); raise SystemExit(3)"])
        proc.start()
        started = time.perf_counter()
        with self.assertRaisesRegex(RuntimeError, r"exited with code 3[\s\S]*CUDA driver missing"):
            dashboard.wait_until(lambda: False, 30, dashboard.SYSTEM_RUN_ID, "Voice engine", proc)
        self.assertLess(time.perf_counter() - started, 10)

    def test_speech_server_that_is_still_starting_counts_as_started(self):
        starting = MagicMock()
        starting.running.return_value = True
        dashboard.JOBS["relay-asr"] = starting
        python = self.root / "python.exe"
        python.touch()
        with (
            patch.object(dashboard, "relay_asr_ready", return_value=False),
            patch.object(dashboard, "RELAY_ASR_PYTHON", python),
            patch.object(dashboard, "RELAY_ASR_MODEL_DIR", self.root / "models"),
            patch.object(dashboard, "RELAY_ASR_TMP", self.root / "asr-tmp"),
        ):
            self.assertTrue(dashboard.start_relay_asr())
        self.assertIs(dashboard.JOBS["relay-asr"], starting)

    def test_speech_model_errors_reach_the_dashboard(self):
        server = relay_asr_server.Server(("127.0.0.1", 0), relay_asr_server.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with (
            patch.object(relay_asr_server, "load_model", side_effect=OSError("download failed")),
            patch.object(dashboard, "RELAY_ASR_URL", f"http://127.0.0.1:{server.server_address[1]}"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not load the .* speech model: download failed"):
                dashboard.warm_up_relay_asr()

    def test_a_second_server_cannot_share_the_port(self):
        port = self.start_server()
        with self.assertRaises(OSError):
            dashboard.DashboardServer(("127.0.0.1", port), dashboard.Handler)

    def test_closed_connections_write_no_traceback(self):
        outputs = self.root / "outputs"
        outputs.mkdir()
        (outputs / "big.wav").write_bytes(bytes(8 * 1024 * 1024))
        self.enterContext(patch.object(dashboard, "OUTPUT_DIR", outputs))
        port = self.start_server()
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            client = socket.create_connection(("127.0.0.1", port))
            client.sendall(f"GET /outputs/big.wav HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
            client.recv(1024)
            # A zero linger time resets the connection, as a browser does when it cancels a download.
            client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            client.close()
            time.sleep(1)
        self.assertNotIn("Traceback", errors.getvalue())

    def test_generated_files_get_distinct_names(self):
        with patch.object(dashboard, "datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 10, 12, 0, 0)
            first = dashboard.new_output_path(self.root, "Voice")
            first.touch()
            second = dashboard.new_output_path(self.root, "Voice")
        self.assertNotEqual(first, second)

    def test_only_the_newest_relay_lines_are_kept(self):
        relay = self.root / "relay"
        relay.mkdir()
        self.enterContext(patch.object(dashboard, "RELAY_OUTPUT_DIR", relay))
        for index in range(5):
            path = relay / f"line{index}.wav"
            path.touch()
            os.utime(path, (1000 + index, 1000 + index))
        dashboard.prune_relay_outputs(keep=3)
        self.assertEqual(sorted(path.name for path in relay.iterdir()), ["line2.wav", "line3.wav", "line4.wav"])

    def test_dataset_clips_get_one_row_and_distinct_names(self):
        datasets = self.root / "datasets"
        outputs = self.root / "outputs"
        datasets.mkdir()
        outputs.mkdir()
        self.enterContext(patch.object(dashboard, "DATASETS_DIR", datasets))
        self.enterContext(patch.object(dashboard, "OUTPUT_DIR", outputs))
        self.enterContext(patch.object(dashboard, "convert_audio", lambda source, target: Path(target).write_bytes(b"RIFF")))
        dataset_id = dashboard.create_dataset("Test Voice")["id"]
        audio = base64.b64encode(b"audio").decode("ascii")
        dashboard.add_dataset_audio(dataset_id, "a b.mp3", audio, "First line\nwith a break | and a bar")
        summary = dashboard.add_dataset_audio(dataset_id, "a-b.wav", audio, "Second line")
        root = datasets / dataset_id
        rows = (root / f"{dataset_id}.list").read_bytes().decode("utf-8").split("\n")
        self.assertEqual(rows, ["wavs/a-b.wav|speaker|en|First line with a break and a bar", "wavs/a-b-2.wav|speaker|en|Second line", ""])
        self.assertTrue(summary["ready"])
        (root / "wavs/stray.wav").touch()
        self.assertTrue(dashboard.dataset_summary(dashboard.dataset_descriptor(dataset_id))["ready"])
        (root / "wavs/a-b.wav").unlink()
        self.assertFalse(dashboard.dataset_summary(dashboard.dataset_descriptor(dataset_id))["ready"])
        self.assertEqual(list(outputs.iterdir()), [])

    def test_dataset_names_without_latin_letters_get_distinct_ids(self):
        first = dashboard.dataset_id_for_name("駅員")
        self.assertEqual(first, dashboard.dataset_id_for_name(" 駅員 "))
        self.assertNotEqual(first, dashboard.dataset_id_for_name("車掌"))
        self.assertEqual(dashboard.slugify(first), first)
        self.assertEqual(dashboard.dataset_id_for_name("Station Announcer"), "station-announcer")

    def test_config_errors_are_reported(self):
        path = self.root / "config.json"
        path.write_text('{"asr_model_dir": "C:\\models"}', encoding="utf-8")
        config, error = dashboard.load_config(path)
        self.assertEqual(config, {})
        self.assertIn("could not be read", error)
        path.write_text('\ufeff{"asr_model_dir": "C:/models"}', encoding="utf-8")
        self.assertEqual(dashboard.load_config(path), ({"asr_model_dir": "C:/models"}, ""))
        self.assertIn("Copy config.example.json", dashboard.load_config(self.root / "none.json")[1])

    def test_wsl_paths_have_no_defaults(self):
        paths = {
            "wsl_gpt_sovits_root": "/home/me/GPT-SoVITS",
            "wsl_python": "/home/me/miniforge3/envs/gsv/bin/python",
            "wsl_datasets_root": "/home/me/voice-datasets",
        }
        self.assertEqual(dashboard.wsl_paths_error(paths), "")
        self.assertIn("wsl_python, wsl_datasets_root", dashboard.wsl_paths_error({"wsl_gpt_sovits_root": "/x"}))
        self.assertIn("wsl_python", dashboard.wsl_paths_error({**paths, "wsl_python": "gsv/bin/python"}))
        self.assertIn("replace YOUR_WSL_USER", dashboard.wsl_paths_error({**paths, "wsl_python": "/home/YOUR_WSL_USER/py"}))
        # Nothing runs in WSL or writes to it without the paths, but jobs can still be stopped.
        with (
            patch.object(dashboard, "WSL_PATHS_ERROR", "Set the paths."),
            patch.object(dashboard, "wsl_distro_running", return_value=True),
            patch.object(dashboard.subprocess, "run") as run,
        ):
            for action in (lambda: dashboard.wsl_command("true"), dashboard.voice_runtime):
                with self.assertRaisesRegex(ValueError, "Set the paths."):
                    action()
            dashboard.stop_in_wsl(dashboard.WSL_STOP_SCRIPTS["api"])
        self.assertEqual(run.call_args[0][0][-1], dashboard.WSL_STOP_SCRIPTS["api"])

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

    def test_port_checks_return_at_once(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertTrue(dashboard.port_open(port))
        started = time.perf_counter()
        self.assertFalse(dashboard.port_open(port))
        self.assertLess(time.perf_counter() - started, 0.1)

    @unittest.skipIf(os.name == "nt", "Checks the WSL environment in a local bash")
    def test_wsl_preserves_paths_with_spaces_and_quotes(self):
        with patch.object(dashboard, "WSL_PYTHON", "/tmp/user's env/bin/python"):
            script = dashboard.wsl_command('printf "%s\\n" "$PATH" "$LD_LIBRARY_PATH"')[-1]
        result = subprocess.run(["bash", "-c", script], env={"LD_LIBRARY_PATH": "/existing/lib"}, text=True, capture_output=True, check=True)
        paths = result.stdout.splitlines()
        self.assertTrue(paths[0].startswith("/tmp/user's env/bin:"))
        self.assertEqual(paths[1], "/tmp/user's env/lib:/existing/lib")
