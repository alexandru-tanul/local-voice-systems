import base64
import contextlib
import http.client
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
import zipfile
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
            # Cleanups run last first: the process is ended, then waited for.
            self.addCleanup(proc.wait)
            self.addCleanup(proc.kill)
        with patch.object(dashboard, "DASHBOARD_PORT", 18799):
            dashboard.kill_orphan_ptt_helpers()
        self.assertIsNotNone(processes["helper"].wait(timeout=10))
        for name in ("names the file", "other dashboard", "not python"):
            with self.subTest(process=name):
                self.assertIsNone(processes[name].poll())

    @unittest.skipUnless(os.name == "nt", "Stops Windows processes")
    def test_orphan_cleanup_stops_only_this_dashboards_restore_run(self):
        sleeper = "import time; time.sleep(60)"
        quiet = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        job = lambda port: str(self.root / f"restore-{port}-20261012-120000-abcd.json")
        processes = {
            "worker": subprocess.Popen([sys.executable, "-c", sleeper, "restore_audio.py", job(18799)], **quiet),
            "other dashboard": subprocess.Popen([sys.executable, "-c", sleeper, "restore_audio.py", job(18798)], **quiet),
            "names the file": subprocess.Popen([sys.executable, "-c", sleeper, "restore_audio.py"], **quiet),
        }
        for proc in processes.values():
            # Cleanups run last first: the process is ended, then waited for.
            self.addCleanup(proc.wait)
            self.addCleanup(proc.kill)
        with patch.object(dashboard, "DASHBOARD_PORT", 18799), patch.object(dashboard, "TMP_DIR", self.root):
            dashboard.kill_orphan_restore_workers()
        self.assertIsNotNone(processes["worker"].wait(timeout=10))
        for name in ("other dashboard", "names the file"):
            with self.subTest(process=name):
                self.assertIsNone(processes[name].poll())

    def test_restore_settings_follow_lavasr_and_explain_mistakes(self):
        self.assertEqual(
            dashboard.restore_settings({"denoise": True, "enhance": False, "quality": "16000", "keep_below": "6000", "model": "v1", "pieces": True}),
            {"music": "", "echo": False, "denoise": True, "enhance": False, "model": "v1", "quality": 16000, "keep_below": 6000.0, "pieces": True},
        )
        self.assertEqual(dashboard.restore_settings({"enhance": True})["quality"], "detect")
        self.assertEqual(dashboard.restore_settings({"music": "main", "echo": 1})["music"], "main")
        for settings, message in [
            ({}, "at least one thing"),
            ({"music": "karaoke"}, "separation model"),
            ({"enhance": True, "model": "v3"}, "version 1 or 2"),
            ({"enhance": True, "quality": "12345"}, "from the list"),
            ({"enhance": True, "keep_below": 500}, "1 to 24 kHz"),
        ]:
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, message):
                dashboard.restore_settings(settings)

    def restore_setup(self, rate=22050):
        """Restore audio in a temporary folder, with uploads converted without ffmpeg, and every
        program a run starts replaced: calls lists them, and WSL separation writes its results."""
        def fake_convert(source, target, sample_rate=44100):
            self.assertIsNone(sample_rate)
            frames = rate if b"sound" in Path(source).read_bytes() else 0
            with wave.open(str(target), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(b"\x01\x00" * frames)

        calls = []

        def fake_run(job, command, timeout=None):
            Path(command[-1]).write_bytes(b"")
            return subprocess.CompletedProcess(command, 0, "", "")

        def fake_stream(job, command, env=None, stop_script=None):
            if stop_script:
                # separate_audio.py in WSL: a voice file for each recording.
                spec = json.loads(Path(re.search(r"'([^']*separate\.json)'", command[-1]).group(1)).read_text(encoding="utf-8"))
                calls.append(("separate", spec, stop_script, job.uses_gpu))
                for item in spec["items"]:
                    self.write_wav(Path(item["output"]), 1.0)
            else:
                calls.append(("lavasr", command, json.loads(Path(command[2]).read_text(encoding="utf-8"))))
            return 0

        self.enterContext(patch.object(dashboard, "RESTORE_DIR", self.root / "restored"))
        self.enterContext(patch.object(dashboard, "OUTPUT_DIR", self.root))
        self.enterContext(patch.object(dashboard, "TMP_DIR", self.root / "tmp"))
        self.enterContext(patch.object(dashboard, "convert_audio", fake_convert))
        self.enterContext(patch.object(dashboard, "RESTORE_PYTHON", Path(sys.executable)))
        self.enterContext(patch.object(dashboard.ThreadJob, "run", fake_run))
        self.enterContext(patch.object(dashboard.ThreadJob, "stream", fake_stream))
        self.enterContext(patch.object(dashboard, "free_gpu_for"))
        return calls

    def run_restore_job(self, item_ids, settings):
        results = dashboard.start_restore(item_ids, settings)
        job = dashboard.JOBS["restore"]
        job.thread.join(timeout=30)
        return results, job

    def test_restore_uploads_keep_their_rate_and_runs_reach_the_worker(self):
        calls = self.restore_setup()
        item = dashboard.add_restore_audio("Old Call.mp3", base64.b64encode(b"sound").decode())
        self.assertEqual((item["name"], item["sample_rate"], item["duration"]), ("Old Call", 22050, 1.0))
        self.assertEqual([row["id"] for row in dashboard.restore_items()], [item["id"]])
        with self.assertRaisesRegex(RuntimeError, "at least one recording"):
            dashboard.start_restore(["missing"], {"enhance": True})
        results, job = self.run_restore_job([item["id"]], {"denoise": True, "enhance": True})
        kind, command, spec = calls[-1]
        self.assertEqual((kind, command[:2]), ("lavasr", [sys.executable, str(dashboard.ROOT / "restore_audio.py")]))
        self.assertEqual(spec["items"][0]["path"], str(self.root / "restored" / item["file"]))
        self.assertEqual(spec["items"][0]["out_id"], results[item["id"]])
        self.assertTrue(spec["settings"]["denoise"] and spec["settings"]["enhance"])
        # The replaced worker wrote no result, so the run says so.
        self.assertEqual((job.status, job.lines[-1]), ("failed", "Failed: 0 of 1 file(s) restored; 1 failed."))
        self.assertEqual(list((self.root / "tmp").iterdir()), [])
        with patch.object(dashboard, "RESTORE_PYTHON", self.root / "missing.exe"):
            with self.assertRaisesRegex(RuntimeError, "Setup-Restore-Env.bat"):
                dashboard.start_restore([item["id"]], {"enhance": True})
            # Music removal runs in WSL, so it needs no restore_env.
            self.run_restore_job([item["id"]], {"music": "best"})
        dashboard.delete_restore_item(item["id"])
        self.assertEqual([row["made_from"] for row in dashboard.restore_items()], [item["id"]])

    def test_music_and_echo_removal_run_in_wsl_before_lavasr(self):
        calls = self.restore_setup()
        item = dashboard.add_restore_audio("Stream.mp3", base64.b64encode(b"sound").decode())
        results, job = self.run_restore_job([item["id"]], {"music": "main", "echo": True})
        self.assertEqual(job.status, "done", job.lines)
        kind, spec, stop_script, uses_gpu = calls[0]
        self.assertEqual((kind, spec["steps"], stop_script, uses_gpu), ("separate", ["main", "echo"], dashboard.WSL_STOP_SCRIPTS["separate"], True))
        self.assertFalse(job.uses_gpu)
        made = next(row for row in dashboard.restore_items() if row["id"] == results[item["id"]])
        self.assertEqual((made["made_from"], made["steps"], made["sample_rate"], made["duration"]),
                         (item["id"], ["music and backing voices removed (HP5), echo and reverb removed"], 44100, 1.0))
        # With noise removal too, LavaSR works on the voice that separation left.
        self.run_restore_job([item["id"]], {"music": "best", "denoise": True})
        kind, command, spec = calls[-1]
        self.assertEqual(kind, "lavasr")
        self.assertTrue(spec["items"][0]["path"].endswith("-voice.wav"))
        self.assertEqual(spec["items"][0]["steps"], ["music removed (BS-Roformer)"])
        with patch.object(dashboard, "job_running", side_effect=lambda key: key == "train-sovits"):
            with self.assertRaisesRegex(RuntimeError, "training is using"):
                dashboard.start_restore([item["id"]], {"echo": True})

    def test_restore_results_of_one_run_never_share_a_file(self):
        calls = self.restore_setup(rate=16000)
        with self.assertRaisesRegex(RuntimeError, "no sound"):
            dashboard.add_restore_audio("empty.wav", base64.b64encode(b"empty").decode())
        self.assertEqual(list((self.root / "restored").iterdir()), [])
        # A result keeps its recording's name, so a recording and its result share one.
        first = dashboard.add_restore_audio("clip.wav", base64.b64encode(b"sound").decode())
        second = dashboard.add_restore_audio("clip.wav", base64.b64encode(b"sound").decode())
        self.assertNotEqual(first["id"], second["id"])
        # A stopped run leaves its work folder and a result it was still writing.
        (self.root / "tmp").mkdir()
        leftovers = [self.root / "tmp" / f"restore-{dashboard.DASHBOARD_PORT}-old", self.root / "restored" / "clip-old.wav.part"]
        leftovers[0].mkdir()
        leftovers[1].write_bytes(b"")
        results, job = self.run_restore_job([first["id"], second["id"]], {"enhance": True})
        spec = calls[-1][2]
        self.assertEqual(results, {item["id"]: item["out_id"] for item in spec["items"]})
        self.assertEqual(len(set(results.values())), 2)
        self.assertFalse(any(path.exists() for path in leftovers))

    @unittest.skipUnless(os.name == "nt", "Windows refuses to delete an open file")
    def test_deleting_a_recording_waits_for_a_player_to_let_go(self):
        restored = self.root / "restored"
        restored.mkdir()
        (restored / "clip.wav").write_bytes(b"RIFF")
        (restored / "clip.json").write_text(json.dumps({"id": "clip", "name": "clip", "file": "clip.wav"}), encoding="utf-8")
        with patch.object(dashboard, "RESTORE_DIR", restored):
            # As the server does while it still sends the file to a player.
            sending = (restored / "clip.wav").open("rb")
            threading.Timer(0.5, sending.close).start()
            dashboard.delete_restore_item("clip")
            self.assertEqual(list(restored.iterdir()), [])

    def test_a_stopped_job_reports_stopped_and_one_that_cannot_start_reports_failed(self):
        for attempt in range(5):
            with self.subTest(attempt=attempt):
                proc = dashboard.ManagedProcess("Sleeper", [sys.executable, "-c", "import time; time.sleep(30)"])
                proc.start()
                proc.stop()
                proc.process.wait(timeout=10)
                for _ in range(50):
                    if proc.status != "stopping":
                        break
                    time.sleep(0.05)
                time.sleep(0.2)
                self.assertEqual(proc.snapshot()["status"], "stopped")
        missing = dashboard.ManagedProcess("Missing", [str(self.root / "missing.exe")])
        with self.assertRaises(OSError):
            missing.start()
        self.assertEqual(missing.snapshot()["status"], "failed")

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("librosa", "torch")), "Needs restore_env's audio libraries")
    def test_detect_finds_where_a_recording_stops(self):
        import librosa
        import numpy as np

        import restore_audio

        rng = np.random.default_rng(0)
        sr = 44100
        # Noise shaped like speech: strongest at low frequencies, fading towards the top.
        spectrum = np.fft.rfft(rng.standard_normal(sr * 3))
        freqs = np.fft.rfftfreq(sr * 3, 1 / sr)
        speech = np.fft.irfft(spectrum / (1 + freqs / 1000), n=sr * 3)
        fade = np.ones_like(speech)
        fade[:441] = fade[-441:][::-1] = np.linspace(0, 1, 441)
        resample = lambda audio, src, dst: librosa.resample(audio, orig_sr=src, target_sr=dst, res_type="soxr_hq")
        for rate, expected in [(8000, 4000), (16000, 8000), (32000, 16000), (None, sr / 2)]:
            # Made at a lower rate and saved at 44.1 kHz, as an old recording; faded in and out, or
            # cut out of a longer recording mid-sound.
            audio = resample(resample(speech, sr, rate), rate, sr)[: len(speech)] if rate else speech
            audio = (audio / np.abs(audio).max() * 0.5).astype(np.float32)
            for ends, clip in [("faded", audio * fade), ("cut", audio[sr // 3 :])]:
                with self.subTest(rate=rate, ends=ends):
                    self.assertEqual(restore_audio.detect_band(clip, sr), expected)
        self.assertEqual(restore_audio.detect_band(np.zeros(100, dtype=np.float32), sr), sr / 2)

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

    def test_the_page_and_its_files_are_served(self):
        port = self.start_server()

        def get(path):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                connection.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
                response = connection.getresponse()
                return response.status, response.getheader("Content-Type"), response.read()
            finally:
                connection.close()

        def text(name):
            return (dashboard.STATIC_DIR / name).read_text(encoding="utf-8").encode("utf-8")

        page = get("/")
        self.assertEqual(page, (200, "text/html; charset=utf-8", text("index.html")))
        # The page loads its style and script by these paths.
        for name, content_type in [("app.css", "text/css"), ("app.js", "text/javascript")]:
            with self.subTest(name=name):
                self.assertIn(f'"/static/{name}"'.encode(), page[2])
                self.assertEqual(get(f"/static/{name}"), (200, f"{content_type}; charset=utf-8", text(name)))
        # Nothing else is reachable through /static/.
        for path in ["/static/", "/static/../voice_dashboard.py", "/static/..%2fconfig.json", "/static/C:app.js",
                     "/static/sub/app.js", "/static/missing.js", "/static/index.py"]:
            with self.subTest(path=path):
                self.assertEqual(get(path)[0], 404)

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

    def make_dataset(self):
        """A dataset in a temporary DATASETS_DIR, for the clip tests."""
        datasets = self.root / "datasets"
        datasets.mkdir(exist_ok=True)
        self.enterContext(patch.object(dashboard, "DATASETS_DIR", datasets))
        return dashboard.create_dataset("Clip Test", language="en")["id"]

    def write_wav(self, path, seconds, rate=44100, loud=6000):
        frames = int(rate * seconds)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(struct.pack(f"<{frames}h", *[loud * ((i // 40) % 2 * 2 - 1) for i in range(frames)]))

    def test_cuts_fall_in_pauses_at_any_recording_level(self):
        def levels(loud, quiet, parts):
            frames = []
            for speech, seconds in parts:
                frames += [loud if speech else quiet] * round(seconds / dashboard.CUT_STEP)
            return frames

        # A 0.5 s pause in a clip still under 3 s is kept, a 1.2 s pause always cuts, a 0.4 s pause
        # cuts a clip of 3 s or more, a lone 0.3 s click is left out, 0.05 s dips never cut, and a
        # clip ends before it would grow past 10 s. Each clip keeps up to 0.2 s of its pauses.
        parts = [(1, 2.0), (0, 0.5), (1, 2.0), (0, 1.2), (1, 3.2), (0, 0.4), (1, 2.0), (0, 1.5), (1, 0.3), (0, 1.5)]
        parts += [(1, 1.95), (0, 0.05)] * 3 + [(1, 2.0), (0, 0.2)] * 3 + [(0, 1.0)]
        for loud, quiet in [(-18.0, -75.0), (-45.0, -95.0), (-20.0, float("-inf"))]:
            with self.subTest(loud=loud, quiet=quiet):
                clips = dashboard.plan_clips(levels(loud, quiet, parts))
                self.assertEqual(clips, [(0.0, 4.7), (5.5, 9.1), (9.1, 11.5), (14.4, 22.7), (22.7, 27.2)])
        self.assertEqual(dashboard.plan_clips([float("-inf")] * 100), [])
        self.assertEqual(dashboard.plan_clips([-30.0] * 100), [])

    def test_clips_can_be_listed_corrected_and_deleted(self):
        dataset_id = self.make_dataset()
        root = dashboard.DATASETS_DIR / dataset_id
        for name, seconds, text in [("one", 2.0, "First line."), ("two", 0.5, "Short."), ("three", 3.0, "Third line.")]:
            self.write_wav(self.root / f"{name}.wav", seconds)
            dashboard.add_dataset_clip(root, self.root / f"{name}.wav", name, f"{name}.wav", text)
        listed = dashboard.dataset_clips(dataset_id)
        self.assertEqual([(clip["wav"], clip["duration"], clip["text"]) for clip in listed["clips"]],
                         [("one.wav", 2.0, "First line."), ("two.wav", 0.5, "Short."), ("three.wav", 3.0, "Third line.")])
        self.assertEqual(listed["seconds"], 5.5)
        self.assertTrue(listed["clips"][0]["url"].startswith(f"/datasets/{dataset_id}/wavs/one.wav?v="))
        self.assertEqual(dashboard.update_dataset_clip(dataset_id, "one.wav", " First |  line\nagain. "), "First line again.")
        with self.assertRaisesRegex(RuntimeError, "needs its transcript"):
            dashboard.update_dataset_clip(dataset_id, "one.wav", " | ")
        dashboard.delete_dataset_clip(dataset_id, "two.wav")
        with self.assertRaisesRegex(RuntimeError, "no longer in the dataset"):
            dashboard.delete_dataset_clip(dataset_id, "two.wav")
        self.assertFalse((root / "wavs" / "two.wav").exists())
        self.assertEqual((root / f"{dataset_id}.list").read_text(encoding="utf-8"),
                         "wavs/one.wav|speaker|en|First line again.\nwavs/three.wav|speaker|en|Third line.\n")
        self.assertTrue(dashboard.dataset_summary(dashboard.dataset_descriptor(dataset_id))["ready"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "Needs ffmpeg")
    def test_a_long_recording_becomes_transcribed_clips(self):
        dataset_id = self.make_dataset()
        rate = 44100
        tone = lambda seconds: [6000 * ((i // 40) % 2 * 2 - 1) for i in range(int(rate * seconds))]
        samples = tone(4.0) + [0] * int(rate * 1.5) + tone(3.0) + [0] * int(rate * 1.5) + tone(2.0) + [0] * rate
        (self.root / "tmp").mkdir()
        upload = self.root / "tmp" / "upload-1.audio"
        with wave.open(str(upload), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        heard = iter(["First part.", "", "Last part."])
        with (
            patch.object(dashboard, "TMP_DIR", self.root / "tmp"),
            patch.object(dashboard, "CUT_DIR", self.root / "cut"),
            patch.object(dashboard, "transcribe_dataset_wav", side_effect=lambda path, language: next(heard)),
        ):
            dashboard.start_dataset_cut(dataset_id, [(upload, "talk show.wav")])
            job = dashboard.JOBS["dataset-cut"]
            job.thread.join(timeout=60)
        self.assertEqual(job.status, "done", job.lines)
        self.assertEqual(job.lines[-1], "Done. 2 clips to check in the list below. 1 had no words and were left out.")
        # The clips wait for the page, which lists them to be checked; the dataset is unchanged.
        self.assertEqual([(clip["name"], clip["text"]) for clip in job.results], [("talk-show-001.wav", "First part."), ("talk-show-003.wav", "Last part.")])
        self.assertEqual(dashboard.dataset_clips(dataset_id)["clips"], [])
        first = self.root / "cut" / job.results[0]["url"].rsplit("/", 1)[1]
        self.assertTrue(job.results[0]["url"].startswith("/outputs/cut/"))
        self.assertAlmostEqual(dashboard.wav_seconds(first), 4.2, delta=0.05)
        self.assertEqual(len(list((self.root / "cut").iterdir())), 2)
        self.assertEqual(list((self.root / "tmp").iterdir()), [])

    def test_a_stopped_thread_job_stops_its_program(self):
        started = threading.Event()

        def work(job):
            started.set()
            job.run([sys.executable, "-c", "import time; time.sleep(30)"])
            job.say("not reached")

        job = dashboard.ThreadJob("Sleeper", work)
        job.start()
        started.wait(5)
        time.sleep(0.5)
        job.stop()
        self.assertFalse(job.running())
        self.assertEqual(job.snapshot()["status"], "stopped")
        self.assertNotIn("not reached", job.lines)

    def test_dataset_transcripts_ask_for_their_model_and_language(self):
        calls = []
        loaded = []

        class Model:
            def transcribe(self, path, **options):
                calls.append(options)
                return [MagicMock(text=" Hello there. ")], MagicMock(language=options["language"])

        server = relay_asr_server.Server(("127.0.0.1", 0), relay_asr_server.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        wav = self.root / "clip.wav"
        self.write_wav(wav, 1.0, rate=16000)
        with (
            patch.object(relay_asr_server, "load_model", side_effect=lambda name=None: loaded.append(name) or Model()),
            patch.object(relay_asr_server, "TMP_DIR", self.root / "asr"),
            patch.object(dashboard, "RELAY_ASR_URL", f"http://127.0.0.1:{server.server_address[1]}"),
            patch.object(dashboard, "relay_asr_ready", return_value=True),
        ):
            self.assertEqual(dashboard.transcribe_dataset_wav(wav, "en"), "Hello there.")
            self.assertEqual(dashboard.transcribe_dataset_wav(wav, "yue"), "Hello there.")
            with patch.object(dashboard, "DATASET_ASR_MODEL", "large-v3"):
                dashboard.transcribe_dataset_wav(wav, "yue")
            # A server from before an update answers with the Mic to Voice model.
            with patch.object(relay_asr_server, "transcribe_wav", return_value={"ok": True, "text": "x", "model": "base.en"}):
                with self.assertRaisesRegex(RuntimeError, "older version"):
                    dashboard.transcribe_dataset_wav(wav, "en")
        self.assertEqual(loaded, ["small.en", "small", "large-v3"])
        self.assertEqual([(call["language"], call["beam_size"], call["vad_filter"]) for call in calls], [("en", 5, False), ("zh", 5, False), ("yue", 5, False)])

    def test_training_progress_follows_each_epoch(self):
        # As GPT-SoVITS logs them: SoVITS continuing from epoch 9 of an earlier run, and GPT's
        # Lightning progress bar, which counts from 0 and shows 100% more than once.
        sovits = dashboard.TrainingProgress("sovits", 16)
        gpt = dashboard.TrainingProgress("gpt", 4)
        with patch.object(dashboard.time, "time", side_effect=[100.0, 160.0, 220.0, 500.0, 530.0]):
            for line in ["start training from epoch 9", "INFO:voice:Train Epoch: 9 [0%]", " 52%|#####2    | 15/29 [00:03<00:02]",
                         "INFO:voice:====> Epoch: 9", "INFO:voice:====> Epoch: 10", "INFO:voice:Train Epoch: 11 [50%]", "INFO:voice:====> Epoch: 11"]:
                sovits(line)
            self.assertEqual(sovits.state(), {"epoch": 11, "epochs": 16, "seconds_left": 300})
            for line in ["Epoch 0:  50%|#####     | 6/12 [00:02<00:02,  2.9it/s]", "Epoch 0: 100%|##########| 12/12 [00:04<00:00,  2.9it/s]",
                         "Epoch 0: 100%|##########| 12/12 [00:04<00:00,  2.9it/s, v_num=0, loss=3.1]", "Epoch 1: 100%|##########| 12/12 [00:04<00:00]"]:
                gpt(line)
        self.assertEqual(gpt.state(), {"epoch": 2, "epochs": 4, "seconds_left": 60})
        self.assertEqual(dashboard.TrainingProgress("gpt", 4).state(), {"epoch": 0, "epochs": 4, "seconds_left": None})

    def test_a_watched_job_reports_its_progress(self):
        proc = dashboard.ManagedProcess("Train", [sys.executable, "-c", "print('INFO:x:====> Epoch: 1'); print('INFO:x:====> Epoch: 2')"])
        proc.watch = dashboard.TrainingProgress("sovits", 5)
        proc.start()
        proc.process.wait(timeout=10)
        proc.reader.join(timeout=5)
        self.assertEqual(proc.snapshot()["progress"]["epoch"], 2)

    def test_epochs_are_found_by_model_name(self):
        models = {
            "gpt": [{"name": name, "path": f"GPT_weights_v2Pro/{name}"} for name in ["joe-e10.ckpt", "joe-e5.ckpt", "joe2-e5.ckpt", "s1v3.ckpt"]],
            "sovits": [{"name": name, "path": f"SoVITS_weights_v2Pro/{name}"} for name in ["joe_e8_s232.pth", "joe_e16_s464.pth", "joe-x_e4_s1.pth"]],
        }
        with patch.object(dashboard, "model_files", return_value=models):
            self.assertEqual(dashboard.model_epochs("Joe", "gpt"), [(5, "GPT_weights_v2Pro/joe-e5.ckpt"), (10, "GPT_weights_v2Pro/joe-e10.ckpt")])
            self.assertEqual([epoch for epoch, _ in dashboard.model_epochs("joe", "sovits")], [8, 16])
            self.assertEqual(dashboard.model_epochs("nobody", "sovits"), [])

    def test_v2proplus_trains_with_its_own_decoder_and_folder(self):
        dashboard.write_s2_config(1, 1, 1, "joe")
        pro = json.loads((self.root / "TEMP/joe_s2.json").read_text())
        dashboard.write_s2_config(1, 1, 1, "joe", version="v2ProPlus")
        plus = json.loads((self.root / "TEMP/joe_s2.json").read_text())
        # GPT-SoVITS's s2v2ProPlus.json differs from s2v2Pro.json in the decoder alone.
        self.assertEqual((plus["model"]["upsample_initial_channel"], plus["model"]["upsample_kernel_sizes"]), (768, [20, 16, 8, 2, 2]))
        self.assertEqual((pro["model"]["upsample_initial_channel"], pro["model"]["upsample_kernel_sizes"]), (512, [16, 16, 8, 2, 2]))
        self.assertEqual((plus["version"], plus["model"]["version"], plus["save_weight_dir"]), ("v2ProPlus", "v2ProPlus", "SoVITS_weights_v2ProPlus"))
        self.assertEqual(plus["train"]["pretrained_s2G"], "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth")
        self.assertEqual(plus["train"]["pretrained_s2D"], "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth")
        for config in (pro, plus):
            config["model"].pop("upsample_initial_channel"), config["model"].pop("upsample_kernel_sizes")
            for key in ("pretrained_s2G", "pretrained_s2D"):
                config["train"].pop(key)
            for key in ("version", "save_weight_dir"):
                config.pop(key)
            config["model"].pop("version")
        self.assertEqual(pro, plus)

        ok, missing = subprocess.CompletedProcess([], 0, "", ""), subprocess.CompletedProcess([], 1, "", "")
        with (
            patch.object(dashboard, "dataset_descriptor", return_value={"name": "Test"}),
            patch.object(dashboard, "free_gpu_for"),
            patch.object(dashboard.ManagedProcess, "start"),
        ):
            # An older GPT-SoVITS without the v2ProPlus models says so before training starts.
            with patch.object(dashboard, "run_wsl", side_effect=[ok, missing]):
                with self.assertRaisesRegex(RuntimeError, "missing the pretrained v2ProPlus models"):
                    dashboard.start_train("sovits", 1, 1, 1, "test", "joe", "v2ProPlus")
            self.assertNotIn("train-sovits", dashboard.JOBS)
            with self.assertRaisesRegex(RuntimeError, "Choose v2Pro or v2ProPlus"):
                dashboard.start_train("sovits", 1, 1, 1, "test", "joe", "v4")
            with patch.object(dashboard, "run_wsl", return_value=ok):
                dashboard.start_train("sovits", 1, 1, 1, "test", "joe", "v2ProPlus")
        self.assertEqual(dashboard.JOBS["train-sovits"].name, "SoVITS training: joe (v2ProPlus)")
        self.assertEqual(json.loads((self.root / "TEMP/joe_s2.json").read_text())["version"], "v2ProPlus")

    def test_each_version_has_its_own_epochs(self):
        for folder, name in [("SoVITS_weights_v2Pro", "joe_e4_s10.pth"), ("SoVITS_weights_v2Pro", "joe_e8_s20.pth"),
                             ("SoVITS_weights_v2ProPlus", "joe_e4_s10.pth"), ("GPT_weights_v2Pro", "joe-e5.ckpt"),
                             ("GPT_SoVITS/pretrained_models/v2Pro", "s2Gv2Pro.pth"), ("GPT_SoVITS/pretrained_models/v2Pro", "s2Gv2ProPlus.pth")]:
            (self.root / folder).mkdir(parents=True, exist_ok=True)
            (self.root / folder / name).write_bytes(b"x")
        models = dashboard.model_files()
        self.assertEqual([(row["path"], row["version"]) for row in models["sovits"][-2:]],
                         [("GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth", "v2Pro"), ("GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth", "v2ProPlus")])
        self.assertEqual(dashboard.model_epochs("joe", "sovits"), [(4, "SoVITS_weights_v2Pro/joe_e4_s10.pth"), (8, "SoVITS_weights_v2Pro/joe_e8_s20.pth")])
        self.assertEqual(dashboard.model_epochs("joe", "sovits", "v2ProPlus"), [(4, "SoVITS_weights_v2ProPlus/joe_e4_s10.pth")])
        self.assertEqual(dashboard.model_epochs("joe", "gpt", "v2ProPlus"), [(5, "GPT_weights_v2Pro/joe-e5.ckpt")])

    def test_gpt_epochs_are_compared_with_a_trained_sovits_of_either_version(self):
        for folder, name in [("GPT_weights_v2Pro", "joe-e5.ckpt"), ("SoVITS_weights_v2Pro", "joe_e8_s20.pth"),
                             ("GPT_SoVITS/pretrained_models/v2Pro", "s2Gv2ProPlus.pth")]:
            (self.root / folder).mkdir(parents=True, exist_ok=True)
            (self.root / folder / name).write_bytes(b"x")
        reference = {"id": "ref", "text": "Hello.", "lang": "en", "wsl_path": "/d/ref.wav"}
        with (
            patch.object(dashboard, "dataset_descriptor", return_value={"id": "joe", "language": "en"}),
            patch.object(dashboard, "parse_references", return_value=[reference]),
            patch.object(dashboard, "wake_wsl"),
            patch.object(dashboard, "compare_epochs") as compare,
        ):
            for version in ("v2ProPlus", "v2Pro"):
                with self.subTest(version=version):
                    dashboard.start_compare("joe", "gpt", "joe", "ref.wav", "Hi.", version)
                    dashboard.JOBS["compare"].thread.join(5)
                    # No v2ProPlus SoVITS is trained, so the trained v2Pro one speaks, not the pretrained v2ProPlus.
                    self.assertEqual(compare.call_args.args[1]["other"], "SoVITS_weights_v2Pro/joe_e8_s20.pth")

    def test_a_v2proplus_voice_is_installed_with_its_version(self):
        datasets = self.root / "datasets"
        datasets.mkdir()
        archive = self.root / "voice.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("voice.json", json.dumps({"format": 1, "name": "Joe", "language": "en", "version": "v2ProPlus", "gpt": "joe.ckpt", "sovits": "joe.pth",
                                                      "reference": {"file": "reference.wav", "text": "Hello.", "language": "en"}, "extra_references": []}))
            bundle.writestr("models/joe.ckpt", b"gpt")
            bundle.writestr("models/joe.pth", b"sovits")
            bundle.writestr("reference.wav", b"audio")
        with (
            patch.object(dashboard, "DATASETS_DIR", datasets),
            patch.object(dashboard, "run_wsl", return_value=subprocess.CompletedProcess([], 0, "", "")),
            patch.object(dashboard, "convert_audio", side_effect=lambda source, target, sample_rate=44100: shutil.copyfile(source, target)),
            patch.object(dashboard, "model_files"),
        ):
            result = dashboard.import_voice(archive)
        self.assertEqual((result["gpt"], result["sovits"]), ("GPT_weights_v2Pro/joe.ckpt", "SoVITS_weights_v2ProPlus/joe.pth"))
        self.assertEqual((self.root / "SoVITS_weights_v2ProPlus" / "joe.pth").read_bytes(), b"sovits")

    def test_a_voice_survives_export_and_import(self):
        dataset_id = self.make_dataset()
        root = dashboard.DATASETS_DIR / dataset_id
        for name, seconds, text in [("ref", 4.0, "Reference words."), ("more", 2.0, "More words.")]:
            self.write_wav(self.root / f"{name}.wav", seconds)
            dashboard.add_dataset_clip(root, self.root / f"{name}.wav", name, f"{name}.wav", text)
        gsv = self.root / "gsv"
        for folder, name, data in [("GPT_weights_v2Pro", "joe-e10.ckpt", b"gpt weights"), ("SoVITS_weights_v2Pro", "joe_e8_s1.pth", b"sovits weights")]:
            (gsv / folder).mkdir(parents=True)
            (gsv / folder / name).write_bytes(data)
        models = {"gpt": [{"name": "joe-e10.ckpt", "path": "GPT_weights_v2Pro/joe-e10.ckpt"}],
                  "sovits": [{"name": "joe_e8_s1.pth", "path": "SoVITS_weights_v2Pro/joe_e8_s1.pth"}]}
        synced = []
        self.enterContext(patch.object(dashboard, "WSL_GSV_WIN", gsv))
        self.enterContext(patch.object(dashboard, "TMP_DIR", self.root / "tmp"))
        self.enterContext(patch.object(dashboard, "model_files", return_value=models))
        self.enterContext(patch.object(dashboard, "run_wsl", side_effect=lambda script, **kwargs: synced.append(script) or subprocess.CompletedProcess(script, 0, "", "")))
        self.enterContext(patch.object(dashboard, "convert_audio", side_effect=lambda source, target, sample_rate=44100: shutil.copyfile(source, target)))

        path, name = dashboard.export_voice("GPT_weights_v2Pro/joe-e10.ckpt", "SoVITS_weights_v2Pro/joe_e8_s1.pth", dataset_id, "ref",
                                            " Reference |  words. ", ["more", "ref", "missing"])
        self.assertEqual(name, "clip-test.voice.zip")
        with zipfile.ZipFile(path) as bundle:
            self.assertEqual(sorted(bundle.namelist()), ["extra-1.wav", "models/joe-e10.ckpt", "models/joe_e8_s1.pth", "reference.wav", "voice.json"])
            manifest = json.loads(bundle.read("voice.json"))
        self.assertEqual((manifest["name"], manifest["language"], manifest["version"], manifest["reference"]["text"], manifest["extra_references"][0]["text"]),
                         ("Clip Test", "en", "v2Pro", "Reference words.", "More words."))
        with self.assertRaisesRegex(RuntimeError, "reference clip of 3 to 10 seconds"):
            dashboard.export_voice("GPT_weights_v2Pro/joe-e10.ckpt", "SoVITS_weights_v2Pro/joe_e8_s1.pth", dataset_id, "more", "", [])
        with self.assertRaisesRegex(RuntimeError, "Choose a GPT and a SoVITS model"):
            dashboard.export_voice("../../secret.ckpt", "SoVITS_weights_v2Pro/joe_e8_s1.pth", dataset_id, "ref", "", [])

        # The same models are used as they are, in a new dataset with the reference clips.
        result = dashboard.import_voice(path)
        self.assertEqual(result, {"dataset": "clip-test-2", "name": "Clip Test 2", "gpt": "GPT_weights_v2Pro/joe-e10.ckpt",
                                  "sovits": "SoVITS_weights_v2Pro/joe_e8_s1.pth", "reference": "reference"})
        self.assertEqual([(clip["wav"], clip["text"]) for clip in dashboard.dataset_clips("clip-test-2")["clips"]],
                         [("reference.wav", "Reference words."), ("extra-1.wav", "More words.")])
        self.assertIn("/clip-test-2", synced[-1])
        # Another model of the same size and name is kept beside the one already there.
        (gsv / "GPT_weights_v2Pro" / "joe-e10.ckpt").write_bytes(b"gpt weightz")
        self.assertEqual(dashboard.import_voice(path)["gpt"], "GPT_weights_v2Pro/joe-e10-2.ckpt")
        self.assertEqual((gsv / "GPT_weights_v2Pro" / "joe-e10-2.ckpt").read_bytes(), b"gpt weights")
        self.assertEqual(sorted(p.name for p in (gsv / "GPT_weights_v2Pro").iterdir()), ["joe-e10-2.ckpt", "joe-e10.ckpt"])
        (self.root / "tmp" / "not-a-voice.zip").write_bytes(b"plain text")
        with self.assertRaisesRegex(RuntimeError, "not a voice file"):
            dashboard.import_voice(self.root / "tmp" / "not-a-voice.zip")
        with zipfile.ZipFile(self.root / "tmp" / "other.zip", "w") as bundle:
            bundle.writestr("voice.json", json.dumps({**manifest, "version": "v4"}))
        with self.assertRaisesRegex(RuntimeError, "GPT-SoVITS v4"):
            dashboard.import_voice(self.root / "tmp" / "other.zip")

    def write_speech(self, path, background=20, loud=8000, clip_at=None, seconds=3.0, rate=44100):
        """Speech-like audio: half-second bursts with pauses, over a background of the given level."""
        rng = __import__("random").Random(7)
        samples = []
        for i in range(int(rate * seconds)):
            value = rng.uniform(-background, background)
            if (i // (rate // 2)) % 2 == 0:
                value += loud * ((i // 40) % 2 * 2 - 1)
            if clip_at:
                value = max(-clip_at, min(clip_at, value)) * 32767 / clip_at
            samples.append(int(max(-32768, min(32767, value))))
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    def test_transcripts_are_compared_by_their_words(self):
        self.assertEqual(dashboard.transcript_difference("Please, remain calm!", "please remain calm", "en"), 0.0)
        self.assertEqual(dashboard.transcript_difference("Please remain calm now.", "Change remain calm now.", "en"), 0.25)
        self.assertEqual(dashboard.transcript_difference("Up to 1210 degrees.", "Up to 1,210 degrees.", "en"), 0.0)
        self.assertEqual(dashboard.transcript_difference("Hello there.", "", "en"), 1.0)
        self.assertEqual(dashboard.transcript_difference("", "", "en"), 0.0)
        # Chinese has no spaces between words, so its characters are compared.
        self.assertEqual(dashboard.transcript_difference("你好，世界。", "你好世界", "zh"), 0.0)
        self.assertEqual(dashboard.transcript_difference("你好世界", "你好地界", "zh"), 0.25)
        # Korean spacing varies, so its letters are compared.
        self.assertEqual(dashboard.transcript_difference("안녕하세요 세계", "안녕 하세요 세계", "ko"), 0.0)

    def test_clip_sound_finds_noise_quiet_and_clipping(self):
        cases = {"clean": {}, "noisy": {"background": 3000}, "quiet": {"loud": 200, "background": 2},
                 "clipped": {"clip_at": 4000}}
        sounds = {}
        for name, options in cases.items():
            self.write_speech(self.root / f"{name}.wav", **options)
            sounds[name] = dashboard.clip_sound(self.root / f"{name}.wav")
        self.assertGreater(sounds["clean"]["speech_db"], dashboard.CHECK_NOISE_DB)
        self.assertLess(sounds["noisy"]["speech_db"], dashboard.CHECK_NOISE_DB)
        self.assertGreater(sounds["noisy"]["speech_db"], dashboard.CHECK_FLAT_DB)
        # A steady tone has no speech standing out at all.
        self.write_wav(self.root / "tone.wav", 2.0)
        self.assertLess(dashboard.clip_sound(self.root / "tone.wav")["speech_db"], dashboard.CHECK_FLAT_DB)
        self.assertLess(sounds["quiet"]["peak_db"], dashboard.CHECK_QUIET_DB)
        self.assertGreater(sounds["clean"]["peak_db"], dashboard.CHECK_QUIET_DB)
        self.assertGreaterEqual(sounds["clipped"]["clipped"], dashboard.CHECK_CLIPPED_PLACES)
        self.assertEqual(sounds["clean"]["clipped"], 0)
        (self.root / "broken.wav").write_bytes(b"RIFF nonsense")
        self.assertIsNone(dashboard.clip_sound(self.root / "broken.wav"))
        # A file cut short in the middle of a sample is still measured.
        data = (self.root / "clean.wav").read_bytes()
        (self.root / "cut.wav").write_bytes(data[:-1])
        self.assertGreater(dashboard.clip_sound(self.root / "cut.wav")["speech_db"], dashboard.CHECK_NOISE_DB)

    def test_the_check_names_each_clip_to_look_at(self):
        dataset_id = self.make_dataset()
        root = dashboard.DATASETS_DIR / dataset_id
        for name, options, text in [("good", {}, "Good words."), ("noisy", {"background": 3000}, "Noisy words."),
                                    ("clipped", {"clip_at": 4000}, "Clipped words."), ("again", {}, "Good words."),
                                    ("wrong", {"seconds": 2.0}, "Something else entirely.")]:
            self.write_speech(self.root / f"{name}.wav", **options)
            dashboard.add_dataset_clip(root, self.root / f"{name}.wav", name, f"{name}.wav", text)
        heard = {"good.wav": "Good words.", "noisy.wav": "Noisy words.", "clipped.wav": "Clipped words.", "again.wav": "Good words.",
                 "wrong.wav": "Not what it says."}
        with patch.object(dashboard, "transcribe_dataset_wav", side_effect=lambda path, language: heard[Path(path).name]):
            dashboard.start_dataset_check(dataset_id)
            job = dashboard.JOBS["dataset-check"]
            job.thread.join(timeout=60)
        self.assertEqual(job.status, "done", job.lines)
        found = {row["wav"]: row for row in job.results}
        self.assertEqual(sorted(found), ["again.wav", "clipped.wav", "noisy.wav", "wrong.wav"])
        self.assertIn("Loud background", found["noisy.wav"]["problems"][0])
        self.assertIn("Distorted", found["clipped.wav"]["problems"][0])
        self.assertEqual(found["again.wav"]["problems"], ["The same audio as good.wav."])
        self.assertEqual((found["wrong.wav"]["problems"], found["wrong.wav"]["heard"]),
                         (["The transcript differs from what speech recognition hears."], "Not what it says."))
        self.assertIsNone(found["noisy.wav"]["heard"])
        self.assertEqual(job.lines[-1], "Done. 4 of 5 clips need a look.")
        with patch.object(dashboard, "transcribe_dataset_wav", side_effect=lambda path, language: time.sleep(0.2) or "Words."):
            dashboard.start_dataset_check(dataset_id)
            with self.assertRaisesRegex(RuntimeError, "already running"):
                dashboard.start_dataset_check(dataset_id)
            dashboard.JOBS["dataset-check"].stop()
        self.assertEqual(dashboard.JOBS["dataset-check"].status, "stopped")

    def make_speaking_voice(self):
        """A dataset with a reference clip that has spoken on Text to Voice."""
        dataset_id = self.make_dataset()
        for folder, name in [("GPT_weights_v2Pro", "joe-e10.ckpt"), ("SoVITS_weights_v2Pro", "joe_e8_s1.pth")]:
            (self.root / folder).mkdir(parents=True, exist_ok=True)
            (self.root / folder / name).write_bytes(b"weights")
        root = dashboard.DATASETS_DIR / dataset_id
        self.write_wav(self.root / "ref.wav", 4.0)
        dashboard.add_dataset_clip(root, self.root / "ref.wav", "ref", "ref.wav", "Reference words.")
        dashboard.save_speech_settings(dataset_id, dashboard.speech_settings_from_generation({
            "gpt": "GPT_weights_v2Pro/joe-e10.ckpt", "sovits": "SoVITS_weights_v2Pro/joe_e8_s1.pth",
            "ref_audio_path": "/home/me/datasets/clip-test/wavs/ref.wav", "aux_ref_audio_paths": [],
            "prompt_text": "Reference words.", "text_lang": "en", "top_k": "20", "speed_factor": "1.1", "seed": "-1",
        }))
        return dataset_id

    def fake_engine(self, sent):
        wav = io.BytesIO()
        with wave.open(wav, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(32000)
            out.writeframes(struct.pack("<3200h", *[3000 * ((i // 20) % 2 * 2 - 1) for i in range(3200)]))

        def urlopen(request, timeout=None):
            sent.append(json.loads(request.data))
            return contextlib.nullcontext(io.BytesIO(wav.getvalue()))

        return urlopen

    def test_a_voice_speaks_for_other_programs_as_on_text_to_voice(self):
        dataset_id = self.make_speaking_voice()
        self.assertEqual(dashboard.speech_voices(), [{"id": dataset_id, "name": "Clip Test"}])
        sent = []
        with (
            patch.object(dashboard, "api_ready", return_value=True),
            patch.object(dashboard, "set_api_weights") as weights,
            patch.object(dashboard.urllib.request, "urlopen", self.fake_engine(sent)),
        ):
            audio, content_type = dashboard.speak_for_api({"input": " Hello  there. ", "voice": "clip test", "response_format": "wav", "speed": 2})
        self.assertEqual((audio[:4], content_type), (b"RIFF", "audio/wav"))
        weights.assert_called_once_with("GPT_weights_v2Pro/joe-e10.ckpt", "SoVITS_weights_v2Pro/joe_e8_s1.pth")
        payload = sent[0]
        self.assertEqual((payload["text"], payload["prompt_text"], payload["top_k"], payload["speed_factor"]), ("Hello there.", "Reference words.", 20, 2.2))
        self.assertTrue(payload["ref_audio_path"].endswith("/clip-test/wavs/ref.wav"))
        for body, message in [({"input": "", "voice": dataset_id}, "input is empty"), ({"input": "Hi.", "voice": "alloy"}, "no voice named 'alloy'"),
                              ({"input": "Hi.", "voice": dataset_id, "response_format": "ogg"}, "response_format must be"),
                              ({"input": "Hi.", "voice": dataset_id, "speed": 9}, "speed must be")]:
            with self.subTest(body=body), self.assertRaisesRegex(dashboard.SpeechApiError, message):
                dashboard.speak_for_api(body)
        # Nothing starts the engine while training uses the GPU.
        with patch.object(dashboard, "api_ready", return_value=False), patch.object(dashboard, "gpu_busy", return_value=True), \
                patch.object(dashboard, "start_api") as start_api:
            with self.assertRaisesRegex(dashboard.SpeechApiError, "Training or another GPU job"):
                dashboard.speak_for_api({"input": "Hi.", "voice": dataset_id})
            start_api.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg"), "Needs ffmpeg")
    def test_speech_comes_in_the_format_asked_for(self):
        dataset_id = self.make_speaking_voice()
        with (
            patch.object(dashboard, "api_ready", return_value=True),
            patch.object(dashboard, "set_api_weights"),
            patch.object(dashboard.urllib.request, "urlopen", self.fake_engine([])),
        ):
            mp3, mp3_type = dashboard.speak_for_api({"input": "Hi.", "voice": dataset_id})
            pcm, pcm_type = dashboard.speak_for_api({"input": "Hi.", "voice": dataset_id, "response_format": "pcm"})
        self.assertEqual(mp3_type, "audio/mpeg")
        self.assertTrue(mp3.startswith(b"ID3") or mp3[0] == 0xFF)
        # A tenth of a second at 32 kHz becomes 2400 samples at OpenAI's 24 kHz.
        self.assertEqual((pcm_type, abs(len(pcm) // 2 - 2400) < 50), ("audio/pcm", True))

    def test_the_speech_api_answers_in_openai_shape(self):
        dataset_id = self.make_speaking_voice()
        port = self.start_server()
        status, _, body = self.request(port, "GET", "/v1/audio/voices", Host=f"127.0.0.1:{port}")
        self.assertEqual((status, json.loads(body)), (200, {"voices": [{"id": dataset_id, "name": "Clip Test"}]}))
        status, _, body = self.request(port, "GET", "/v1/models", Host=f"localhost:{port}")
        self.assertEqual(json.loads(body)["data"][0]["id"], "gpt-sovits")
        status, _, body = self.request(port, "POST", "/v1/audio/speech", body=json.dumps({"input": "Hi.", "voice": "nobody"}), Host=f"127.0.0.1:{port}")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["param"], "voice")
        # Programs in Docker name this PC host.docker.internal; the dashboard itself stays local.
        self.assertEqual(self.request(port, "GET", "/v1/audio/voices", Host=f"host.docker.internal:{port}")[0], 200)
        self.assertEqual(self.request(port, "GET", "/api/state", Host=f"host.docker.internal:{port}")[0], 403)
        self.assertEqual(self.request(port, "GET", "/v1/audio/voices", Host=f"evil.example:{port}")[0], 403)
        with patch.object(dashboard, "SPEECH_API_KEY", "secret"):
            speech = json.dumps({"input": "Hi.", "voice": "nobody"})
            status, _, body = self.request(port, "POST", "/v1/audio/speech", body=speech, Host=f"127.0.0.1:{port}", Authorization="Bearer wrong")
            self.assertEqual((status, json.loads(body)["error"]["type"]), (401, "invalid_request_error"))
            self.assertEqual(self.request(port, "POST", "/v1/audio/speech", body=speech, Host=f"127.0.0.1:{port}", Authorization="Bearer secret")[0], 400)
            # Open WebUI lists the voices without the key.
            self.assertEqual(self.request(port, "GET", "/v1/audio/voices", Host=f"127.0.0.1:{port}")[0], 200)
        # A body that is not a JSON object is the client's mistake.
        for body in ["not json", "[1, 2]", json.dumps({"input": ["a", "b"], "voice": dataset_id})]:
            with self.subTest(body=body):
                status, _, answer = self.request(port, "POST", "/v1/audio/speech", body=body, Host=f"127.0.0.1:{port}")
                self.assertEqual((status, json.loads(answer)["error"]["type"]), (400, "invalid_request_error"))

    def test_a_voice_whose_model_is_gone_says_so(self):
        dataset_id = self.make_speaking_voice()
        (self.root / "SoVITS_weights_v2Pro" / "joe_e8_s1.pth").unlink()
        with patch.object(dashboard, "api_ready", return_value=True), patch.object(dashboard, "set_api_weights") as weights:
            with self.assertRaisesRegex(dashboard.SpeechApiError, "SoVITS model of Clip Test, SoVITS_weights_v2Pro/joe_e8_s1.pth, is no longer"):
                dashboard.speak_for_api({"input": "Hi.", "voice": dataset_id})
        weights.assert_not_called()

    def test_the_reference_text_follows_its_clip_unless_edited(self):
        dataset_id = self.make_speaking_voice()
        settings = json.loads((dashboard.DATASETS_DIR / dataset_id / "speech.json").read_text(encoding="utf-8"))
        # Text to Voice sent the clip's own transcript, so a later correction of the clip counts.
        self.assertNotIn("prompt_text", settings)
        dashboard.update_dataset_clip(dataset_id, "ref.wav", "Corrected words.")
        sent = []
        with patch.object(dashboard, "api_ready", return_value=True), patch.object(dashboard, "set_api_weights"), \
                patch.object(dashboard.urllib.request, "urlopen", self.fake_engine(sent)):
            dashboard.speak_for_api({"input": "Hi.", "voice": dataset_id, "response_format": "wav"})
            self.assertEqual(sent[-1]["prompt_text"], "Corrected words.")
            # A reference text edited on Text to Voice is kept.
            dashboard.save_speech_settings(dataset_id, {**settings, "prompt_text": "Edited words."})
            dashboard.speak_for_api({"input": "Hi.", "voice": dataset_id, "response_format": "wav"})
            self.assertEqual(sent[-1]["prompt_text"], "Edited words.")

    def test_the_check_goes_on_when_a_clip_is_deleted_meanwhile(self):
        dataset_id = self.make_dataset()
        root = dashboard.DATASETS_DIR / dataset_id
        for name in ("a", "b", "c"):
            # Lengths differ, so no clip is the same audio as another.
            self.write_speech(self.root / f"{name}.wav", seconds=1.5 + ord(name) / 100)
            dashboard.add_dataset_clip(root, self.root / f"{name}.wav", name, f"{name}.wav", f"Words {name}.")
        heard = []

        def transcribe(path, language):
            if Path(path).name == "a.wav":
                dashboard.delete_dataset_clip(dataset_id, "b.wav")
            Path(path).read_bytes()
            heard.append(Path(path).name)
            return f"Words {Path(path).stem}."

        with patch.object(dashboard, "transcribe_dataset_wav", side_effect=transcribe):
            dashboard.start_dataset_check(dataset_id)
            job = dashboard.JOBS["dataset-check"]
            job.thread.join(timeout=30)
        self.assertEqual((job.status, heard, job.lines[-1]), ("done", ["a.wav", "c.wav"], "Done. No problems found in 3 clips."))

    def test_cantonese_transcripts_are_compared_only_with_large_v3(self):
        self.make_dataset()
        dataset_id = dashboard.create_dataset("Yue", language="yue")["id"]
        root = dashboard.DATASETS_DIR / dataset_id
        self.write_speech(self.root / "a.wav")
        dashboard.add_dataset_clip(root, self.root / "a.wav", "a", "a.wav", "你哋好。")
        for model, compared in [("", False), ("large-v3", True)]:
            with self.subTest(model=model), patch.object(dashboard, "DATASET_ASR_MODEL", model), \
                    patch.object(dashboard, "transcribe_dataset_wav", return_value="你们好。") as transcribe:
                dashboard.start_dataset_check(dataset_id)
                job = dashboard.JOBS["dataset-check"]
                job.thread.join(timeout=30)
                self.assertEqual(transcribe.called, compared)
                self.assertEqual("large-v3 dataset_asr_model" in job.lines[-1], not compared, job.lines[-1])

    def test_cutting_finds_sparse_speech_and_splits_long_speech(self):
        rng = __import__("random").Random(3)
        # Ten minutes of quiet hiss with eight 3 s lines.
        levels = [-70 + rng.uniform(-2, 2) for _ in range(30000)]
        for line in range(8):
            start = 1500 + line * 3500
            levels[start : start + 150] = [-20 + rng.uniform(-3, 3) for _ in range(150)]
        clips = dashboard.plan_clips(levels)
        self.assertEqual(len(clips), 8)
        self.assertTrue(all(3.0 < end - start < 3.5 for start, end in clips), clips)
        # A minute of speech with no pause of 0.1 s is cut where it is quietest, into clips of 10 s or less.
        levels = [-20.0 if index % 4 else -28.0 for index in range(3000)]
        levels[1234] = -40.0
        clips = dashboard.plan_clips(levels)
        self.assertTrue(all(end - start <= 10.0 for start, end in clips), clips)
        self.assertIn(round(1234 * dashboard.CUT_STEP, 3), [end for _, end in clips])
        self.assertEqual(sum(end - start for start, end in clips), 60.0)
        # Nothing stands out: no speech to cut.
        self.assertEqual(dashboard.plan_clips([-30.0 + rng.uniform(-2, 2) for _ in range(3000)]), [])

    def test_a_program_that_runs_too_long_is_ended(self):
        def work(job):
            job.run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)

        job = dashboard.ThreadJob("Slow", work)
        started = time.time()
        job.start()
        job.thread.join(timeout=20)
        self.assertLess(time.time() - started, 10)
        self.assertEqual(job.status, "failed")
        self.assertIn("did not finish within 1 seconds", job.lines[-1])

    def test_one_long_job_uses_the_gpu_at_a_time(self):
        class Running:
            uses_gpu = True

            def running(self):
                return True

        self.make_dataset()
        with patch.dict(dashboard.JOBS, {"restore": Running()}, clear=True), patch.object(dashboard, "free_gpu_for") as free:
            for start in (lambda: dashboard.start_train("sovits", 1, 1, 1, "clip-test"), lambda: dashboard.prepare_tts_dataset_job("clip-test", "voice")):
                with self.subTest(start=start), patch.object(dashboard, "run_wsl", return_value=subprocess.CompletedProcess([], 0)), \
                        patch.object(dashboard, "dataset_summary", return_value={"ready": True}):
                    with self.assertRaisesRegex(RuntimeError, "Removing music or echo is using the GPU"):
                        start()
            with self.assertRaisesRegex(RuntimeError, "Removing music or echo is using the GPU"):
                dashboard.start_compare("voice", "gpt", "clip-test", "x.wav", "Hello.")
            free.assert_not_called()
        with patch.dict(dashboard.JOBS, {"compare": Running()}, clear=True), patch.object(dashboard, "RESTORE_DIR", self.root / "restored"):
            with self.assertRaisesRegex(RuntimeError, "Compare epochs is using the voice engine"):
                dashboard.start_restore(["x"], {"music": "best"})

    @unittest.skipUnless(shutil.which("ffmpeg"), "Needs ffmpeg")
    def test_one_unreadable_recording_does_not_stop_the_others(self):
        dataset_id = self.make_dataset()
        (self.root / "tmp").mkdir()
        bad, good = self.root / "tmp" / "upload-bad.audio", self.root / "tmp" / "upload-good.audio"
        bad.write_bytes(b"not audio at all")
        rate = 44100
        tone = [6000 * ((i // 40) % 2 * 2 - 1) for i in range(rate * 4)]
        with wave.open(str(good), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(struct.pack(f"<{len(tone) + rate}h", *tone, *[0] * rate))
        with patch.object(dashboard, "TMP_DIR", self.root / "tmp"), patch.object(dashboard, "CUT_DIR", self.root / "cut"),                 patch.object(dashboard, "transcribe_dataset_wav", return_value="Words."):
            dashboard.start_dataset_cut(dataset_id, [(bad, "broken.mp3"), (good, "fine.wav")])
            job = dashboard.JOBS["dataset-cut"]
            job.thread.join(timeout=60)
        self.assertEqual(job.status, "failed")
        self.assertIn("Done. 1 clip to check in the list below.", job.lines)
        self.assertEqual(job.lines[-1], "Failed: 1 recording could not be cut: broken.mp3.")
        self.assertEqual([clip["name"] for clip in job.results], ["fine-001.wav"])
        self.assertEqual(list((self.root / "tmp").iterdir()), [])

    def test_a_failed_import_leaves_no_half_made_voice(self):
        datasets = self.root / "datasets"
        datasets.mkdir()
        gsv = self.root / "gsv"
        archive = self.root / "voice.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("voice.json", json.dumps({"format": 1, "name": "Joe", "language": "en", "version": "v2Pro", "gpt": "joe.ckpt", "sovits": "joe.pth",
                                                      "reference": {"file": "reference.wav", "text": "Hello.", "language": "en"}, "extra_references": []}))
            bundle.writestr("models/joe.ckpt", b"gpt")
            bundle.writestr("models/joe.pth", b"sovits")
            bundle.writestr("reference.wav", b"not audio")
        with (
            patch.object(dashboard, "DATASETS_DIR", datasets),
            patch.object(dashboard, "WSL_GSV_WIN", gsv),
            patch.object(dashboard, "TMP_DIR", self.root / "tmp"),
            patch.object(dashboard, "run_wsl", return_value=subprocess.CompletedProcess([], 0, "", "")),
            patch.object(dashboard, "convert_audio", side_effect=RuntimeError("Invalid data found when processing input")),
        ):
            with self.assertRaisesRegex(RuntimeError, "Invalid data"):
                dashboard.import_voice(archive)
        self.assertEqual(list(datasets.iterdir()), [])

    def test_stop_reaches_a_separation_left_by_an_earlier_run(self):
        port = self.start_server()
        with patch.dict(dashboard.JOBS, {}, clear=True), patch.object(dashboard, "stop_in_wsl") as stop_in_wsl:
            status = self.request(port, "POST", "/api/stop-job", body=json.dumps({"key": "restore"}), Host=f"127.0.0.1:{port}")[0]
        self.assertEqual(status, 200)
        stop_in_wsl.assert_called_once_with(dashboard.WSL_STOP_SCRIPTS["separate"])

    def test_loading_one_speech_model_does_not_hold_up_another(self):
        release = threading.Event()

        class Slow:
            def __init__(self, name, **kwargs):
                if name == "small.en":
                    release.wait(10)

        fake = MagicMock(WhisperModel=Slow)
        with patch.dict(sys.modules, {"faster_whisper": fake}), patch.dict(relay_asr_server.MODELS, {}, clear=True), \
                patch.object(relay_asr_server, "MODEL_DIR", self.root / "models"):
            loading = threading.Thread(target=relay_asr_server.load_model, args=("small.en",))
            loading.start()
            time.sleep(0.3)
            started = time.time()
            relay_asr_server.load_model("base.en")
            self.assertLess(time.time() - started, 2)
            release.set()
            loading.join(5)
            self.assertEqual(sorted(relay_asr_server.MODELS), ["base.en", "small.en"])

    def test_clips_reach_a_dataset_on_another_drive(self):
        # os.replace cannot move a file to another drive, as from TMP_DIR on E: to datasets on C:.
        # Here it refuses any move to another folder, which is the same failure.
        dataset_id = self.make_dataset()
        root = dashboard.DATASETS_DIR / dataset_id
        replace = os.replace

        def same_folder_only(source, target):
            if Path(source).parent != Path(target).parent:
                raise OSError(17, "The system cannot move the file to a different disk drive")
            replace(source, target)

        self.write_wav(self.root / "elsewhere.wav", 1.0)
        with patch.object(dashboard.os, "replace", side_effect=same_folder_only):
            dashboard.add_dataset_clip(root, self.root / "elsewhere.wav", "clip", "elsewhere.wav", "Words.")
        self.assertEqual(sorted(path.name for path in (root / "wavs").iterdir()), ["clip.wav"])
        self.assertFalse((self.root / "elsewhere.wav").exists())
