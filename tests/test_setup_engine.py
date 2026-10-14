import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import setup_engine  # noqa: E402


class SetupEngineTests(unittest.TestCase):
    def test_each_graphics_card_gets_its_build(self):
        for name, choice in [
            ("NVIDIA GeForce RTX 4090", "rtx"), ("NVIDIA RTX A4000", "rtx"), ("NVIDIA GeForce RTX 2060", "rtx"),
            ("NVIDIA GeForce GTX 1660 SUPER", "gtx16"), ("NVIDIA GeForce GTX 1080 Ti", "pascal"), ("NVIDIA TITAN V", "pascal"),
            ("NVIDIA GeForce GTX 970", "cpu"), ("AMD Radeon RX 7900 XTX", "amd"), ("AMD Radeon RX 9070 XT", "amd"),
            ("AMD Radeon PRO W7900", "amd"), ("AMD Radeon RX 6800 XT", "cpu"), ("AMD Radeon(TM) Graphics", "cpu"),
            ("Intel(R) UHD Graphics 770", "cpu"), ("Virtual Desktop Monitor", "cpu"),
            # Laptop cards: MX150 to MX350 are Pascal, MX130 and 940MX older, MX450 and later newer.
            ("NVIDIA GeForce MX150", "pascal"), ("NVIDIA GeForce MX250", "pascal"), ("NVIDIA GeForce MX350", "pascal"),
            ("NVIDIA GeForce MX130", "cpu"), ("NVIDIA GeForce 940MX", "cpu"), ("NVIDIA GeForce MX450", "rtx"),
        ]:
            with self.subTest(name=name):
                self.assertEqual(setup_engine.choice_for_card(name), choice)

    def test_the_best_card_in_a_pc_is_chosen(self):
        self.assertEqual(setup_engine.detect_choice(["Intel(R) UHD Graphics 770", "NVIDIA GeForce RTX 3060"]), "rtx")
        self.assertEqual(setup_engine.detect_choice(["Virtual Desktop Monitor", "AMD Radeon RX 7900 XTX"]), "amd")
        self.assertEqual(setup_engine.detect_choice([]), "cpu")

    def test_config_points_at_the_install_and_keeps_other_settings(self):
        config = setup_engine.engine_config({"dashboard_port": 8800, "tts_device": "cuda"}, "Ubuntu-24.04", "/home/ana", "cpu")
        self.assertEqual(config, {
            "dashboard_port": 8800, "wsl_distro": "Ubuntu-24.04", "wsl_gpt_sovits_root": "/home/ana/GPT-SoVITS",
            "wsl_python": "/home/ana/miniforge3/envs/gsv/bin/python", "wsl_datasets_root": "/home/ana/voice-datasets",
            "tts_device": "cpu", "tts_precision": "fp32",
        })
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            setup_engine.write_config(path, config)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), config)
            self.assertEqual(setup_engine.read_config(path), config)
            # Without config.json, the example is the start.
            self.assertIn("wsl_distro", setup_engine.read_config(Path(folder) / "missing.json"))
            # Notepad can save it with a byte order mark, which the dashboard accepts too.
            path.write_bytes(b"\xef\xbb\xbf" + json.dumps(config).encode("utf-8"))
            self.assertEqual(setup_engine.read_config(path), config)
            path.write_text('{"wsl_distro": "C:\\bad\\path\\"}', encoding="utf-8")
            with patch("builtins.print") as printed:
                self.assertEqual(setup_engine.main(["--config", str(path)]), 1)
            self.assertIn("is not valid JSON", str(printed.call_args_list[-1].args[0]))

    def test_wsl_output_is_read_in_either_encoding(self):
        self.assertEqual(setup_engine.wsl_text("Ubuntu-24.04\r\n".encode("utf-16-le")), "Ubuntu-24.04\r\n")
        self.assertEqual(setup_engine.wsl_text(b"/home/ana\n"), "/home/ana\n")

    def test_without_the_distro_setup_says_how_to_install_it(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(setup_engine, "installed_distros", return_value=["Ubuntu"]), \
                patch("builtins.print") as printed:
            self.assertEqual(setup_engine.main(["--config", str(Path(folder) / "config.json"), "--distro", "Ubuntu-24.04"]), 1)
        self.assertIn("wsl --install -d Ubuntu-24.04", " ".join(str(call.args[0]) for call in printed.call_args_list))

    def run_setup(self, answer):
        """main() with WSL answered by answer(phase or "torch"), which returns an exit status and output."""
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if "wslpath" in command:
                return subprocess.CompletedProcess(command, 0, b"/mnt/c/lvs/scripts/setup_engine.sh\n", b"")
            phase = command[command.index("bash") + 2] if "bash" in command else "torch"
            code, output = answer(phase)
            return subprocess.CompletedProcess(command, code, output, b"")

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(setup_engine, "installed_distros", return_value=["Ubuntu-24.04"]), \
                patch.object(setup_engine, "linux_identity", return_value=("ana", "/home/ana")), \
                patch.object(setup_engine, "graphics_cards", return_value=[]), \
                patch.object(setup_engine.subprocess, "run", side_effect=run), patch("builtins.print") as printed:
            config = Path(folder) / "config.json"
            code = setup_engine.main(["--config", str(config), "--build", "cpu", "--yes"])
            written = config.exists()
        phases = [command[command.index("bash") + 2] for command in calls if "bash" in command]
        return code, written, phases, " ".join(str(call.args[0]) for call in printed.call_args_list if call.args), calls

    def test_an_install_of_the_users_own_is_left_alone(self):
        # The check finds a GPT-SoVITS it did not install, before root installs anything.
        code, written, phases, said, _ = self.run_setup(lambda phase: (setup_engine.OWN_INSTALL if phase == "check" else 0, b""))
        self.assertEqual((code, written, phases), (1, False, ["check"]))
        self.assertIn("from an install of your own", said)
        # Exit status 3, as wget gives for a full disk, is an error like any other.
        code, written, phases, said, _ = self.run_setup(lambda phase: (3 if phase == "user" else 0, b""))
        self.assertEqual((code, written, phases), (1, False, ["check", "root", "user"]))
        self.assertIn("Setup stopped with an error", said)

    def test_a_full_install_writes_config(self):
        code, written, phases, said, calls = self.run_setup(lambda phase: (0, b"False\n" if phase == "torch" else b""))
        self.assertEqual((code, written, phases), (0, True, ["check", "root", "user"]))
        self.assertIn(["-u", "root"], [command[3:5] for command in calls if "bash" in command])

    def test_a_pytorch_that_does_not_load_leaves_config_alone(self):
        code, written, _, said, _ = self.run_setup(lambda phase: (1, b"ImportError: libamdhip64.so.7") if phase == "torch" else (0, b""))
        self.assertEqual((code, written), (1, False))
        self.assertIn("PyTorch does not load", said)
        self.assertIn("libamdhip64.so.7", said)


if __name__ == "__main__":
    unittest.main()
