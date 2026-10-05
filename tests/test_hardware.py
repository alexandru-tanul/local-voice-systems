import unittest
from unittest.mock import Mock

from voice_hardware import detect_runtime


def gpu(name="NVIDIA GeForce RTX 5070", capability=(12, 0)):
    torch = Mock()
    torch.version.hip = None
    torch.cuda.is_available.return_value = True
    torch.cuda.device_count.return_value = 2
    torch.cuda.get_device_name.return_value = name
    torch.cuda.get_device_capability.return_value = capability
    from unittest.mock import MagicMock
    torch.ones.return_value = MagicMock()
    torch.ones.return_value.__matmul__.return_value.isfinite.return_value.all.return_value.item.return_value = True
    return torch


class HardwareTests(unittest.TestCase):
    def test_cpu_never_probes_cuda(self):
        torch = Mock()
        runtime = detect_runtime(torch, device="cpu")
        self.assertEqual(runtime["device"], "cpu")
        self.assertFalse(runtime["is_half"])
        torch.cuda.is_available.assert_not_called()

    def test_no_gpu_falls_back_to_cpu(self):
        torch = gpu()
        torch.cuda.is_available.return_value = False
        self.assertEqual(detect_runtime(torch)["device"], "cpu")

    def test_explicit_cuda_reports_missing_gpu(self):
        torch = gpu()
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "driver and PyTorch"):
            detect_runtime(torch, device="cuda")

    def test_rtx_uses_half_precision(self):
        runtime = detect_runtime(gpu())
        self.assertEqual(runtime["device"], "cuda")
        self.assertTrue(runtime["is_half"])

    def test_pascal_uses_full_precision(self):
        self.assertFalse(detect_runtime(gpu("NVIDIA GeForce GTX 1080", (6, 1)))["is_half"])

    def test_gtx_16_uses_full_precision(self):
        self.assertFalse(detect_runtime(gpu("NVIDIA GeForce GTX 1660 Ti", (7, 5)))["is_half"])

    def test_rocm_uses_gpu(self):
        torch = gpu("AMD Radeon", (11, 0))
        torch.version.hip = "6.4"
        self.assertIn("ROCm", detect_runtime(torch)["detail"])

    def test_fp32_override(self):
        self.assertFalse(detect_runtime(gpu(), precision="fp32")["is_half"])

    def test_selected_gpu_is_exercised(self):
        torch = gpu()
        runtime = detect_runtime(torch, gpu_index=1)
        self.assertEqual(runtime["gpu_index"], "1")
        self.assertEqual(torch.ones.call_args.kwargs["device"], "cuda:1")
        torch.cuda.synchronize.assert_called_once_with(1)

    def test_unsupported_kernel_falls_back(self):
        torch = gpu()
        torch.ones.side_effect = RuntimeError("no kernel image is available")
        runtime = detect_runtime(torch)
        self.assertEqual(runtime["device"], "cpu")
        self.assertIn("no kernel image", runtime["detail"])

    def test_forced_cuda_preserves_kernel_error(self):
        torch = gpu()
        torch.ones.side_effect = RuntimeError("no kernel image is available")
        with self.assertRaisesRegex(RuntimeError, "no kernel image"):
            detect_runtime(torch, device="cuda")

    def test_invalid_settings_fail_clearly(self):
        for options in [dict(device="metal"), dict(precision="int8"), dict(gpu_index=-1), dict(gpu_index=9), dict(device="cpu", precision="fp16")]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                detect_runtime(gpu(), **options)


if __name__ == "__main__":
    unittest.main()
