from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from psm_model.device_policy import enforce_local_device_policy, resolve_device_name


class DevicePolicyTests(unittest.TestCase):
    def test_cpu_stays_cpu(self) -> None:
        self.assertEqual(resolve_device_name("cpu", True), "cpu")

    def test_auto_uses_cpu_locally_even_when_cuda_available(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_device_name("auto", True), "cpu")

    def test_auto_uses_cuda_on_runpod(self) -> None:
        with patch.dict(os.environ, {"PSM_RUNPOD": "1"}, clear=True):
            self.assertEqual(resolve_device_name("auto", True), "cuda")
            self.assertEqual(resolve_device_name("auto", False), "cpu")

    def test_cuda_forced_to_cpu_locally(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_device_name("cuda", True), "cpu")

    def test_cuda_allowed_on_runpod(self) -> None:
        with patch.dict(os.environ, {"PSM_RUNPOD": "1"}, clear=True):
            self.assertEqual(resolve_device_name("cuda", True), "cuda")

    def test_force_cpu_overrides_runpod(self) -> None:
        with patch.dict(os.environ, {"PSM_RUNPOD": "1", "PSM_FORCE_CPU": "1"}, clear=True):
            self.assertEqual(resolve_device_name("auto", True), "cpu")

    def test_enforce_uses_torch_availability_on_runpod(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.dict(os.environ, {"PSM_RUNPOD": "1"}, clear=True):
                self.assertEqual(enforce_local_device_policy("auto"), "cuda")


if __name__ == "__main__":
    unittest.main()
