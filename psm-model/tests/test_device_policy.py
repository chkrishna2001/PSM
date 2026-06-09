from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from psm_model.device_policy import enforce_local_device_policy


class DevicePolicyTests(unittest.TestCase):
    def test_cuda_forced_to_cpu_locally(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(enforce_local_device_policy("cuda"), "cpu")

    def test_cuda_allowed_on_runpod(self) -> None:
        with patch.dict(os.environ, {"PSM_RUNPOD": "1"}, clear=True):
            self.assertEqual(enforce_local_device_policy("cuda"), "cuda")


if __name__ == "__main__":
    unittest.main()
