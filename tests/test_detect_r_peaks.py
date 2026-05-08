import os
import unittest
from pathlib import Path

import numpy as np

from r_peak_detection import detect_r_peaks


class DetectRPeaksTests(unittest.TestCase):
    def test_sampling_rate_uses_t_ecg_when_available(self):
        mat = {"t_ECG": np.array([[0.0, 0.001, 0.002, 0.003]])}

        self.assertEqual(detect_r_peaks.get_sampling_rate(mat), 1000)

    def test_sampling_rate_falls_back_without_valid_time(self):
        self.assertEqual(detect_r_peaks.get_sampling_rate({}), detect_r_peaks.DEFAULT_FS)
        self.assertEqual(
            detect_r_peaks.get_sampling_rate({"t_ECG": np.array([[1.0, 1.0]])}),
            detect_r_peaks.DEFAULT_FS,
        )

    def test_default_checkpoint_is_bundled(self):
        checkpoint_path = Path(detect_r_peaks.get_default_checkpoint_path())

        self.assertTrue(checkpoint_path.is_file())

    def test_config_path_can_be_overridden_by_environment(self):
        config_path = Path("custom-config.json")
        old_value = os.environ.get("R_PEAK_DETECTION_CONFIG")
        os.environ["R_PEAK_DETECTION_CONFIG"] = str(config_path)
        try:
            self.assertEqual(detect_r_peaks.get_config_path(), config_path)
        finally:
            if old_value is None:
                os.environ.pop("R_PEAK_DETECTION_CONFIG", None)
            else:
                os.environ["R_PEAK_DETECTION_CONFIG"] = old_value


if __name__ == "__main__":
    unittest.main()
