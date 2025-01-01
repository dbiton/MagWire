import unittest
import numpy as np
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from magwire_footage_generator import MagwireFootageGenerator
from footage_parser import FootageParser

class TestFootageParser(unittest.TestCase):
    def test_circle(self):
        video_path = "test_circle.mp4"
        footage_generator = MagwireFootageGenerator()
        footage_parser = FootageParser(1280 - 100, 720 - 100)
        actual_pos = footage_generator.generate_circle(360-50, 1, 10, video_path)
        pred_pos = footage_parser.parse_video(video_path)
        actual_pos = np.array(list(actual_pos.values()))
        pred_pos = pd.Series([p["pos"] for p in pred_pos], dtype="object")
        pred_pos = pred_pos.fillna(method="ffill")
        pred_pos = pred_pos.fillna(method="bfill")
        if pred_pos.isnull().all():
            pred_pos = np.zeros((300, 2))
        else:
            pred_pos = np.array(list(pred_pos))
        os.remove(video_path)
        mean_error = np.mean(np.linalg.norm(pred_pos - actual_pos, axis=1))
        self.assertLess(mean_error, 1)        
        
if __name__ == "__main__":
    unittest.main()
