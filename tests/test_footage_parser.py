import unittest
import numpy as np
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from magwire_footage_generator import MagwireFootageGenerator
from footage_parser import parse_video

class TestFootageParser(unittest.TestCase):
    def test_circle(self):
        video_path = "test_circle.mp4"
        footage_generator = MagwireFootageGenerator()
        actual_pos = footage_generator.generate_circle(100, 1, 10, video_path)
        pred_pos = parse_video(video_path)
        actual_pos = np.array(list(actual_pos.values()))
        pred_pos = pd.Series([p["pos"] for p in pred_pos], dtype="object")
        pred_pos = pred_pos.fillna(method="ffill")
        if pred_pos.isnull().all():
            pred_pos = np.zeros((300, 2))
        os.remove(video_path)
        mean_error = np.mean(np.linalg.norm(pred_pos - actual_pos, axis=1))
        self.assertLess(mean_error, 10)        
        
if __name__ == "__main__":
    unittest.main()
