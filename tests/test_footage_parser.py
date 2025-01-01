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
        footage_parser = FootageParser()
        actual_pos = footage_generator.generate_circle(300, 1, 20, video_path)
        actual_pos = np.array(list(actual_pos.values()))
        pred_pos = footage_parser.parse_video(video_path)
        pred_pos = np.array(list(pred_pos.values()))
        os.remove(video_path)
        mean_error = np.mean(np.linalg.norm(pred_pos - actual_pos, axis=1))
        self.assertLess(mean_error, 1)
    
    def test_brownian(self):
        video_path = "test_brownian.mp4"
        footage_generator = MagwireFootageGenerator()
        footage_parser = FootageParser()
        actual_pos = footage_generator.generate_brownian((100, 1000), (200, 700), 30, 10, video_path)
        actual_pos = np.array(list(actual_pos.values()))
        pred_pos = footage_parser.parse_video(video_path)
        pred_pos = np.array(list(pred_pos.values()))
        os.remove(video_path)
        mean_error = np.mean(np.linalg.norm(pred_pos - actual_pos, axis=1))
        self.assertLess(mean_error, 1)
        
if __name__ == "__main__":
    unittest.main()
