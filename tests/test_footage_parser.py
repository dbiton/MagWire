import unittest
import numpy as np
import pandas as pd

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from magwire_footage_generator import MagwireFootageGenerator
from footage_parser import FootageParser
from modify_video import *

res = np.array([1280, 720])
margin = 100

def mean_distance(actual, predicted):
    pred_times = np.array([t for _, t in predicted])
    pred_positions = np.array([p for p, _ in predicted])
    actual_times = np.array([t for _, t in actual])
    actual_positions = np.array([p for p, _ in actual])
    indices = np.searchsorted(actual_times, pred_times, side='right') - 1
    indices = np.clip(indices, 0, len(actual_times) - 1)
    matched_actual = actual_positions[indices]
    distances = np.linalg.norm(pred_positions - matched_actual, axis=1)
    return distances.mean()
    
class TestFootageParser(unittest.TestCase):
    def test_panning(self):
        video_path = "test_panning.mp4"
        footage_generator = MagwireFootageGenerator(width=res[0], height=res[1], corner_margin=margin)
        footage_parser = FootageParser()
        actual_pos = footage_generator.generate_circle(300, 1, 10, video_path)
        apply_panning_to_video(video_path, 64, -32)
        pred_pos = [v for v in footage_parser.parse_video(video_path, True, False, False)]
        os.remove(video_path)
        error = mean_distance(actual_pos, pred_pos)
        self.assertLess(error, 0.05)

    def test_circle(self):
        video_path = "test_circle.mp4"
        footage_generator = MagwireFootageGenerator(width=res[0], height=res[1], corner_margin=margin)
        footage_parser = FootageParser()
        actual_pos = footage_generator.generate_circle(300, 1, 10, video_path)
        pred_pos = [v for v in footage_parser.parse_video(video_path, True, False, False)]
        os.remove(video_path)
        error = mean_distance(actual_pos, pred_pos)
        self.assertLess(error, 0.05)
    
    def test_brownian(self):
        video_path = "test_brownian.mp4"
        footage_generator = MagwireFootageGenerator(width=res[0], height=res[1], corner_margin=margin)
        footage_parser = FootageParser()
        actual_pos = footage_generator.generate_brownian((100, 1000), (200, 700), 10, 10, video_path)
        pred_pos = [v for v in footage_parser.parse_video(video_path, True, False, False)]
        os.remove(video_path)
        error = mean_distance(actual_pos, pred_pos)
        self.assertLess(error, 0.05)
    
    def test_noise(self):
        video_path = "test_noise.mp4"
        footage_generator = MagwireFootageGenerator(width=res[0], height=res[1], corner_margin=margin)
        footage_parser = FootageParser()
        actual_pos = footage_generator.generate_circle(300, 1, 10, video_path)
        apply_noise_to_video(video_path, 80)
        pred_pos = [v for v in footage_parser.parse_video(video_path, True, False, False)]
        os.remove(video_path)
        error = mean_distance(actual_pos, pred_pos)
        self.assertLess(error, 0.05)
    
    def test_view_angle(self):
        video_path = "test_view_angle.mp4"
        footage_generator = MagwireFootageGenerator(width=res[0], height=res[1], corner_margin=margin)
        footage_parser = FootageParser()
        actual_pos = footage_generator.generate_circle(300, 1, 10, video_path)
        apply_viewing_angle_to_video(video_path, 0.1)
        pred_pos = [v for v in footage_parser.parse_video(video_path, True, False, False)]
        os.remove(video_path)
        error = mean_distance(actual_pos, pred_pos)
        self.assertLess(error, 0.05)

if __name__ == "__main__":
    unittest.main()
