import os
import random
from typing import Callable, Tuple
import numpy as np
import pygame
import math
import time
import shutil

COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_YELLOW = (255, 255, 0)

PosFuncType = Callable[[float], Tuple[float, float]]

def delete_folder(folder_path: str):
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' and its contents have been deleted.")
        except Exception as e:
            print(f"Error deleting folder '{folder_path}': {e}")
    else:
        print(f"Folder '{folder_path}' does not exist.")

class MagwireFootageGenerator:
    def __init__(self, width = 1280, height = 720, fps = 30, corner_margin = 50, grid_size = 50, corner_dot_size = 32, magwire_dot_size = 16):
        self.width = width
        self.height = height
        self.margin = corner_margin
        self.grid_size = grid_size
        self.magwire_dot_size = magwire_dot_size
        self.corner_dot_size = corner_dot_size
        self.fps = fps
    
    def generate_circle(self, radius: float, angular_velocity: float, duration_seconds: float, output_path: str):
        center = (self.width / 2, self.height / 2)
        magwire_pos = lambda t: (center[0] + radius * math.cos(angular_velocity * t),  center[1] + radius * math.sin(angular_velocity * t))
        return self.generate(magwire_pos, duration_seconds, output_path)
    
    def generate_brownian(self, x_limit: Tuple[float, float], y_limit: Tuple[float, float], speed: float, duration_seconds: float, output_path: str):
        center = (self.width / 2, self.height / 2)
        def magwire_pos(t):
            if t == 0:
                magwire_pos.current_x = center[0]
                magwire_pos.current_y = center[1]
            delta_x = random.uniform(-speed, speed)
            delta_y = random.uniform(-speed, speed)
            magwire_pos.current_x = max(x_limit[0], min(x_limit[1], magwire_pos.current_x + delta_x))
            magwire_pos.current_y = max(y_limit[0], min(y_limit[1], magwire_pos.current_y + delta_y))
            return magwire_pos.current_x, magwire_pos.current_y
        magwire_pos.current_x = center[0]
        magwire_pos.current_y = center[1]
        return self.generate(magwire_pos, duration_seconds, output_path)
    
    def process_frames_into_video(self, frames_path: str, output_path: str):
        os.system(f"ffmpeg -r {self.fps} -i {frames_path}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p {output_path}")
    
    def generate(self, magwire_pos: PosFuncType, duration_seconds: float, output_path: str):
        frames_folder = f".{output_path}_frames"
        delete_folder(frames_folder)
        os.makedirs(frames_folder)
        pygame.init()
        screen = pygame.Surface((self.width, self.height))
        corners = [
            (self.margin, self.margin),                             # top-left corner
            (self.width - self.margin, self.margin),                # top-right corner
            (self.margin, self.height - self.margin),               # bottom-left corner
            (self.width - self.margin, self.height - self.margin)   # bottom-right corner
        ]
        magwire_positions = {}
        for i_frame, total_time in enumerate(np.arange(0, duration_seconds, 1 / self.fps)):
            screen.fill(COLOR_WHITE)
            # Draw grid
            for x in range(self.margin, self.width - self.margin, self.grid_size):
                pygame.draw.line(screen, COLOR_BLUE, (x, self.margin), (x, self.height - self.margin))
            for y in range(self.margin, self.height - self.margin, self.grid_size):
                pygame.draw.line(screen, COLOR_BLUE, (self.margin, y), (self.width - self.margin, y))
            # Draw green dots on the corners
            for corner in corners:
                pygame.draw.circle(screen, COLOR_GREEN, corner, self.corner_dot_size)
            pos = magwire_pos(total_time)
            magwire_positions[total_time] = (pos[0] - self.margin, pos[1] - self.margin)
            pygame.draw.circle(screen, COLOR_YELLOW, (int(pos[0]), int(pos[1])), self.magwire_dot_size)
            pygame.image.save(screen, f"{frames_folder}/frame_{i_frame:04d}.png")
        pygame.quit()
        self.process_frames_into_video(frames_folder, output_path)
        delete_folder(frames_folder)
        return magwire_positions