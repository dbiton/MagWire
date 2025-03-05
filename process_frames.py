import json
import os

def process_frames(frames_dir, output_file):
    frames_names = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
    frames_times = sorted([float(f[:-4]) for f in frames_names])
    os.system(f'cd {frames_dir} && ffmpeg -framerate 30 -pattern_type glob -i "*.png" -c:v libx264 -pix_fmt yuv420p {output_file}')
    with open(f"{frames_dir}\\frame_times.json", "w") as f:
        json.dump(frames_times, f)

if __name__=="__main__":
    process_frames("frames", "output.mp4")