import json
from footage_parser import parse_video

def load_robot_pos(filepath: str):
  with open(filepath, "r") as f:
    return json.load(f)

def load_magwire_pos():
    wire_pos = parse_video()
    return wire_pos

robot_pos = load_robot_pos('data\\28.11.24\\robot_pos.json')
wire_pos = load_magwire_pos()
x = 3