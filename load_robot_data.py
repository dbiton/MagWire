import json


def load_robot_pos(filepath: str):
  with open(filepath, "r") as f:
    return json.load(f)

def load_magwire_pos(filepath: str):
    pass

robot_pos = load_robot_pos('data\\28.11.24\\robot_pos.json')
x = 3