import numpy as np
import pymunk

from magwire_simulator import simulate
from model import Model

def evaluate(goal_pos: pymunk.Vec2d, magnet_pos: pymunk.Vec2d, wire_speed: float) -> float:
  end_position = simulate(wire_speed, magnet_pos, goal_pos)
  return (goal_pos-end_position).length

def generate_data(num_samples = 1000):
    X = np.random.uniform(-10, 10, (num_samples, 5))
    y = np.array([evaluate(x, ) for x in X])
    return X, y

def main():
  goal_position = pymunk.Vec2d(300,300)
  magnet_position = pymunk.Vec2d(500,300)
  print(evaluate(goal_position, magnet_position, 15))


if __name__=="__main__":
  main()