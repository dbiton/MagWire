import json
from math import radians
import math
from time import sleep, time
from multiprocessing import Process
import winsound

from Robot_Integration import BaseRobot
from numpy import random

def to_rad(p):
  return [radians(a) for a in p]

r = BaseRobot()

def log_data():
  with open("robot_pos.txt", "w") as f:
    while True:
      res = {"time": time(), "pos": r.get_config()}
      sleep(1 / 100)
      json.dump(res, f)

if __name__=="__main__":
  positions_degrees = [
    #[-90, -70, -90, -90, 90, 0],
    [-74.78, -82.06, -101.13, -88.01, 86.98, 10.61],
    [-97.04, -89.09, -94.09, -84.81, 88.36, 172.78],
  ]
  # make last angle random 
  # 50 - 120 second to last
  process = Process(target=log_data)
  process.start()

  pos_corners = [
    (65.07, -100.24, 112, -103.62, 266.73, 65.18),#UR
    (65.16, -89.62, 101.31, -99.55, 273.13, 64.98),#UL
    (56.88, -88.73, 101.28, -104.98, 269.22, 56.72),#BL
    (53.89, -97.6, 110.23, -105.95, 265.53, 54.19) #BR
  ]

  frequency = 500  # Frequency in Hertz
  duration = 5000    # Duration in milliseconds
  winsound.Beep(int(frequency), duration)

  for i in range(1000):
    next_pos_i = random.randint(0, len(positions_degrees))
    next_pos = positions_degrees[next_pos_i]
    next_pos[-1] = random.random() * 310
    next_pos[-2] = random.random() * 70 + 50
    next_pos = to_rad(next_pos)
    r.move(next_pos)
    frequency = 1000  # Frequency in Hertz
    duration = 1000    # Duration in milliseconds
    winsound.Beep(int(frequency), duration)
  process.join()
