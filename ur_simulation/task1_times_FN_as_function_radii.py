
import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from building_blocks import Building_Blocks
import time
import matplotlib.pyplot as plt

def main():
    inflation_factors = np.linspace(1.0, 1.8, 9)
    times = []
    is_collision_instances = []
    for inflation_factor in inflation_factors:
        print(f"iteration inflation factor: ",inflation_factor)
        ur_params = UR5e_PARAMS(inflation_factor=inflation_factor)
        env = Environment(env_idx=0)
        transform = Transform(ur_params)
        bb = Building_Blocks(transform=transform, ur_params=ur_params, env=env, resolution=0.1, p_bias=0.03) 
        # change the path 
        random_samples = np.load('./random_samples/'+'random_samples_100k.npy')
        
        start_time = time.time()
        collision_counter = 0
        for conf in random_samples:
            if bb.is_in_collision(conf) is True:
                collision_counter+=1
        finish_time = time.time()
        is_collision_instances.append(collision_counter)
        time_needed = finish_time-start_time
        times.append(time_needed)

    inf_one_collision = is_collision_instances[0]
    is_collision_instances = [x - inf_one_collision for x in is_collision_instances]
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlabel('min radii factor')
    ax2 = ax1.twinx()
    ax1.set_ylabel('time (s)', color='blue')
    ax2.set_ylabel('False Negative Instances', color='red') 
    ax1.scatter(inflation_factors, times, c='blue')
    ax2.scatter(inflation_factors, is_collision_instances, c='red')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    fig.tight_layout()
    plt.show()




if __name__ == '__main__':
    main()



