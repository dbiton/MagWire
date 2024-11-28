import numpy as np


class Building_Blocks(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''

    def __init__(self, transform, ur_params, env, resolution=0.1, p_bias=0.05):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

    def sample(self, goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        """
        # hint - use self.ur_params.mechamical_limits
        probabilities = [self.p_bias, 1 - self.p_bias]
        is_goal_choice = np.random.choice([True, False], p=probabilities)
        if is_goal_choice:
            return np.array(goal_conf)
        else:
            result_conf = []
            for interval in self.ur_params.mechamical_limits.values():
                low, high = interval
                result_conf.append(np.random.uniform(low, high))

            return np.array(result_conf)

        # return np.array(conf)

    def is_in_collision(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration
        """
        # hint: use self.transform.conf2sphere_coords(), self.ur_params.sphere_radius, self.env.obstacles
        global_sphere_coords = self.transform.conf2sphere_coords(conf)
        # arm - arm collision
        links = list(global_sphere_coords.keys())
        for i in range(len(links)):  # iterate on links
            link_i = global_sphere_coords[links[i]]
            radius_i = self.ur_params.sphere_radius[links[i]]

            for j in range(i + 2, len(links)):  # iterate on next links
                link_j = global_sphere_coords[links[j]]
                radius_j = self.ur_params.sphere_radius[links[j]]

                # now we compare each one of the spheres in one link with another link's spheres
                for sphere_i in link_i:
                    for sphere_j in link_j:
                        # now check collision based on dist(center_i,center_j) < r_i+r_j
                        centers_dist = np.linalg.norm(sphere_i[0:3] - sphere_j[0:3])
                        if (centers_dist <= radius_i + radius_j):
                            # print(f"collision in "+ links[i]+" and "+links[j])
                            return True

        # arm - obstacle collision

        # floor
        for i in range(1, len(links)):  # iterate on links without the first link which is base
            link_i = global_sphere_coords[links[i]]
            radius_i = self.ur_params.sphere_radius[links[i]]
            for sphere_i in link_i:
                if (sphere_i[2] - radius_i <= 0):
                    return True

        # obstacle
        obstable_sphere_radius = self.env.radius
        for i in range(len(links)):  # iterate on links
            link_i = global_sphere_coords[links[i]]
            radius_i = self.ur_params.sphere_radius[links[i]]
            for sphere_i in link_i:
                for obstable_sphere in self.env.obstacles:
                    centers_dist = np.linalg.norm(sphere_i[0:3] - np.array(obstable_sphere))
                    if (centers_dist <= radius_i + obstable_sphere_radius):
                        # print("obstable collision")
                        return True

        return False

    def local_planner(self, prev_conf, current_conf):
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        res = 3
        if self.resolution < 3:
            res = 2
        else:
            res = self.resolution

        # check prev_conf validity
        if self.is_in_collision(prev_conf) is True:
            return False

        # check current_conf validity
        if self.is_in_collision(current_conf) is True:
            return False

        # check intermediate configurations collision
        intermediate = 10  # num of intermidiate configurations to check
        divide_by = intermediate + 1
        if intermediate > 0:
            delta_conf = (current_conf - prev_conf) / divide_by
            for i in range(1, intermediate + 1):
                check_intermediate_conf = prev_conf + delta_conf * i
                if self.is_in_collision(check_intermediate_conf) is True:
                    return False

        return True

    def edge_cost(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5





