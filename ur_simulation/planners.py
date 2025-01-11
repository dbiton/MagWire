import numpy as np
import time
from RRTTree import RRTTree

class RRT_CONNECT(object):
    def __init__(self, max_step_size, max_itr, bb):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.tree_start = RRTTree(bb)
        self.tree_goal = RRTTree(bb)
        self.flip = 0
        self.connection_conf = None

    def find_path(self, start_conf, goal_conf, filename):
        """Implement RRT-Connect"""

        time_start = time.time()
        self.tree_start.AddVertex(start_conf)
        self.tree_goal.AddVertex(goal_conf)

        for i in range(self.max_itr):
            print(i)
            # Sample random state
            x_random = self.tree_start.bb.sample(goal_conf=goal_conf)

            # Extend tree_start
            x_new_start = self.extend_tree(self.tree_start, x_random)

            if x_new_start is not None:
                # Try to connect to tree_goal
                if self.connect_trees(self.tree_goal, x_new_start):
                    # Path found
                    time_end = time.time() - time_start
                    path = self.get_path()
                    path_cost = self.compute_cost(path)

                    print(f"Path found in {i} iterations.")
                    print(f"Total cost: {path_cost:.2f}, Total time: {time_end:.2f}s")
                    return path, path_cost, time_end

            # Swap trees
            self.tree_start, self.tree_goal = self.tree_goal, self.tree_start
            self.flip = (self.flip+1)%2

        print("No path was found.")
        return None, None, None

    def extend_tree(self, tree, x_random):
        """
        Extend the given tree towards the random sample.
        """
        x_near_idx, x_near = tree.GetNearestVertex(config=x_random)
        x_new = self.extend(x_near, x_random)

        if not tree.bb.is_in_collision(conf=x_new) and tree.bb.local_planner(prev_conf=x_near, current_conf=x_new):
            x_new_id = tree.AddVertex(x_new)
            tree.AddEdge(x_near_idx,x_new_id, edge_cost=tree.bb.edge_cost(conf1=x_near, conf2=x_new))
            return x_new
        return None

    def connect_trees(self, tree, x_new):
        """
        Attempt to connect a new node to the other tree.
        """
        print("in connect tree")
        x_near_idx, x_near = tree.GetNearestVertex(config=x_new)

        while True:
            x_new_connect = self.extend(x_near, x_new)
            #print("x_new_connect =", x_new_connect)
            #print("x_near =", x_near)

            if tree.bb.is_in_collision(conf=x_new_connect) or not tree.bb.local_planner(prev_conf=x_near, current_conf=x_new_connect):
                return False

            x_new_id = tree.AddVertex(x_new_connect)
            tree.AddEdge(x_near_idx,x_new_id, edge_cost=tree.bb.edge_cost(conf1=x_near, conf2=x_new_connect))

            if np.array_equal(x_new_connect, x_new):
                self.connection_conf = x_new
                return True

            x_near = x_new_connect

    def extend(self, x_near, x_random):
        """
        Implement the Extend method.
        """
        dist = self.tree_start.bb.edge_cost(conf1=x_random, conf2=x_near)

        if dist > self.max_step_size:
            num_of_points = int(dist / self.max_step_size)
            x_steps = np.linspace(x_near, x_random, num_of_points + 1)
            return x_steps[1]
        else:
            return x_random

    # def get_path(self):
    #     """
    #     Retrieve the path from the two trees.
    #     """
    #     path_start = []
    #     current = self.tree_start.GetRootID()
    #     print("get path tree start : ")
    #     print("tree start edges: ", self.tree_start.edges)
    #     while current in self.tree_start.edges:
    #         print("current is ",current)
    #         print("edge is ",self.tree_start.edges[current])
    #         path_start.append(self.tree_start.vertices[current].conf)
    #         current = self.tree_start.edges[current]
    #     path_start.append(self.tree_start.vertices[current].conf)
    #     #print("path start is: ", path_start)
    #
    #
    #
    #     path_goal = []
    #     current = self.tree_goal.GetRootID()
    #     print("get path tree goal : ")
    #     print("tree end edges: ", self.tree_goal.edges)
    #     while current in self.tree_goal.edges:
    #         print("current is ", current)
    #         print("edge is ", self.tree_goal.edges[current])
    #         path_goal.append(self.tree_goal.vertices[current].conf)
    #         current = self.tree_goal.edges[current]
    #     path_goal.append(self.tree_goal.vertices[current].conf)
    #     #print("path goal is: ", path_goal)
    #     #reverse path goal
    #     path_goal = path_goal[::-1]
    #
    #     return np.vstack((np.array(path_start[::-1]), np.array(path_goal)))

    def get_path(self):
        """
        Retrieve the path from the two trees.
        """

        if (self.flip == 1):
            self.tree_start, self.tree_goal = self.tree_goal, self.tree_start

        path_start = []
        current = self.tree_start.getIndexForState(self.connection_conf)
        print("get path tree start : ")
        print("tree start edges: ", self.tree_start.edges)
        print("connection conf idx is : ",current)

        while current in self.tree_start.edges:
            print("current is ",current)
            print("edge is ",self.tree_start.edges[current])
            path_start.append(self.tree_start.vertices[current].conf)
            current = self.tree_start.edges[current]
        path_start.append(self.tree_start.vertices[current].conf)
        print("path start is: ", path_start)



        path_goal = []
        current = self.tree_goal.getIndexForState(self.connection_conf)
        print("get path tree goal : ")
        print("tree end edges: ", self.tree_goal.edges)
        print("connection conf idx is : ", current)

        while current in self.tree_goal.edges:
            print("current is ", current)
            print("edge is ", self.tree_goal.edges[current])
            path_goal.append(self.tree_goal.vertices[current].conf)
            current = self.tree_goal.edges[current]
        path_goal.append(self.tree_goal.vertices[current].conf)
        print("path goal is: ", path_goal)

        #return np.array(path_goal)
        return np.vstack((np.array(path_start[::-1]), np.array(path_goal[1:])))

    def compute_cost(self, path):
        """
        Compute the total cost of the given path.
        """
        path_cost = 0.0
        for i in range(len(path) - 1):
            path_cost += self.tree_start.bb.edge_cost(conf1=path[i], conf2=path[i + 1])
        return path_cost
