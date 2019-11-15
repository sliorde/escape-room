from itertools import cycle

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

from agent import Robot
from policies import Policy
from utils import my_arctan2, encode_action

class Room:
    def __init__(self,width,height,door_width):
        self.width = width
        self.height = height
        self.door_width = door_width

        self.robots = []

        self.step = 0

        self.dist_calculation_step = -1
        self.distances_between_robot_centers = None
        self.azimuth_of_vector_differences = None

        self.set_room_positions()

        self.prev_states = []
        self.prev_actions = []

    def set_room_positions(self):
        self.bottom_left = np.array([0.0,0.0])
        self.bottom_right = np.array([self.width, 0.0])
        self.top_left = np.array([0.0, self.height])
        self.top_right = np.array([self.width, self.height])
        self.door_left = np.array([self.width/2-self.door_width/2,self.height])
        self.door_right = np.array([self.width / 2 + self.door_width / 2, self.height])

    def populate_with_robots(self,num_robots):
        for i in range(num_robots):
            robot = Robot('robot_{:d}'.format(i))
            self.put_robot_somewhere(robot)
            self.robots.append(robot)
            self.prev_states.append(None)
            self.prev_actions.append(None)
        self.inverse_robot_mapping = {r:i for (i,r) in enumerate(self.robots)}

    def put_robot_somewhere(self,robot):
        found = False
        while not found:
            found = True
            x = np.random.uniform(robot.radius, self.width - robot.radius)
            y = np.random.uniform(robot.radius, self.height - robot.radius)
            found = not self.collision_with_wall(x,y,robot.radius)
            if not found:
                continue
            for other_robot in self.robots:
                if (other_robot is not robot):
                    found = not self.collision_between_robots(x,y, robot.radius, *other_robot.location, other_robot.radius)
                    if not found:
                        break
        robot.update_location(np.asarray([x,y]),np.random.uniform(-180,180),speed=0.0)

    def assign_policies(self,policies):
        if isinstance(policies,Policy):
            policies = [policies]
        for robot,policy in zip(self.robots,cycle(policies)):
                robot.assign_policy(policy)

    def collision_with_wall(self, robot_x, robot_y, robot_r):
        if robot_x < robot_r: # detect collision with left wall
            return True
        if robot_x > self.width- robot_r:  # detect collision with right wall
            return True
        if robot_y < robot_r: # detect collision with bottom wall
            return True
        if (robot_y > (self.height - robot_r)): # detect collision with top wall (which contains a door)
            if robot_x < self.door_left[0]:
                return True
            if robot_x > self.door_right[0]:
                return True
            if (robot_x - self.door_left[0]) ** 2 + (self.height-robot_y) ** 2 < robot_r**2:
                return True
            if (robot_x - self.door_right[0]) ** 2 + (self.height-robot_y) ** 2 < robot_r**2:
                return True
        return False

    def collision_between_robots(self, robot1_x, robot1_y, robot1_r, robot2_x, robot2_y, robot2_r):
        return ((robot1_x - robot2_x) ** 2 + (robot1_y - robot2_y) ** 2 < (robot1_r+robot2_r) ** 2)

    def escaped(self,robot_x,robot_y,robot_radius):
        return robot_y > self.height

    def get_dist_and_azimuth_between_robot_centers(self, robot1, robot2):
        if self.step > self.dist_calculation_step:
            locations_matrix = np.asarray([r.location for r in self.robots])
            self.distances_between_robot_centers = squareform(pdist(locations_matrix))
            vector_differences = locations_matrix[None,:,:] - locations_matrix[:,None,:]
            self.azimuth_of_vector_differences = my_arctan2(vector_differences[..., 0], vector_differences[..., 1])
            self.dist_calculation_step = self.step
        ind1 = self.inverse_robot_mapping[robot1]
        ind2 = self.inverse_robot_mapping[robot2]
        return self.distances_between_robot_centers[ind1 ,ind2], \
               self.azimuth_of_vector_differences[ind1,ind2]

    def global_step(self):
        inds = np.arange(0,len(self.robots))
        np.random.shuffle(inds)
        escaped_robots = []
        for i in inds:
            robot = self.robots[i]
            robot.observe_and_perform_action(self)
            if self.escaped(*robot.location,robot.radius):
                escaped_robots.append(robot)
        for robot in escaped_robots:
            self.put_robot_somewhere(robot)

        self.step += 1

    def global_step_dqn(self,inference=False):
        inds = np.arange(0,len(self.robots))
        np.random.shuffle(inds)
        escaped_robots = []
        for i in inds:
            robot = self.robots[i]
            speed_action,turn_action,state = robot.observe_and_perform_action(self,inference)
            if self.prev_states[i] is not None:
                robot.policy.replay_buffer.push(torch.tensor(self.prev_states[i]).flatten(), torch.tensor(encode_action(*self.prev_actions[i])), torch.tensor(state).flatten(), torch.tensor(0.0))
            self.prev_states[i] = state.copy()
            self.prev_actions[i] = (speed_action,turn_action)
            if self.escaped(*robot.location,robot.radius):
                escaped_robots.append(robot)
                robot.policy.replay_buffer.push(torch.tensor(state).flatten(),torch.tensor(encode_action(speed_action,turn_action)),None,torch.tensor(1.0))
                self.prev_states[i] = None
            robot.policy.optimization_step()
        for robot in escaped_robots:
            self.put_robot_somewhere(robot)

        self.step += 1
