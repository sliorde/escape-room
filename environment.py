from itertools import cycle
from typing import List
import pickle
import zlib
from os import path

import numpy as np
from scipy.spatial.distance import pdist, squareform

from agent import Robot
from policies import Policy
from rewards import Rewards
from utils import my_arctan2

class Room:
    def __init__(self, width, height, door_width, max_episode_steps, history_save_interval):
        self.width = width
        self.height = height
        self.door_width = door_width
        self.max_episode_steps = max_episode_steps
        self.history_save_interval = history_save_interval

        self.step = 0

        self.robots:List[Robot] = []
        self.inverse_robot_mapping = {}
        self.policies = []

        self.dist_calculation_step = -1
        self.distances_between_robot_centers = None
        self.azimuth_of_vector_differences = None

        self.set_room_positions()

        self.location_history = np.empty((0,0,4))
        self.location_ind = 0


    def set_room_positions(self):
        self.bottom_left = np.array([0.0, 0.0])
        self.bottom_right = np.array([self.width, 0.0])
        self.top_left = np.array([0.0, self.height])
        self.top_right = np.array([self.width, self.height])
        self.door_left = np.array([self.width / 2 - self.door_width / 2, self.height])
        self.door_right = np.array([self.width / 2 + self.door_width / 2, self.height])

    def populate_with_robots(self, num_robots):
        for i in range(len(self.robots), len(self.robots) + num_robots):
            robot = Robot('robot_{:d}'.format(i))
            self.put_robot_somewhere(robot)
            self.robots.append(robot)
            self.inverse_robot_mapping[robot] = i
        self.location_history = np.concatenate((self.location_history,np.full((num_robots,self.location_history.shape[1],4),np.nan)),0)

    def put_robot_somewhere(self, robot):
        found = False
        while not found:
            x = np.random.uniform(robot.radius, self.width - robot.radius)
            y = np.random.uniform(robot.radius, self.height - 1.1*robot.radius)
            found = not self.collision_with_wall(x, y, robot.radius)
            if not found:
                continue
            for other_robot in self.robots:
                if (other_robot is not robot):
                    found = not self.collision_between_robots(x, y, robot.radius, *other_robot.location,
                                                              other_robot.radius)
                    if not found:
                        break
        robot.update_location(np.asarray([x, y]), np.random.uniform(-180, 180), speed=0.0)

    def assign_policies(self, policies):
        if isinstance(policies, Policy):
            policies = [policies]
        for robot, policy in zip(self.robots, cycle(policies)):
            if robot.policy is None:
                robot.assign_policy(policy)
        self.policies += [policy for policy in policies if policy not in self.policies]

    def collision_with_wall(self, robot_x, robot_y, robot_r):
        if robot_x < robot_r:  # detect collision with left wall
            return True
        if robot_x > self.width - robot_r:  # detect collision with right wall
            return True
        if robot_y < robot_r:  # detect collision with bottom wall
            return True
        if (robot_y > (self.height - robot_r)):  # detect collision with top wall (which contains a door)
            if robot_x < self.door_left[0]:
                return True
            if robot_x > self.door_right[0]:
                return True
            if (robot_x - self.door_left[0]) ** 2 + (self.height - robot_y) ** 2 < robot_r ** 2:
                return True
            if (robot_x - self.door_right[0]) ** 2 + (self.height - robot_y) ** 2 < robot_r ** 2:
                return True
        return False

    def collision_between_robots(self, robot1_x, robot1_y, robot1_r, robot2_x, robot2_y, robot2_r):
        return ((robot1_x - robot2_x) ** 2 + (robot1_y - robot2_y) ** 2 < (robot1_r + robot2_r) ** 2)

    def escaped(self, robot_x, robot_y, robot_radius):
        return robot_y > self.height - robot_radius

    def get_dist_and_azimuth_between_robot_centers(self, robot1, robot2):
        if self.step > self.dist_calculation_step:
            locations_matrix = np.asarray([r.location for r in self.robots])
            self.distances_between_robot_centers = squareform(pdist(locations_matrix))
            vector_differences = locations_matrix[None, :, :] - locations_matrix[:, None, :]
            self.azimuth_of_vector_differences = my_arctan2(vector_differences[..., 0], vector_differences[..., 1])
            self.dist_calculation_step = self.step
        ind1 = self.inverse_robot_mapping[robot1]
        ind2 = self.inverse_robot_mapping[robot2]
        return self.distances_between_robot_centers[ind1, ind2], \
               self.azimuth_of_vector_differences[ind1, ind2]

    def check_final_state(self):
        escaped_robots = []
        for robot in self.robots:
            escaped = self.escaped(*robot.location, robot.radius)
            reset = (self.max_episode_steps is not None) and (robot.steps_in_episode > self.max_episode_steps) and not escaped
            if escaped:
                robot.give_reward(reward=Rewards.escape, is_final_state=True)
                robot.update_steps_in_episode(reset=True)
                escaped_robots.append(robot)
                print(robot.name + ' escaped @ {:d}'.format(self.step))
            elif reset:
                robot.give_reward(is_final_state=True)
                robot.update_steps_in_episode(reset=True)
                escaped_robots.append(robot)
                print(robot.name + ' reset @ {:d}'.format(self.step))
            else:
                robot.give_reward(is_final_state=False)
        return escaped_robots

    def update_location_history(self,output_dir):
        if self.location_ind == self.location_history.shape[1]:
            self.location_history = np.concatenate((self.location_history,np.full((self.location_history.shape[0],50000,4),np.nan)),1)
        self.location_history[:,self.location_ind,:] = [list(robot.location)+[robot.direction, robot.prev_reward] for robot in self.robots]
        self.location_ind += 1

        if (self.history_save_interval is not None) and (self.location_ind == self.history_save_interval):
            with open(path.join(output_dir,'history_{:d}.pickle'.format(self.step+1-self.history_save_interval)),'wb') as f:
                f.write(zlib.compress(pickle.dumps(self.location_history[:,:self.location_ind,:])))
                self.location_ind = 0
                self.location_history = np.empty((self.location_history.shape[0],0,4))

    def global_step(self, output_dir, inference=False):

        escaped_robots = self.check_final_state()

        self.update_location_history(output_dir)

        for robot in np.random.permutation(escaped_robots):
            self.put_robot_somewhere(robot)

        for robot in np.random.permutation(self.robots):
            state = robot.get_observed_state(self)
            if robot.prev_action is not None:
                robot.policy.add_to_replay_buffer(robot.prev_state,robot.prev_action,robot.prev_reward,state,robot.prev_is_final_state)
            robot.give_reward(reward=Rewards.neutral, is_final_state=False)
            action = robot.choose_and_perform_action(self, inference)
            robot.update_prev_state_action(state,action)

        for policy in self.policies:
            policy.optimization_step(output_dir)

        self.step += 1