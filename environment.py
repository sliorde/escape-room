from itertools import cycle

import numpy as np
from scipy.spatial.distance import pdist, squareform

from agent import Robot
from policies import Policy
from utils import my_arctan2


class Room:
    def __init__(self, width, height, door_width, max_episode_steps):
        self.width = width
        self.height = height
        self.door_width = door_width
        self.max_episode_steps = max_episode_steps

        self.robots = []

        self.step = 0

        self.dist_calculation_step = -1
        self.distances_between_robot_centers = None
        self.azimuth_of_vector_differences = None

        self.set_room_positions()

        self.num_robots = 0
        self.inverse_robot_mapping = {}

    def set_room_positions(self):
        self.bottom_left = np.array([0.0, 0.0])
        self.bottom_right = np.array([self.width, 0.0])
        self.top_left = np.array([0.0, self.height])
        self.top_right = np.array([self.width, self.height])
        self.door_left = np.array([self.width / 2 - self.door_width / 2, self.height])
        self.door_right = np.array([self.width / 2 + self.door_width / 2, self.height])

    def populate_with_robots(self, num_robots):
        for i in range(self.num_robots, self.num_robots + num_robots):
            robot = Robot('robot_{:d}'.format(i))
            self.put_robot_somewhere(robot)
            self.robots.append(robot)
            self.inverse_robot_mapping[robot] = i
        self.num_robots += num_robots

    def put_robot_somewhere(self, robot):
        found = False
        while not found:
            found = True
            x = np.random.uniform(robot.radius, self.width - robot.radius)
            y = np.random.uniform(robot.radius, self.height - robot.radius)
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

    def global_step(self, inference=False):
        inds = np.arange(0, len(self.robots))
        np.random.shuffle(inds)
        escaped_robots = []
        for i in inds:
            robot = self.robots[i]
            speed_action, turn_action, state = robot.observe_and_perform_action(self, inference)
            escaped = self.escaped(*robot.location, robot.radius)
            reset = (robot.steps_in_episode > self.max_episode_steps) and not escaped
            if escaped:
                reward = 1.0
                escaped_robots.append(robot)
                final_state = True
            elif reset:
                reward = 0.0
                robot.update_steps_in_episode(reset=True)
                escaped_robots.append(robot)
                final_state = True
            else:
                reward = 0.0
                final_state = False
            action = (speed_action, turn_action)
            robot.policy.add_to_replay_buffer(state, action, reward, robot)
            if final_state:
                # action and reward are dummies here...
                robot.policy.add_to_replay_buffer(state=None, action=(float('nan'),)*2, reward=float('nan'), robot=robot)
            robot.policy.optimization_step()
        for robot in escaped_robots:
            self.put_robot_somewhere(robot)

        self.step += 1
