from typing import Optional

import numpy as np

from rewards import Rewards
from utils import my_arctan2, fix_angle, encode_action


class Robot:
    radius = 1
    fov_size = 120  # fov = field of view
    num_fov_pixels = 30
    max_speed = 2
    num_speeds = 3
    turn_speed = 10
    curr_frame_ind = 0
    prev_frame_ind = 1
    dist_ind = 0
    wall_ind = 1
    robot_ind = 2
    door_ind = 3

    def __init__(self, name):
        self.name = name
        self.location = None
        self.direction = None
        self.speed = None

        self.policy:Optional['Policy'] = None

        self.steps_in_episode = self.update_steps_in_episode(reset=True)

        self.initialize_state_array()

    def initialize_state_array(self):

        self.state = np.full((self.num_fov_pixels, 2, 1 + 3),
                             np.nan,
                             np.float32)  # 2, because we take two last frames. 1 is the distance to nearest object. 3 is number of object types (wall,robot,door)  Robo

        self.prev_state = self.state.copy()
        self.prev_action = None
        self.prev_reward = None
        self.prev_is_final_state = None

    def update_location(self, location, direction, speed):
        self.location = location
        self.direction = direction
        self.speed = speed

    def get_observed_state(self, room, without_update_self=False):

        if without_update_self:
            state = self.state.copy()
        else:
            state = self.state

        state[:, self.prev_frame_ind, :] = state[:, self.curr_frame_ind, :]
        state[:, self.curr_frame_ind, self.dist_ind] = np.inf
        state[:, self.curr_frame_ind, self.wall_ind] = 1.0
        state[:, self.curr_frame_ind, [self.robot_ind, self.door_ind]] = 0.0

        offset = fix_angle(self.direction) - 180

        limit_right = fix_angle(self.direction - self.fov_size / 2, offset)  # right limit of FOV
        limit_left = limit_right + self.fov_size  # left limit of FOV
        angle_bin_edges = np.linspace(start=limit_right, stop=limit_left, num=self.num_fov_pixels + 1)
        angle_bin_centers = (angle_bin_edges[:-1] + angle_bin_edges[1:]) / 2

        wall_bottom_left = my_arctan2(*(room.bottom_left - self.location), offset)
        wall_bottom_right = my_arctan2(*(room.bottom_right - self.location), offset)
        wall_top_right = my_arctan2(*(room.top_right - self.location), offset)
        wall_top_left = my_arctan2(*(room.top_left - self.location), offset)
        door_right = my_arctan2(*(room.door_right - self.location), offset)
        door_left = my_arctan2(*(room.door_left - self.location), offset)

        self.update_state_with_wall_or_door(state, wall_bottom_left, wall_bottom_right, self.location[1],
                                            270 - angle_bin_centers, angle_bin_edges)
        self.update_state_with_wall_or_door(state, wall_bottom_right, wall_top_right, room.width - self.location[0],
                                            angle_bin_centers, angle_bin_edges)
        self.update_state_with_wall_or_door(state, wall_top_left, wall_bottom_left, self.location[0],
                                            180 - angle_bin_centers, angle_bin_edges)
        self.update_state_with_wall_or_door(state, wall_top_right, wall_top_left, room.height - self.location[1],
                                            90 - angle_bin_centers, angle_bin_edges)
        self.update_state_with_wall_or_door(state, door_right, door_left, room.height - self.location[1],
                                            90 - angle_bin_centers, angle_bin_edges, is_door=True)

        # other robots
        for other_robot in room.robots:
            if other_robot is not self:
                dist, dir = room.get_dist_and_azimuth_between_robot_centers(self, other_robot)
                angular_diameter = np.rad2deg(2 * np.arcsin(other_robot.radius / dist))
                angle_limit1 = fix_angle(dir - angular_diameter / 2, offset)
                angle_limit2 = fix_angle(angle_limit1 + angular_diameter, offset)
                if angle_limit1 <= angle_limit2:
                    inds = (angle_bin_edges[1:] >= angle_limit1) & (angle_bin_edges[:-1] <= angle_limit2)
                else:
                    inds = (angle_bin_edges[1:] >= angle_limit1) | (angle_bin_edges[:-1] <= angle_limit2)
                if np.any(inds):
                    inds = np.nonzero(inds)[0]
                    a = np.deg2rad(angle_bin_centers[inds] - fix_angle(dir, offset))
                    inside_sqrt = other_robot.radius ** 2 - (dist ** 2) * np.sin(a) ** 2
                    inds2 = inside_sqrt >= 0
                    robot_dists = dist * np.cos(a[inds2]) - np.sqrt(inside_sqrt[inds2])
                    other_dists = state[inds, self.curr_frame_ind, self.dist_ind][inds2]
                    closer = robot_dists < other_dists
                    if np.any(closer):
                        state[inds[inds2][closer], self.curr_frame_ind, self.dist_ind] = robot_dists[closer]
                        state[inds[inds2][closer], self.curr_frame_ind, self.wall_ind] = 0.0
                        state[inds[inds2][closer], self.curr_frame_ind, self.door_ind] = 0.0
                        state[inds[inds2][closer], self.curr_frame_ind, self.robot_ind] = 1.0

        if np.isnan(state[0, self.prev_frame_ind, 0]):
            state[:, self.prev_frame_ind, :] = state[:, self.curr_frame_ind, :]
        assert np.all(np.isfinite(state))

        return state

    def give_reward(self, reward=None, is_final_state=None):
        if reward is not None:
            self.prev_reward = reward
        if is_final_state is not None:
            self.prev_is_final_state = is_final_state

    def update_prev_state_action(self,state,action):
        self.prev_action = action
        self.prev_state[:] = state

    def update_state_with_wall_or_door(self, state, edge1, edge2, dist, angs, angle_bin_edges, is_door=False):
        if edge1 <= edge2:
            inds = (angle_bin_edges[1:] >= edge1) & (angle_bin_edges[:-1] <= edge2)
        else:
            inds = (angle_bin_edges[1:] >= edge1) | (angle_bin_edges[:-1] <= edge2)
        if np.any(inds):
            dists = np.abs(dist / np.cos(np.deg2rad(angs[inds])))
            other_wall_dists = state[inds, self.curr_frame_ind, self.dist_ind]
            state[inds, self.curr_frame_ind, self.dist_ind] = np.minimum(dists, other_wall_dists)

            if is_door:
                state[inds, self.curr_frame_ind, self.dist_ind] = dists
                state[inds, self.curr_frame_ind, self.wall_ind] = 0.0
                state[inds, self.curr_frame_ind, self.door_ind] = 1.0

    def get_next_location(self, turn_action, speed_action):
        next_speed = self.speed + (speed_action - 1) * (self.max_speed / (self.num_speeds - 1))
        next_speed = np.clip(next_speed, 0, self.max_speed)
        next_direction = fix_angle(self.direction + (turn_action - 1) * self.turn_speed, 180)
        next_location = self.location + next_speed * np.asarray(
            [np.cos(np.deg2rad(next_direction)), np.sin(np.deg2rad(next_direction))])

        return next_location, next_speed, next_direction

    def choose_action(self, state, inference=True):
        return self.policy.choose_action(state, inference)

    def apply_action(self, speed_action, turn_action, room: 'Room'):
        next_location, next_speed, next_direction = self.get_next_location(turn_action, speed_action)
        collision = room.collision_with_wall(*next_location, self.radius)
        if not collision:
            for other_robot in room.robots:
                if other_robot is not self:
                    collision = room.collision_between_robots(*next_location, self.radius, *other_robot.location,
                                                              other_robot.radius)
                    if collision:
                        break
        if collision:
            next_speed = 0
            next_location = self.location
            self.give_reward(reward=Rewards.collision)
            # next_direction = self.direction+180
        self.update_location(next_location, next_direction, next_speed)

    def choose_and_perform_action(self, room: 'Room', inference=False):
        speed_action, turn_action = self.choose_action(self.state, inference)
        self.apply_action(speed_action, turn_action, room)
        self.update_steps_in_episode()
        return encode_action(speed_action, turn_action)


    def assign_policy(self, policy):
        self.policy = policy

    def update_steps_in_episode(self,reset=False):
        if reset:
            self.steps_in_episode = 0
        else:
            self.steps_in_episode +=1
        return self.steps_in_episode
