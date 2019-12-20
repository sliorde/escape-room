from collections import namedtuple
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from environment import Room

mpl.rcParams['toolbar'] = 'None'

FakeRobot = namedtuple('FakeRobot',['location','direction','radius','num_fov_pixels'])

class Animator:
    """
    use this class for viewing the room and the filters of the Q-network
    """

    def __init__(self, room:Optional[Room]=None, robots_to_debug=None, width=None, height=None,door_width=None,radi=None,num_fov_pixels=None,pause_time=None):
        args = [width, height, door_width,radi,num_fov_pixels]
        assert (room is not None and all([x is None for x in args]) ) or (room is None and robots_to_debug is None and all(
            [x is not None for x in args]))

        use_room = height is None
        if use_room:
            height = room.height
            width = room.width
            door_width = room.door_width
            robots = room.robots
        else:
            robots = [FakeRobot(location=None,direction=None,radius=r,num_fov_pixels=n) for r,n in zip(radi,num_fov_pixels)]


        self.robots_to_debug = robots_to_debug if robots_to_debug is not None else []
        self.fig = plt.figure(figsize=(2,2*height/width))
        self.ax1 = plt.axes()
        self.ax1.plot([0, width], [0, 0], color='black')
        self.ax1.plot([width, width], [0, height], color='black')
        self.ax1.plot([0, 0], [0, height], color='black')
        self.ax1.plot([0, width / 2 - door_width / 2], [height, height], color='black')
        self.ax1.plot([width, width / 2 + door_width / 2], [height, height], color='black')
        self.circles = [plt.Circle((0, 0), robot.radius, color='green' if robot in self.robots_to_debug else 'blue',axes=self.ax1) for
                        robot in robots]
        self.dirs = [self.ax1.plot([0, 0], [0, 0], color='yellow', linewidth=2)[0] for _ in robots]
        self.views = [[self.ax1.plot([0, 0], [0, 0], color='silver', linestyle='-', linewidth=0.4)[0] for _ in
                       range(robot.num_fov_pixels)] for robot in robots if robot in self.robots_to_debug]
        for art in self.circles:
            self.ax1.add_artist(art)
        self.ax1.axis('equal')
        self.ax1.axis((-0.1, width + 0.1, -0.1, height + 0.1))
        self.ax1.set_frame_on(False)
        self.ax1.set_title(' ')
        self.fig.tight_layout()
        self.room = room

        self.pause_time = pause_time

    def get_figure(self):
        return self.fig

    def Update(self,step=None,locations=None,directions=None):
        args = [step,locations,directions]
        assert (all([x is None for x in args]) and self.room is not None) or all([x is not None for x in args])

        use_room = step is None

        self.ax1.set_title('time = {:d}'.format(self.room.step if use_room else step))

        if not use_room:
            itr = [FakeRobot(l,d,None,None) for l,d in zip(locations,directions)]
        else:
            itr = self.room.robots

        for j, robot in enumerate(itr):
            self.circles[j].center = robot.location
            length = self.circles[j].radius
            self.dirs[j].set_xdata([robot.location[0], robot.location[0] + length * np.cos(
                np.deg2rad(robot.direction))])
            self.dirs[j].set_ydata([robot.location[1], robot.location[1] + length * np.sin(
                np.deg2rad(robot.direction))])
            if use_room and (robot in self.robots_to_debug):
                limit_right = robot.direction - robot.fov_size / 2
                limit_left = limit_right + robot.fov_size
                angle_bin_edges = np.deg2rad(
                    np.linspace(start=limit_right, stop=limit_left, num=robot.num_fov_pixels + 1))
                angle_bin_centers = (angle_bin_edges[:-1] + angle_bin_edges[1:]) / 2

                state = robot.get_observed_state(self.room, without_update_self=True)
                for i, (a, obj) in enumerate(zip(angle_bin_centers, state[:, robot.curr_frame_ind])):
                    r = obj[robot.dist_ind]
                    if np.isnan(r):
                        r = 1000
                    self.views[j][i].set_xdata([robot.location[0], robot.location[0] + r * np.cos(a)])
                    self.views[j][i].set_ydata([robot.location[1], robot.location[1] + r * np.sin(a)])
                    object_type = obj[[robot.wall_ind, robot.robot_ind, robot.door_ind]]
                    if np.max(object_type) == 0.0:
                        clr = 'red'
                    else:
                        object_type = np.argmax(object_type)
                        clr = ['silver', 'gold', 'lime'][object_type]
                    self.views[j][i].set_color(clr)
        plt.sca(self.ax1)

        # plt.waitforbuttonpress()
        if self.pause_time is not None:
            plt.pause(self.pause_time)  # 0.01
