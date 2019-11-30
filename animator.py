import matplotlib.pyplot as plt
import numpy as np

from environment import Room


class Animator:
    """
    use this class for viewing the room and the filters of the Q-network
    """

    def __init__(self, room: Room, robots_to_debug=None):
        self.robots_to_debug = robots_to_debug if robots_to_debug is not None else []
        self.fig = plt.figure()
        self.ax1 = plt.axes()
        plt.plot([0, room.width], [0, 0], color='black')
        plt.plot([room.width, room.width], [0, room.height], color='black')
        plt.plot([0, 0], [0, room.height], color='black')
        plt.plot([0, room.width / 2 - room.door_width / 2], [room.height, room.height], color='black')
        plt.plot([room.width, room.width / 2 + room.door_width / 2], [room.height, room.height], color='black')
        self.circles = [plt.Circle((0, 0), robot.radius, color='green' if robot in self.robots_to_debug else 'blue') for
                        robot in room.robots]
        self.dirs = [plt.plot([0, 0], [0, 0], color='yellow', linewidth=2)[0] for _ in room.robots]
        self.views = [[plt.plot([0, 0], [0, 0], color='silver', linestyle='-', linewidth=0.4)[0] for _ in
                       range(robot.num_fov_pixels)] for robot in room.robots if robot in self.robots_to_debug]
        for art in self.circles:
            self.ax1.add_artist(art)
        plt.axis('equal')
        plt.axis((-0.5, room.width + 0.5, -0.5, room.height + 0.5))

        self.room = room

    def Update(self):
        self.ax1.set_title(self.room.step)
        for j, robot in enumerate(self.room.robots):
            self.circles[j].center = robot.location
            length = robot.radius
            self.dirs[j].set_xdata([robot.location[0], robot.location[0] + length * np.cos(
                np.deg2rad(robot.direction))])
            self.dirs[j].set_ydata([robot.location[1], robot.location[1] + length * np.sin(
                np.deg2rad(robot.direction))])
            if robot in self.robots_to_debug:
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
        plt.pause(0.01)  # 0.01
