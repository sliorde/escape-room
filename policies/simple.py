import numpy as np

from policies import Policy
from agent import Robot

class StupidPolicy(Policy):
    def __init__(self,p_speed=None,p_turn=None):
        self.p_speed = p_speed if p_speed is not None else [0.2,0.0,0.8]
        self.p_turn = p_turn if p_turn is not None else [0.3,0.5,0.2]
    def choose_action_inference(self,state):
        speed_action = np.random.choice([0,1,2],p=self.p_speed)
        turn_action = np.random.choice([0,1,2],p=self.p_turn)
        return speed_action, turn_action

    def choose_action_training(self,state):
        return self.choose_action_inference(state)

class HeuristicPolicy(Policy):
    def choose_action_inference(self,state):

        door_inds = np.flatnonzero(state[:,Robot.curr_frame_ind,Robot.door_ind]>0)
        if len(door_inds)==0:
            speed_action = 0
            turn_action = 0
        else:
            if np.any((state[np.arange(-2,10)+Robot.num_fov_pixels//2, Robot.curr_frame_ind, Robot.wall_ind]>0) & (state[np.arange(-2,10)+Robot.num_fov_pixels//2, Robot.curr_frame_ind, Robot.dist_ind]<6)):
                speed_action = 0
                turn_action = 0
            else:
                speed_action = 2
                offset = np.mean(door_inds)-Robot.num_fov_pixels/2
                turn_action = 0 if (offset < -3) else 2 if (offset > 3) else 1
        return speed_action, turn_action

    def choose_action_training(self, state):
        return self.choose_action_inference()

