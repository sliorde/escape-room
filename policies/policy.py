from utils import ReplayMemory


class Policy:

    def __init__(self, replay_buffer_to: ReplayMemory = None, replay_buffer_from: ReplayMemory = None):
        self.replay_buffer_to = replay_buffer_to
        if replay_buffer_from is None:
            self.replay_buffer_from = replay_buffer_to
        else:
            self.replay_buffer_from = replay_buffer_from

    def choose_action_inference(self, state):
        raise NotImplementedError

    def choose_action_training(self, state):
        raise NotImplementedError

    def choose_action(self, state, inference=True):
        if inference:
            return self.choose_action_inference(state)
        else:
            return self.choose_action_training(state)

    def add_to_replay_buffer(self, state, action, reward, robot):
        if self.replay_buffer_to is not None:
            self.replay_buffer_to.add_to_replay_buffer(state, action, reward, robot)

    def optimization_step(self):
        pass
