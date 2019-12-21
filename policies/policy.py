from utils import PrioritizedReplayBuffer

class Policy:

    def __init__(self, replay_buffer: PrioritizedReplayBuffer = None):
        self.replay_buffer = replay_buffer

    def choose_action_inference(self, state):
        raise NotImplementedError

    def choose_action_training(self, state):
        raise NotImplementedError

    def choose_action(self, state, inference=True):
        if inference:
            return self.choose_action_inference(state)
        else:
            return self.choose_action_training(state)

    def add_to_replay_buffer(self, prev_state, action, reward, state, is_final_state=False):
        if self.replay_buffer is not None:
            self.replay_buffer.add(prev_state, action, reward, state, float(is_final_state))

    def optimization_step(self,output_dir):
        pass
