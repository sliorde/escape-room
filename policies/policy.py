class Policy:
    def choose_action_inference(self,state):
        raise NotImplementedError

    def choose_action_training(self,state):
        raise NotImplementedError

    def choose_action(self,state,inference=True):
        if inference:
            return self.choose_action_inference(state)
        else:
            return self.choose_action_training(state)
