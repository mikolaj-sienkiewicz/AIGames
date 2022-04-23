from cma_model import Model
import numpy as np

class CmaesAgent:
    def __init__(self, env):
        self.env=env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.weights = np.load('./CMA/20220422-235051/weights.npy')
        self.model =  Model.from_weights(self.observation_space, self.action_space.n, self.weights)

    def process_transition(self, observation, action, reward, next_observation, done):
        raise NotImplementedError()
        
    def get_action(self, observation, learning):
        raise NotImplementedError()

    def predict(self, observation):
        action = np.argmax(self.model(observation))
        state = None
        return action, state 