from cma_model import Model
import numpy as np

import sys,os 
sys.path.append(os.path.join(sys.path[0], ".."))

from environment.GridEnv import GridWorld
from utils import run_agent_in_env


class CmaesAgent:
    def __init__(self, env, model_path="../saved_models/evolution/weights.npy"):
        self.env=env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.weights = np.load(model_path)
        self.model =  Model.from_weights(self.observation_space, self.action_space.n, self.weights)

    def predict(self, observation):
        action = np.argmax(self.model(observation))
        state = None
        return action, state 

if __name__=='__main__':
    GRID_SIDE_LENGTH=12
    env = GridWorld(GRID_SIDE_LENGTH)
    env.reset()
    agent = CmaesAgent(env)
    rewards = run_agent_in_env(env, agent, 5, learning=False, plot=False, render=True, plot_interval=20)