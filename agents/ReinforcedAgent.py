from stable_baselines3 import DQN, PPO, A2C

import sys,os 
sys.path.append(os.path.join(sys.path[0], ".."))

from environment.GridEnv import GridWorld
from utils import run_agent_in_env

class ReinforcedAgent:
    def __init__(self, env, model_path="/home/mikolaj/Desktop/Sem10/AI_in_games/AIGames/saved_models/reinforcement/12gr_3f_2b_2_9200000"):
        self.env=env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.model = A2C.load(model_path)

    def predict(self, observation):
        return self.model.predict(observation, deterministic=False)


if __name__=='__main__':
    GRID_SIDE_LENGTH=12
    env = GridWorld(GRID_SIDE_LENGTH)
    env.reset()
    agent = ReinforcedAgent(env)
    rewards = run_agent_in_env(env, agent, 5, learning=False, plot=False, render=True, plot_interval=20)