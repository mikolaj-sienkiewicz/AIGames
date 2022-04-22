from stable_baselines3 import DQN, PPO, A2C

class ReinforcedAgent:
    def __init__(self, env):
        self.env=env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.model = A2C.load(f"models/1650180249/2_9200000.zip")

    def process_transition(self, observation, action, reward, next_observation, done):
        raise NotImplementedError()
        
    def get_action(self, observation, learning):
        raise NotImplementedError()

    def predict(self, observation):
        return self.model.predict(observation, deterministic=False)