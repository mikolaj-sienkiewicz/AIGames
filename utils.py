import numpy as np
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gym import spaces
import pygame

import matplotlib.pylab as plt
from IPython import display

from GridEnv import GridWorld
from Agent import NeuralQLearningAgent
from render_controller import initialize_game

from stable_baselines3 import DQN, PPO, A2C

def plot_rewards(rewards):
    plt.figure(figsize=(14,6))
    plt.plot(rewards)
    display.display(plt.gcf())
    display.clear_output(wait=True)

    
def run_agent_in_env(env, agent, episodes, learning=False, plot=False, render=False, plot_interval=1000):
    rewards = []
    for episode in range(episodes):
        game_already_initialized = False
        observation = env.reset()
        total_reward = 0
        done = False
        while not done :
            if render:
                if not game_already_initialized:
                    surface=initialize_game()
                    game_already_initialized = True  
                env.render(surface)
            action , _states = agent.predict(observation)#agent.get_action(observation, learning)
            
            # Wykonajmy akcje
            next_observation, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Jeśli się uczymy, przekażmy przejście do agenta
            # if learning:
            #     agent.process_transition(observation, action, reward, next_observation, done)
            
            observation = next_observation
        if render:
            pygame.quit()
        rewards.append(total_reward)
        
        # Wyświetl na wykresie nagrody otrzymane po kolei w epizodach
        if plot and episode % plot_interval == 0:
            plot_rewards(rewards)
    return rewards  


def run_agents_in_env(env, agents, episodes, learning=False, plot=False, render=False, plot_interval=1000):
    rewards = []
    for episode in range(episodes):
        game_already_initialized = False
        observation = env.reset()
        total_reward = 0
        done = False
        while not done :
            if render:
                if not game_already_initialized:
                    surface=initialize_game()
                    game_already_initialized = True  
                env.render(surface)
            for agent in agents:
                action , _states = agent.predict(observation)#agent.get_action(observation, learning)
            
                # Wykonajmy akcje
                next_observation, reward, done, _ = env.step(action)
                total_reward += reward

                # Jeśli się uczymy, przekażmy przejście do agenta
                if learning:
                    agent.process_transition(observation, action, reward, next_observation, done)

                observation = next_observation
        if render:
            pygame.quit()
        rewards.append(total_reward)
        
        # Wyświetl na wykresie nagrody otrzymane po kolei w epizodach
        if plot and episode % plot_interval == 0:
            plot_rewards(rewards)
    return rewards  