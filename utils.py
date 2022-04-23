import pygame
import matplotlib.pylab as plt
from IPython import display

def plot_rewards(rewards):
    plt.figure(figsize=(14,6))
    plt.plot(rewards)
    display.display(plt.gcf())
    display.clear_output(wait=True)

    
def run_agent_in_env(env, agent, episodes, learning=False, plot=False, render=False, plot_interval=1000):
    rewards = []
    for episode in range(episodes):
        observation = env.reset()
        total_reward = 0
        done = False
        while not done :
            if render:
                env.render()
            action , _ = agent.predict(observation)
            
            next_observation, reward, done, _ = env.step(action)
            total_reward += reward
            observation = next_observation
        rewards.append(total_reward)
        
        if plot and episode % plot_interval == 0:
            plot_rewards(rewards)
    if render:
        pygame.quit()
    return rewards 