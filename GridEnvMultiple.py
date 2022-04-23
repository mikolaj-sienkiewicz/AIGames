from sre_constants import SRE_FLAG_DEBUG
from CmaesAgent import CmaesAgent
from HeuristicAgent import HeuristicAgent
from ReinforcedAgent import ReinforcedAgent
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces 
import pygame
from render_controller import draw_map, draw_grid, draw_text_multiple_agents, draw_bases_multiple_agents, draw_multiple_agents, draw_food_multiple_agents
import time 
import copy
from render_controller import initialize_game
from GridEnv import GridWorld

class GridWorldMultiple(object):
    metadata = {'render.modes': ['human']}
    POINTS_TO_COLLECT=10

    def __init__(self, m, heuristicAgents=0, reinforcedAgents=0, cmaesAgents=0):
        super().__init__()
        
        self.game_already_initialized=False
        self.agent_num=heuristicAgents+reinforcedAgents+cmaesAgents
        self.agents=[]
        self.grid_size=m
        #self.img = np.zeros((self.grid_size,self.grid_size,3),dtype='uint8')        
        self.main_env=GridWorld(self.grid_size)
        self.obs_array=[]
        for agent in range(reinforcedAgents):
            env=GridWorld(self.grid_size)
            self.agents.append(ReinforcedAgent(env))

        for agent in range(heuristicAgents):
            env=GridWorld(self.grid_size)
            self.agents.append(HeuristicAgent(env))
        
        for agent in range(cmaesAgents):
            env=GridWorld(self.grid_size)
            self.agents.append(CmaesAgent(env))

        for agent in self.agents:
            self.obs_array.append(agent.env.sync(self.main_env))
            self.main_env.other_agents.append(tuple(agent.env.agentPosition[:2]))

        for agent in self.agents:
            agent.env.update_other_agents(self.main_env.other_agents)

    def step(self):
        #image => main_env.img / step kazdego z kolejnych agentow nadpisuje main_env_img i agentPositions        
        # obs_array=[]
        # for agent in self.agents:
        #     obs_array.append(agent.env.sync(self.main_env))
        
        for e,agent in enumerate(self.agents):
            print(self.main_env.other_agents,tuple(agent.env.agentPosition[:2]))
            self.main_env.other_agents.remove(tuple(agent.env.agentPosition[:2]))
            action , _states = agent.predict(self.obs_array[e])
            next_observation, reward, done, _ = agent.env.step(action)
            self.obs_array[e]=next_observation
            self.main_env.other_agents.append(tuple(agent.env.agentPosition[:2]))
            agent.env.update_other_agents(self.main_env.other_agents)
            #agent.env.sync(self.main_env)

        if not self.game_already_initialized:
            self.surface=initialize_game()
            self.game_already_initialized = True 
        self.render(self.surface)
        

    def render(self, surface):
        draw_map(surface, self.main_env.img,self.main_env.grid_size,self.main_env.grid_size)
        draw_multiple_agents(surface,self.agents, self.main_env.grid_size,self.main_env.grid_size)
        draw_bases_multiple_agents(surface,self.main_env.base_position,self.main_env.base_colors, self.main_env.grid_size,self.main_env.grid_size)
        draw_food_multiple_agents(surface,self.main_env.food_list,self.main_env.base_colors,self.main_env.grid_size,self.main_env.grid_size)
        draw_grid(surface,self.main_env.grid_size,self.main_env.grid_size) 
        draw_text_multiple_agents(surface,self.main_env, self.agents)
        pygame.display.set_caption("Color cakes")
        pygame.display.flip()
        time.sleep(0.1)

if __name__=='__main__':
    simulator=GridWorldMultiple(12,cmaesAgents=1,reinforcedAgents=2,heuristicAgents=2)
    for i in range(200):
        simulator.step()