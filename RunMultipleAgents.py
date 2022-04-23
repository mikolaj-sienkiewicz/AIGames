import time 
import pygame

import os,sys
sys.path.append(os.path.join(sys.path[0], "./agents"))


from environment.GridEnv import GridWorld
from agents.CmaesAgent import CmaesAgent
from agents.HeuristicAgent import HeuristicAgent
from agents.ReinforcedAgent import ReinforcedAgent
from render_controller import RenderController


class RunMultipleAgents(object):
    metadata = {'render.modes': ['human']}
    POINTS_TO_COLLECT=10

    def __init__(self, m, heuristicAgents=0, reinforcedAgents=0, cmaesAgents=0):
        super().__init__()
        
        self.game_already_initialized=False
        self.agent_num=heuristicAgents+reinforcedAgents+cmaesAgents
        self.agents=[]
        self.obs_array=[]

        self.grid_size=m
        self.render_controller=RenderController(self.grid_size,self.grid_size)
        self.main_env=GridWorld(self.grid_size)
        
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
        
        for e,agent in enumerate(self.agents):
            print(self.main_env.other_agents,tuple(agent.env.agentPosition[:2]))
            self.main_env.other_agents.remove(tuple(agent.env.agentPosition[:2]))
            action , _states = agent.predict(self.obs_array[e])
            next_observation, reward, done, _ = agent.env.step(action)
            self.obs_array[e]=next_observation
            self.main_env.other_agents.append(tuple(agent.env.agentPosition[:2]))
            agent.env.update_other_agents(self.main_env.other_agents)

        self.render()
        

    def render(self):
        self.render_controller.draw_map(self.main_env.img)
        self.render_controller.draw_multiple_agents(self.agents)
        self.render_controller.draw_bases_multiple_agents(self.main_env.base_position,self.main_env.base_colors)
        self.render_controller.draw_food_multiple_agents(self.main_env.food_list,self.main_env.base_colors)
        self.render_controller.draw_grid() 
        self.render_controller.draw_text_multiple_agents(self.agents)
        pygame.display.flip()
        time.sleep(0.1)

if __name__=='__main__':
    simulator=RunMultipleAgents(12,cmaesAgents=1,reinforcedAgents=1,heuristicAgents=1)
    for i in range(200):
        simulator.step()