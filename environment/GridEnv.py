import numpy as np
from gym import spaces 
import pygame
import os,sys

sys.path.append(os.path.join(sys.path[0], ".."))

from render_controller import RenderController
import time 
import copy
from typing import List, Tuple
import constants

class GridWorld(object):
    metadata = {'render.modes': ['human']}
    POINTS_TO_COLLECT=1

    def __init__(self, m, max_time=50, num_bases=2, num_food=3):
        super().__init__()
        self.other_agents=[]
        self.grid_size = m
        self.game_already_initialized=False
        self.points_collected=0
        self.render_controller=None

        self.num_bases=num_bases

        #potential hazard - same location for two bases
        self.base_position=[tuple(np.random.randint(0,self.grid_size,size=2)) for _ in range(self.num_bases)]
        if self.num_bases==2:
            self.base_colors={0:constants.DARKGREEN,1:constants.DARKORANGE}
        else:
            self.base_colors={i:list(np.random.randint(100,250,3)) for i in range(self.num_bases)}


        self.time_cap=max_time
        self.current_time=0

        self.max_food=num_food
        self.food_type_prob={i:tuple([i/self.num_bases,(i+1)/self.num_bases]) for i in range(self.num_bases)}

        self.reward_normalizer=self.manhattan(tuple([self.grid_size,self.grid_size]),tuple([0,0]))*2*self.max_food

        
        self.img = np.zeros((self.grid_size,self.grid_size,3),dtype='uint8')
        self.observation_space = spaces.Box(low=-16, high=16,
                                            shape=(3+self.num_bases*2+self.max_food*3,), dtype=np.float32)
        
        self.action_space=spaces.Discrete(4)
        self.actionDict = {0: -self.grid_size, 1: self.grid_size,
                            2: -1, 3: 1}
        self.possibleActions = [0, 1,2,3]


        self.agentPosition = tuple([*np.random.randint(0,self.grid_size,size=2),-1])
        self.food_list=[]
        for i in range(self.max_food):
            x = tuple(np.random.randint(0,self.grid_size,size=2))
        
            while x == self.agentPosition or x in self.base_position or x in [tuple(food[:2]) for food in self.food_list]:
                x=tuple(np.random.randint(0,self.grid_size,size=2))
            food_type=np.random.randint(0,self.num_bases)
            self.food_list.append(tuple([*x,food_type]))

        self.pointsToCollect=self.POINTS_TO_COLLECT

    def update_other_agents(self, other_agents: List[Tuple[int,int]]):
        self.other_agents=other_agents

    def manhattan(self, a, b):
        return sum(abs(val1-val2) for val1, val2 in zip(a,b))

    def isTerminalState(self):
        if self.pointsToCollect==0:
            return True
        else:
            return False

    def collision_with_boundaries(self):
        if self.agentPosition[0]>=self.grid_size or self.agentPosition[0]<0 or self.agentPosition[1]>=self.grid_size or self.agentPosition[1]<0 :
            return 1
        else:
            return 0
    
    def collision_with_my_food(self):
        for e,food in enumerate(self.food_list):
            if self.agentPosition==tuple(food):
                x = tuple(np.random.randint(0,self.grid_size,size=2))
                while x == self.agentPosition or x in self.base_position or x in [tuple(food[:2]) for food in self.food_list]:
                    x=tuple(np.random.randint(0,self.grid_size,size=2))
                food_type=np.random.randint(0,self.num_bases)
                self.food_list[e]=tuple([*x,food_type])
                self.img[x] = self.base_colors[food_type]
                self.pointsToCollect-=1
                return 1
                
        return 0

    def collision_with_other_food(self):
        for food in self.food_list:
            if tuple(self.agentPosition[:2])==tuple(food[:2]) and self.agentPosition[2]!=food[2]:
                return 1
        return 0

    def collision_with_other_agent(self):
        for agent in self.other_agents:
            if self.agentPosition[:2]==agent:
                return 1
        return 0

    def collision_with_base(self):
        for e,base in enumerate(self.base_position):
            if self.agentPosition[:2]==base:
                self.agentPosition = (self.agentPosition[0],self.agentPosition[1],e)
                return 1
        return 0
    
    def collision_with_same_color_base(self):
        for e,base in enumerate(self.base_position):
            if self.agentPosition[:2]==base and e==self.agentPosition[2]:
                return 1
        return 0

    def change_agent_position(self,button_direction):
        # Change the agent position based on the button direction
        if button_direction == 1:
            self.agentPosition = (self.agentPosition[0]+1,self.agentPosition[1],self.agentPosition[2])
        elif button_direction == 0:
            self.agentPosition = (self.agentPosition[0]-1,self.agentPosition[1],self.agentPosition[2])
        elif button_direction == 2:
            self.agentPosition = (self.agentPosition[0],self.agentPosition[1]+1,self.agentPosition[2])
        elif button_direction == 3:
            self.agentPosition = (self.agentPosition[0],self.agentPosition[1]-1,self.agentPosition[2])


    def step(self, action):
                
        done=False
        old_index=copy.deepcopy(self.agentPosition)
        
        self.change_agent_position(action)

        # don't change coords if wall was hit but preserve color change if base was hit
        if self.collision_with_boundaries() or self.collision_with_base() or self.collision_with_other_food() or self.collision_with_other_agent():
            self.agentPosition=tuple([*old_index[:2],self.agentPosition[2]])
        
        reward = 0

        if self.collision_with_my_food():
            self.points_collected+=1
            reward+=self.time_cap/(self.POINTS_TO_COLLECT/2) # (2*self.grid_size)*self.grid_size*self.max_food*2  

        reward-=1

        self.img[tuple(old_index[:2])]=(0,0,0)
        if self.agentPosition[2]!=-1:
            self.img[tuple(self.agentPosition[:2])]=self.base_colors[self.agentPosition[2]]
        else:
            self.img[tuple(self.agentPosition[:2])]=(255,0,0)


        self.current_time+=1
        
        #Add strict time horizon
        if self.current_time>=self.time_cap or self.isTerminalState():
            done=True

        observation = self.patch_observation()

        return observation, reward, \
                done, {}

    def reset(self):

        self.points_collected=0
        
        self.img = np.zeros((self.grid_size,self.grid_size,3),dtype='uint8')

        self.pointsToCollect = self.POINTS_TO_COLLECT
        self.current_time=0
        
        self.base_position=[tuple(np.random.randint(0,self.grid_size,size=2)) for _ in range(self.num_bases)]

        self.agentPosition = tuple([*np.random.randint(0,self.grid_size,size=2),-1])

        self.food_list=[]
        for _ in range(self.max_food):
            x = tuple(np.random.randint(0,self.grid_size,size=2))
            while x == self.agentPosition[:2] or x in self.base_position or x in [tuple(food[:2]) for food in self.food_list]:
                x=tuple(np.random.randint(0,self.grid_size,size=2))
            food_type=np.random.randint(0,self.num_bases)
            self.food_list.append(tuple([*x,food_type]))
            self.img[x] = self.base_colors[food_type]
        
        if self.agentPosition[2]!=-1:
            self.img[tuple(self.agentPosition[:2])] = self.base_colors[self.agentPosition[2]]
        else:
            self.img[tuple(self.agentPosition[:2])]=(255,0,0)

        #DRAW THE BASES
        for e,base in enumerate(self.base_position):
            self.img[base]=[color_channel-50 for color_channel in self.base_colors[e]]

        observation = self.patch_observation() 

        return observation

    def patch_observation(self):
        obs=[]
        obs+=(self.agentPosition)
        for base in self.base_position:
            obs+=([self.agentPosition[0]-base[0],self.agentPosition[1]-base[1]])
        for food in self.food_list:   
            obs+=([self.agentPosition[0]-food[0],self.agentPosition[1]-food[1],food[2]]) 
        return np.array(obs)

    def render(self):
        if not self.game_already_initialized:
            self.render_controller=RenderController(self.grid_size,self.grid_size)
            self.game_already_initialized=True
        self.render_controller.draw_map(self.img)
        self.render_controller.draw_agent(self.agentPosition)
        self.render_controller.draw_bases(self.base_position, self.base_colors)
        self.render_controller.draw_grid() 
        self.render_controller.draw_text(f"POINTS COLLECTED: {self.points_collected}")
        pygame.display.flip()
        time.sleep(0.1)

    def sync(self, env):
        self.num_bases=env.num_bases
        self.base_position=env.base_position
        self.base_colors=env.base_colors
        self.time_cap=env.time_cap
        self.current_time=env.current_time
        self.max_food=env.max_food
        self.other_agents=env.other_agents
        self.points_collected=env.points_collected
        self.reward_normalizer=env.reward_normalizer
        self.food_list=env.food_list
        self.POINTS_TO_COLLECT=env.POINTS_TO_COLLECT
        self.pointsToCollect=self.POINTS_TO_COLLECT

        while self.agentPosition[:2] in self.other_agents or self.agentPosition[:2] in self.base_position or self.agentPosition[:2] in [tuple(food[:2]) for food in self.food_list]:
            self.agentPosition=(*np.random.randint(0,self.grid_size,size=2),self.agentPosition[2])

        return self.patch_observation() #tuple(self.agentPosition[:2])

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]