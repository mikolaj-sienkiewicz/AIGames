from sre_constants import SRE_FLAG_DEBUG
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces 
import pygame
from render_controller import draw_map, draw_grid, draw_text, draw_agent, draw_bases
import time 
import copy
from render_controller import initialize_game

class GridWorld(object):
    metadata = {'render.modes': ['human']}
    POINTS_TO_COLLECT=10

    def __init__(self, m):
        super().__init__()


        self.grid_size = m
        self.game_already_initialized=False
        self.points_collected=0


        self.num_bases=2

        #potential hazard - same location for two bases
        self.base_position=[tuple(np.random.randint(0,self.grid_size,size=2)) for _ in range(self.num_bases)]
        self.base_colors={i:list(np.random.randint(100,250,3)) for i in range(self.num_bases)}


        self.time_cap=500
        self.current_time=0

        self.new_food_prob=0.03
        self.max_food=3
        self.food_counter=0
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

        print(f"FOLLOWING BASES INITIALIZED: COLORS, {self.base_colors}  POSITIONS, {self.base_position}")

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

    def collision_with_base(self):
        for e,base in enumerate(self.base_position):
            if self.agentPosition[:2]==base:
                self.agentPosition = (self.agentPosition[0],self.agentPosition[1],e)
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
        if self.collision_with_boundaries() or self.collision_with_base() or self.collision_with_other_food():
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

        # Render while training
        # if not self.game_already_initialized:
        #     self.surface=initialize_game()
        #     self.game_already_initialized = True 
        # self.render(self.surface)


        observation = self.patch_observation()

        return observation, reward, \
                done, {}

    def reset(self):

        self.points_collected=0
        
        self.food_counter=0
        self.img = np.zeros((self.grid_size,self.grid_size,3),dtype='uint8')

        self.pointsToCollect = self.POINTS_TO_COLLECT
        self.current_time=0
        
        self.base_position=[tuple(np.random.randint(0,self.grid_size,size=2)) for _ in range(self.num_bases)]

        self.agentPosition = tuple([*np.random.randint(0,self.grid_size,size=2),-1])

        self.food_list=[]
        for _ in range(self.max_food):
            x = tuple(np.random.randint(0,self.grid_size,size=2))
            while x == self.agentPosition or x in self.base_position or x in [tuple(food[:2]) for food in self.food_list]:
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

    def render(self, surface):
        draw_map(surface, self.img,self.grid_size,self.grid_size)
        draw_agent(surface,self.agentPosition, self.grid_size,self.grid_size)
        draw_bases(surface,self.base_position, self.grid_size,self.grid_size)
        draw_grid(surface,self.grid_size,self.grid_size) 
        draw_text(surface,f"POINTS COLLECTED: {self.points_collected}")
        pygame.display.set_caption("Color cakes")
        pygame.display.flip()
        time.sleep(0.1)


    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]