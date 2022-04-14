from sre_constants import SRE_FLAG_DEBUG
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces 
import pygame
from render_controller import draw_map, draw_grid
import time 
import copy
from render_controller import initialize_game

class GridWorld(object):
    metadata = {'render.modes': ['human']}
    POINTS_TO_COLLECT=3

    def __init__(self, m):
        super().__init__()


        self.grid_size = m
        self.game_already_initialized=False
        self.points_collected=0


        self.num_bases=2
        #potential hazard - same location for two bases
        self.base_position=[tuple(np.random.randint(0,self.grid_size,size=2)) for _ in range(self.num_bases)]
        self.base_colors={i:list(np.random.randint(100,250,3)) for i in range(self.num_bases)}


        self.time_cap=5000
        self.current_time=0

        self.new_food_prob=0.05
        self.max_food=10
        self.food_counter=0
        self.food_type_prob={i:tuple([i/self.num_bases,(i+1)/self.num_bases]) for i in range(self.num_bases)}


        
        self.img = np.zeros((self.grid_size,self.grid_size,3),dtype='uint8')
        self.observation_space = spaces.Box(low=-self.grid_size, high=self.grid_size,
                                            shape=(3+self.num_bases*2+self.max_food*3,), dtype=np.float32)
        
        self.action_space=spaces.Discrete(4)
        self.actionDict = {0: -self.grid_size, 1: self.grid_size,
                            2: -1, 3: 1}
        self.possibleActions = [0, 1,2,3]


        self.agentPosition = tuple([*np.random.randint(0,self.grid_size,size=2),-1])
        self.food_list=[(-1,-1,-1) for _ in range(self.max_food)]
        
        


        x = tuple(np.random.randint(0,self.grid_size,size=2))
        
        while x == self.agentPosition:
            x=tuple(np.random.randint(0,self.grid_size,size=2))

        self.food_list[self.food_counter]=x
        self.food_counter==1
        
        self.pointsToCollect=self.POINTS_TO_COLLECT

        print(f"FOLLOWING BASES INITIALIZED: COLORS, {self.base_colors}  POSITIONS, {self.base_position}")

    

    def isTerminalState(self):
        # x = np.zeros(self.grid_size*self.n)
        # x[-1]=1
        if self.pointsToCollect==0:
            return True
        else:
            return False
        #return self.agentPosition==self.foodPosition and self.pointsToCollect==1
        # if(state[-1]==1):
        #     print(x, state)
        #     print("Terminal: ",(state == x).all())
        #return (state == x).all() #in self.stateSpacePlus and state not in self.stateSpace

    # def isFoodState(self):
    #     if (self.img[self.agentPosition]==[0,255,0]).all():
    #         #self.img = np.zeros((self.grid_size,self.grid_size,3),dtype='uint8')
    #         #self.img[self.foodPosition]=(255,0,0)
    #         #self.img[self.agentPosition]=[255,0,0]
    #         #Tak długo jak jedzenie jest na graczu
    #         # print("Found food! ",self.img[self.foodPosition])

    #         #print("EY")

    #         self.foodPosition=tuple(np.random.randint(0,self.grid_size,size=2))
            
    #         while (self.img[self.foodPosition]==[255,0,0]).all():
    #             #print("EY2")
    #             self.foodPosition=tuple(np.random.randint(0,self.grid_size,size=2))

    #         self.img[self.foodPosition]=[0,255,0]
    #         self.pointsToCollect-=1
    #         return True
    #     else:
    #         return False


    def collision_with_boundaries(self):
        if self.agentPosition[0]>=self.grid_size or self.agentPosition[0]<0 or self.agentPosition[1]>=self.grid_size or self.agentPosition[1]<0 :
            return 1
        else:
            return 0
    
    def collision_with_my_food(self):
        for food in self.food_list:
            if self.agentPosition==food:
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

        # Generate new food at random
        if np.random.uniform(low=0.0, high=1.0) <= self.new_food_prob:
            if self.food_counter>=self.max_food:
                self.food_counter=0
            
            #if food is replacing different one then remember to reset old pixel
            if self.food_list[self.food_counter][:2]!=(-1,-1):
                self.img[tuple(self.food_list[self.food_counter][:2])]=(0,0,0)
            
            x = tuple(np.random.randint(0,self.grid_size,size=2))
            while x == self.agentPosition[:2] or x in self.base_position:
                x=tuple(np.random.randint(0,self.grid_size,size=2))
            
            food_type=np.random.randint(0,self.num_bases)
            self.food_list[self.food_counter]=[*x,food_type]
            
            self.img[x] = self.base_colors[food_type]

            self.food_counter+=1
                
        done=False
        old_index=copy.deepcopy(self.agentPosition)
        self.change_agent_position(action)

        reward = 0

        for food in self.food_list:
            if tuple(food[:2]) != (-1,-1):
                if self.agentPosition[2]!=food[2]:
                    reward-=np.linalg.norm(np.array(tuple(self.agentPosition[:2])) - np.array(self.base_position[food[2]]))
                    reward-=np.linalg.norm(np.array(self.base_position[food[2]]) - np.array(food[:2]))
                else:
                    reward-=np.linalg.norm(np.array(self.agentPosition[:2]) - np.array(food[:2]))
        # # Keep away from bases!
        # for base in self.food_list:
        #     if base != (-1,-1):
        #         reward-=np.linalg.norm(np.array(self.agentPosition) - np.array(base))
        
        # # Keep away from walls!
        # # reward-=np.linalg.norm(np.array(self.agentPosition) - np.array((0,self.agentPosition[1])))
        # # reward-=np.linalg.norm(np.array(self.agentPosition) - np.array((self.grid_size,self.agentPosition[1])))
        # # reward-=np.linalg.norm(np.array(self.agentPosition) - np.array((self.agentPosition[0],0)))
        # # reward-=np.linalg.norm(np.array(self.agentPosition) - np.array((self.agentPosition[0],self.grid_size)))

        # #normalize the reward
        # reward = reward / (len(self.food_list)+4)
        
        #print(f'reward info: R: {reward}, bases: {self.food_list}')

        # don't change coords if wall was hit but preserve color change if base was hit
        if self.collision_with_boundaries() or self.collision_with_base() or self.collision_with_other_food():
            self.agentPosition=tuple([*old_index[:2],self.agentPosition[2]])
        

        if self.collision_with_my_food():
            self.points_collected+=1


        #reward += 500 if self.isFoodState() else 0   
            

        self.img[tuple(old_index[:2])]=(0,0,0)
        self.img[tuple(self.agentPosition[:2])]=(255,0,0)


        done=self.isTerminalState()

        self.current_time+=1
        
        #Add strict time horizon
        if self.current_time>=self.time_cap:
            done=True


        # Refresh boarders around bases
        for base in self.base_position:
            self.draw_boarder(base)

        # Render while training
        # if not self.game_already_initialized:
        #     self.surface=initialize_game()
        #     self.game_already_initialized = True 
        # self.render(self.surface)



        # Agent observs it's state base positions and food state positions 
        observation = [*self.agentPosition,*[coord for base in self.base_position for coord in base],*[coord for food in self.food_list for coord in food]]
        #print(observation)
        observation = np.array(observation)

        return observation, reward, \
                done, {}

    def draw_boarder(self,coords):
        if coords[0]!=0:
            self.img[coords[0]-1,coords[1]]=(255,255,255)
        if coords[0]!=self.grid_size-1:
            self.img[coords[0]+1,coords[1]]=(255,255,255)   
        if coords[1]!=0:
            self.img[coords[0],coords[1]-1]=(255,255,255)
        if coords[1]!=self.grid_size-1:
            self.img[coords[0],coords[1]+1]=(255,255,255) 

    def reset(self):

        self.points_collected=0

        self.food_counter=0
        self.img = np.zeros((self.grid_size,self.grid_size,3),dtype='uint8')

        self.pointsToCollect = self.POINTS_TO_COLLECT
        self.current_time=0
        
        self.food_list=[(-1,-1,-1) for _ in range(self.max_food)]
        self.agentPosition = tuple([*np.random.randint(0,self.grid_size,size=2),-1])
        
        #SPAWN ONE FOOD
        x = copy.deepcopy(tuple(self.agentPosition[:2]))
        while x == self.agentPosition[:2] or x in self.base_position:
            x=tuple(np.random.randint(0,self.grid_size, size=2))
        food_type=np.random.randint(0,self.num_bases)
        self.food_list[self.food_counter]=[*x,food_type]
        self.food_counter+=1
        self.img[x] = self.base_colors[food_type]
        

        self.img[tuple(self.agentPosition[:2])] = (255,0,0)


        #DRAW THE BASES
        for e,base in enumerate(self.base_position):
            self.img[base]=[color_channel-50 for color_channel in self.base_colors[e]]
            self.draw_boarder(base)

        observation=[*self.agentPosition,*[coord for base in self.base_position for coord in base],*[coord for food in self.food_list for coord in food]]
        #print(observation)
        observation = np.array(observation)
        return observation

    def render(self, surface):
        #print(self.img)
        #grid_2d = np.resize(self.grid, (self.n,self.grid_size))
        #world_map=read_map(observation)
        draw_map(surface, self.img,self.grid_size,self.grid_size)
        draw_grid(surface,self.grid_size,self.grid_size) 
        pygame.display.set_caption(f"POINTS COLLECTED: {self.points_collected}")
        pygame.display.flip()
        time.sleep(0.01)


    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]