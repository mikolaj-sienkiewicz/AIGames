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

    def __init__(self, m, n):
        super().__init__()
        self.time_cap=3000
        self.current_time=0
        self.new_food_prob=0.005
        self.game_already_initialized=False
        #self.grid = np.zeros(m*n)
        self.m = m
        #self.observation_space=spaces.Discrete(m*n)
        self.observation_space = spaces.Box(low=0, high=255,shape=(self.m,self.m,3), dtype=np.uint8)
        self.action_space=spaces.Discrete(4)
        #self.observation_size=self.m*self.n
        #self.stateSpace = [i for i in range(self.m*self.n)]
        #self.stateSpace.remove(80)
        #self.stateSpacePlus = [i for i in range(self.m*self.n)]
        self.actionDict = {0: -self.m, 1: self.m,
                            2: -1, 3: 1}
        self.possibleActions = [0, 1,2,3]
        # dict with magic squares and resulting squares
        #self.addMagicSquares(magicSquares)
        self.agentPosition = tuple(np.random.randint(0,self.m,size=2))
        self.foodPosition = tuple(np.random.randint(0,self.m,size=2))
        while self.foodPosition == self.agentPosition:
            self.foodPosition=tuple(np.random.randint(0,self.m,size=2))
        self.pointsToCollect=self.POINTS_TO_COLLECT

    def isTerminalState(self):
        # x = np.zeros(self.m*self.n)
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

    def isFoodState(self):
        if (self.img[self.agentPosition]==[0,255,0]).all():
            #self.img = np.zeros((self.m,self.m,3),dtype='uint8')
            #self.img[self.foodPosition]=(255,0,0)
            #self.img[self.agentPosition]=[255,0,0]
            #Tak długo jak jedzenie jest na graczu
            # print("Found food! ",self.img[self.foodPosition])

            #print("EY")

            self.foodPosition=tuple(np.random.randint(0,self.m,size=2))
            
            while (self.img[self.foodPosition]==[255,0,0]).all():
                #print("EY2")
                self.foodPosition=tuple(np.random.randint(0,self.m,size=2))

            self.img[self.foodPosition]=[0,255,0]
            self.pointsToCollect-=1
            return True
        else:
            return False


    def collision_with_boundaries(self,newAgentPos):
        if newAgentPos[0]>=self.m or newAgentPos[0]<0 or newAgentPos[1]>=self.m or newAgentPos[1]<0 :
            return 1
        else:
            return 0

    def change_agent_position(self,button_direction):
        # Change the agent position based on the button direction
        if button_direction == 1:
            self.agentPosition = (self.agentPosition[0]+1,self.agentPosition[1])
        elif button_direction == 0:
            self.agentPosition = (self.agentPosition[0]-1,self.agentPosition[1])
        elif button_direction == 2:
            self.agentPosition = (self.agentPosition[0],self.agentPosition[1]+1)
        elif button_direction == 3:
            self.agentPosition = (self.agentPosition[0],self.agentPosition[1]-1)


    def step(self, action):
        # if np.random.uniform(low=0.0, high=1.0) <= self.new_food_prob:
        #     new_f=tuple(np.random.randint(0,self.m,size=2))
        #     while new_f==[255,0,0]:
        #         new_f=tuple(np.random.randint(0,self.m,size=2))
        #     self.img[new_f]=(0,255,0)

                
        done=False
        old_index=copy.deepcopy(self.agentPosition)
        self.change_agent_position(action)

        reward = -np.linalg.norm(np.array(self.agentPosition) - np.array(self.foodPosition)) #if self.collision_with_boundaries(self.agentPosition) else 0
        #if self.collision_with_boundaries(self.agentPosition)

        if self.collision_with_boundaries(self.agentPosition):
            #reward += -1 if not self.isTerminalState() else 0
            #self.grid = np.zeros(self.observation_size)
            # self.img[old_index]=(0,0,0)
            # self.img[self.foodPosition]=(0,255,0)
            self.agentPosition=copy.deepcopy(old_index)
        
        reward += 500 if self.isFoodState() else 0   
            

        self.img[old_index]=(0,0,0)
        self.img[self.agentPosition]=(255,0,0)


        done=self.isTerminalState()
        self.current_time+=1
        
        #Add strict time horizon
        if self.current_time>=self.time_cap:
            done=True

        # Render while training
        # if not self.game_already_initialized:
        #     self.surface=initialize_game()
        #     self.game_already_initialized = True  
        # self.render(self.surface)




        return self.img, reward, \
                done, {}

    def reset(self):
        self.img = np.zeros((self.m,self.m,3),dtype='uint8')
        self.pointsToCollect = self.POINTS_TO_COLLECT
        self.current_time=0
        self.agentPosition = tuple(np.random.randint(0,self.m,size=2))
        self.foodPosition = copy.deepcopy(self.agentPosition)
        while self.foodPosition == self.agentPosition:
            self.foodPosition=tuple(np.random.randint(0,self.m, size=2))
        #self.grid = np.zeros((self.m*self.n))
        #self.addMagicSquares(self.magicSquares)
        self.img[self.agentPosition] = (255,0,0)
        self.img[self.foodPosition] = (0,255,0)
        
        return self.img #self.agentPosition

    def render(self, surface):
        #print(self.img)
        #grid_2d = np.resize(self.grid, (self.n,self.m))
        #world_map=read_map(observation)
        draw_map(surface, self.img,self.m,self.m)
        draw_grid(surface,self.m,self.m) 
        pygame.display.flip()
        time.sleep(0.1)


    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]

if __name__ == '__main__':
    # map magic squares to their connecting square
    magicSquares = {18: 54, 63: 14}
    env = GridWorld(9, 9, magicSquares)
    # model hyperparameters
    ALPHA = 0.1
    GAMMA = 1.0
    EPS = 1.0

    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    numGames = 50000
    totalRewards = np.zeros(numGames)
    for i in range(numGames):
        if i % 5000 == 0:
            print('starting game ', i)
        done = False
        epRewards = 0
        observation = env.reset()
        while not done:
            rand = np.random.random()
            action = maxAction(Q,observation, env.possibleActions) if rand < (1-EPS) \
                                                    else env.actionSpaceSample()
            observation_, reward, done, info = env.step(action)
            epRewards += reward

            action_ = maxAction(Q, observation_, env.possibleActions)
            Q[observation,action] = Q[observation,action] + ALPHA*(reward + \
                        GAMMA*Q[observation_,action_] - Q[observation,action])
            observation = observation_
        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0
        totalRewards[i] = epRewards

    plt.plot(totalRewards)
    plt.show()