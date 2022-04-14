import numpy as np
import matplotlib.pyplot as plt
from gym import spaces 
import pygame
from render_controller import draw_map, draw_grid
import time 
import random


class GridWorld(object):
    metadata = {'render.modes': ['human']}
    POINTS_TO_COLLECT=1

    def __init__(self, m):
        super(GridWorld, self).__init__()
        self.time_cap=max(m*m*5, 256)
        self.current_time=0

        self.grid = np.zeros(m*m)
        self.m = m
        self.n = m
                
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):

        self.observation_space = spaces.Box(low=-self.m, high=self.m,
                                            shape=(5,), dtype=np.float32)
        self.observation_size=self.m*self.n
        #self.stateSpace = [i for i in range(self.m*self.n)]
        #self.stateSpace.remove(80)
        #self.stateSpacePlus = [i for i in range(self.m*self.n)]
        self.actionDict = {0: -self.m, 1: self.m,
                            2: -1, 3: 1}
        self.possibleActions = [0, 1,2,3]
        # dict with magic squares and resulting squares
        #self.addMagicSquares(magicSquares)
        self.agentPosition = np.random.randint(self.m*self.n)
        self.foodPosition = self.agentPosition
        while self.foodPosition == self.agentPosition:
            self.foodPosition=np.random.randint(self.m*self.n)
        self.pointsToCollect=self.POINTS_TO_COLLECT

    def collision_with_apple(self, apple_position, score):
        apple_position = [random.randrange(1,self.m),random.randrange(1,self.m)]
        score += 1
        return apple_position, score

    def collision_with_boundaries(self,snake_head):
        if snake_head[0]>=self.m or snake_head[0]<0 or snake_head[1]>=self.m or snake_head[1]<0 :
            return 1
        else:
            return 0


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
        if self.agentPosition==self.foodPosition:
            
            self.grid[self.foodPosition]=1

            #Tak długo jak jedzenie jest na graczu
            while self.grid[self.foodPosition]==1:
                self.foodPosition=np.random.randint(0,self.m*self.n)

            self.grid[self.foodPosition]=2
            self.pointsToCollect-=1
            return True
        else:
            return False
    # def addMagicSquares(self, magicSquares):
    #     self.magicSquares = magicSquares
    #     i = 2
    #     for square in self.magicSquares:
    #         x = square // self.m
    #         y = square % self.n
    #         self.grid[x][y] = i
    #         i += 1
    #         x = magicSquares[square] // self.m
    #         y = magicSquares[square] % self.n
    #         self.grid[x][y] = i
    #         i += 1

    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.m
        y = self.agentPosition % self.n
        return x, y

    # def setState(self, state):
    #     x, y = self.getAgentRowAndColumn()
    #     self.grid[x][y] = 0
    #     self.agentPosition = state
    #     x, y = self.getAgentRowAndColumn()
    #     self.grid[x][y] = 1

    def offGridMove(self, newAgentPos, oldAgentPos):
        #return False
        # # if we move into a row not in the grid
        if 0 > newAgentPos or newAgentPos >= self.n*self.m:#self.stateSpacePlus:
            return True
        #if we're trying to wrap around to next row
        elif oldAgentPos % self.m == 0 and newAgentPos  % self.m == self.m - 1:
            return True
        elif oldAgentPos % self.m == self.m - 1 and newAgentPos % self.m == 0:
            return True
        else:
            return False

    def step(self, action):
        
        button_direction = action
        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += 10
        elif button_direction == 0:
            self.snake_head[0] -= 10
        elif button_direction == 2:
            self.snake_head[1] += 10
        elif button_direction == 3:
            self.snake_head[1] -= 10


        apple_reward = 0
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position, self.score = self.collision_with_apple(self.apple_position, self.score)
            self.snake_position.insert(0,list(self.snake_head))
            apple_reward = 10000

        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
        
        # On collision kill the snake and print the score
        if self.collision_with_boundaries(self.snake_head) == 1 or self.collision_with_self(self.snake_position) == 1:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((20,20,3),dtype='uint8')
            # cv2.putText(self.img,'Your Score is {}'.format(self.score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            # cv2.imshow('a',self.img)
            self.done = True
        


        euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))

        self.total_reward = ((250 - euclidean_dist_to_apple) + apple_reward)/100

        print(self.total_reward)


        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward

        if self.done:
            self.reward = -10
        info = {}


        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        # create observation:

        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length]
        observation = np.array(observation)

        return observation, self.total_reward, self.done, info

    def reset(self):
        self.pointsToCollect = self.POINTS_TO_COLLECT
        self.current_time=0
        self.agentPosition = np.random.randint(self.m*self.n)
        self.foodPosition = self.agentPosition
        while self.foodPosition == self.agentPosition:
            self.foodPosition=np.random.randint(self.m*self.n)
        self.grid = np.zeros((self.m*self.n))
        #self.addMagicSquares(self.magicSquares)
        self.grid[self.agentPosition] = 1
        self.grid[self.foodPosition] = 2

        return self.grid #self.agentPosition

        self.img = np.zeros((20,20,3),dtype='uint8')
        # Initial Snake and Apple position
        self.snake_position = [self.m//2,self.m//2]
        self.apple_position = [random.randrange(self.m),random.randrange(self.m)]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.prev_reward = 0

        self.done = False

        head_x = self.snake_position[0]
        head_y = self.snake_position[1]

        #snake_length = len(self.snake_position)
        apple_delta_x = self.apple_position[0] - head_x
        apple_delta_y = self.apple_position[1] - head_y

        #self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)  # however long we aspire the snake to be
        for i in range(SNAKE_LEN_GOAL):
            self.prev_actions.append(-1) # to create history

        # create observation:
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
        observation = np.array(observation)

        return observation



    def render(self, surface):
        print(self.grid)
        grid_2d = np.resize(self.grid, (self.n,self.m))
        #world_map=read_map(observation)
        draw_map(surface, grid_2d,self.n,self.m)
        draw_grid(surface,self.n,self.m) 
        pygame.display.flip()
        time.sleep(0.1)
        # for row in grid_2d:
        #     for col in row:
        #         if col == 0:
        #             print('-', end='\t')
        #         elif col == 1:
        #             print('X', end='\t')
        #         elif col == 2:
        #             print('Ain', end='\t')
        #         elif col == 3:
        #             print('Aout', end='\t')
        #         elif col == 4:
        #             print('Bin', end='\t')
        #         elif col == 5:
        #             print('Bout', end='\t')
        #     print('\n')
        # print('------------------------------------------')


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