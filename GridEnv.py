import numpy as np
import matplotlib.pyplot as plt
from gym import spaces 
class GridWorld(object):
    def __init__(self, m, n, magicSquares):
        self.grid = np.zeros(m*n)
        self.m = m
        self.n = n
        self.observation_space=spaces.Discrete(m*n)
        self.action_space=spaces.Discrete(4)
        self.observation_size=self.m*self.n
        #self.stateSpace = [i for i in range(self.m*self.n)]
        #self.stateSpace.remove(80)
        #self.stateSpacePlus = [i for i in range(self.m*self.n)]
        self.actionDict = {0: -self.m, 1: self.m,
                            2: -1, 3: 1}
        self.possibleActions = [0, 1,2,3]
        # dict with magic squares and resulting squares
        #self.addMagicSquares(magicSquares)
        self.agentPosition = 0

    def isTerminalState(self, state):
        x = np.zeros(self.m*self.n)
        x[-1]=1
        # if(state[-1]==1):
        #     print(x, state)
        #     print("Terminal: ",(state == x).all())
        return (state == x).all() #in self.stateSpacePlus and state not in self.stateSpace

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
        if 0 > newAgentPos or newAgentPos > self.n*self.m:#self.stateSpacePlus:
            return True
        #if we're trying to wrap around to next row
        elif oldAgentPos % self.m == 0 and newAgentPos  % self.m == self.m - 1:
            return True
        elif oldAgentPos % self.m == self.m - 1 and newAgentPos % self.m == 0:
            return True
        else:
            return False

    def step(self, action):
        #agentX, agentY = self.getAgentRowAndColumn()
        #old_state = [1 if self.agentPosition==i else 0 for i in range(self.observation_size)]
        #11111111111111111111
        old_index=self.agentPosition
        index=self.agentPosition + self.actionDict[action]
        #print(self.actionDict[action],index,old_index,self.offGridMove(index, old_index))
        #print(resultingState)
        # resultingState = self.agentPosition + self.actionDict[action] 
        # if resultingState in self.magicSquares.keys():
        #     resultingState = self.magicSquares[resultingState]

        #00000000000000000000000

        reward = -1 if not self.isTerminalState(self.grid) else 0

        if not self.offGridMove(index, old_index):
            self.agentPosition=index
            self.grid = [1 if index==i else 0 for i in range(self.observation_size)]

        return self.grid, reward, \
                self.isTerminalState(self.grid), None

    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m*self.n))
        #self.addMagicSquares(self.magicSquares)
        self.grid[self.agentPosition] = 1
        return self.grid #self.agentPosition

    def render(self):
        print('------------------------------------------')
        grid_2d = np.resize(self.grid, (self.n,self.m))
        for row in grid_2d:
            for col in row:
                if col == 0:
                    print('-', end='\t')
                elif col == 1:
                    print('X', end='\t')
                elif col == 2:
                    print('Ain', end='\t')
                elif col == 3:
                    print('Aout', end='\t')
                elif col == 4:
                    print('Bin', end='\t')
                elif col == 5:
                    print('Bout', end='\t')
            print('\n')
        print('------------------------------------------')

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