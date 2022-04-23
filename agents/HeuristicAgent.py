import sys,os 
sys.path.append(os.path.join(sys.path[0], ".."))

from environment.GridEnv import GridWorld
from utils import run_agent_in_env

class HeuristicAgent:
    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env=env

    def process_transition(self, observation, action, reward, next_observation, done):
        raise NotImplementedError()
        
    def get_action(self, target):
        agent_coords = tuple(self.env.agentPosition[:2])
        x_delta=target[1]-agent_coords[1]
        y_delta=target[0]-agent_coords[0]
        if abs(x_delta)>abs(y_delta):
            if x_delta > 0:
                return 2 #'right'
            else:
                return 3 #'left'
        else:
            if y_delta > 0:
                return 1 # down
            else:
                return 0 # up
    
    def unblock_manouver(self,action):
        old_index=self.env.agentPosition
        self.env.change_agent_position(action)
        if self.env.collision_with_boundaries() \
            or self.env.collision_with_other_food() \
                or self.env.collision_with_other_agent() \
                    or self.env.collision_with_same_color_base():
            self.env.agentPosition=tuple([*old_index])
            return self.env.actionSpaceSample()
        else: 
            self.env.agentPosition=tuple([*old_index])
            return action

    def predict(self, _):
        target = self.pick_best_food()
        action = self.get_action(target)
        action = self.unblock_manouver(action)
        return action, _

    def manhattan(self, a, b):
        return sum(abs(val1-val2) for val1, val2 in zip(a,b))

    def pick_best_food(self):
        agent_color = self.env.agentPosition[2]
        agent_coords = tuple(self.env.agentPosition[:2])
        best_distance=self.env.grid_size*10
        best_food=None
        for food in self.env.food_list:
            current_dist=0
            food_coords=tuple(food[:2])
            food_color=food[2]
            if food_color!=agent_color:
                current_dist+=self.manhattan(self.env.base_position[food[2]],agent_coords)
            current_dist+=self.manhattan(food_coords,agent_coords)
            if current_dist<best_distance:
                best_distance=current_dist
                if food_color==agent_color:
                    best_food=food_coords
                else:
                    best_food=self.env.base_position[food[2]]
        return best_food

if __name__=='__main__':
    GRID_SIDE_LENGTH=12
    env = GridWorld(GRID_SIDE_LENGTH)
    env.reset()
    agent = HeuristicAgent(env)
    rewards = run_agent_in_env(env, agent, 5, learning=False, plot=False, render=True, plot_interval=20)