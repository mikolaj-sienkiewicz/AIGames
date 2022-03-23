import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gym import spaces

import numpy as np
#import keras.backend.tensorflow_backend as backend
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
# from keras.optimizers import Adam
# from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
            n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')#('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) #derived from nn class

    def forward(self, state):
        x = F.relu(self.fc1(state.type(T.float)))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:
    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def process_transition(self, observation, action, reward, next_observation, done):
        raise NotImplementedError()
        
    def get_action(self, observation, learning):
        raise NotImplementedError()


class NeuralQLearningAgent(Agent):
    def __init__(self, env, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.05, eps_dec=5e-3,
                update_every=20, useReplay_mem=True,useNetwork_freezing=True,useDouble_Qlearning=True, q_eval=None):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        
        #jeżeli useBach_mem=False to rozmiar pamięci i batcha jest równy 1
        self.mem_size = max_mem_size if useReplay_mem else 1
        self.batch_size = batch_size if useReplay_mem else 1
        
        self.mem_cntr = 0
        self.iter_cntr = 0
        #self.replace_target = 100
        
        
        #Pierwsza sieć - 
        if q_eval==None:
            self.q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                        fc1_dims=256, fc2_dims=256)
        else:
            self.q_eval = q_eval
        
        self.useDouble_Qlearning=useDouble_Qlearning
        if useDouble_Qlearning:
            self.q_next = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=256, fc2_dims=256)#.to(device)
        else:
            self.q_next = self.q_eval
            
        self.update_every = update_every if useNetwork_freezing else 1
        
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
    
    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def replace_target_network(self):
        if self.iter_cntr % self.update_every == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
    
    def process_transition(self, observation, action, reward, next_observation, done):
        self.store_transition(observation, action, reward, 
                                    next_observation, done)
        
        #nnajpierw wypelnijmy pamiec na tyle zeby byl chociaz batch
        if self.mem_cntr < self.batch_size:
            return
        
        #Network freezing
        self.replace_target_network()
        
        #wyzeruj gradient na optimizerze dla sieci eval
        self.q_eval.optimizer.zero_grad()
        
        #wybierz co jest wieksze, żeby z tego losowac
        max_mem = min(self.mem_cntr, self.mem_size)
        
        #wybierz (BEZ POWTORZEN) losowe indeksy z tych co sa do wybrania
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        #wypełnienienie Tensorów (o wielkkosci batcha) wszystkich ptorzebnych składowych
        state_batch = T.tensor(self.state_memory[batch]).to(self.q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.q_eval.device)

        new_state_batch2=new_state_batch
    
    
        q_eval = self.q_eval.forward(state_batch)[batch_index, action_batch]
        
        #Q_next zamiast q_eval żeby było Double DQN
        if self.useDouble_Qlearning:
            q_next = self.q_next.forward(new_state_batch2)
        else:
            q_next = self.q_eval.forward(new_state_batch)
        
        #jeżeli stan jest terminalny to wartosc to 0
        q_next[terminal_batch] = 0.0
        
        #oblicz cel do którego staramy sie dążyć bazując na wzorze - wykorzystujemy nagrody
        #najwieksza wartosc patrzac po akcjach i mnoznika gamma (czyli tego jak przyszlosc jest wazna)
        #bieremy element 0 bo Tmax zwraca tez index a go nie chcemy
        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]
        
        #oblicz funkcje straty i zrób wsteczną propagację i update optimizera sieci eval
        loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        
        self.iter_cntr += 1
        
    def get_action(self, observation, learning):
        if learning:
            #jak sie uczymy epsilon greedy (decaying)
            if np.random.random() > self.epsilon:
                state = T.tensor([observation]).to(self.q_eval.device)
                #print(state,observation)
                actions = self.q_eval.forward(state)
                action = T.argmax(actions).item()
            else:
                action = np.random.choice(self.action_space)
            
            #dekrementacja epsilona (epsilon_decay -> chcemy na poczatku bardziej randomowe akcje a potem mniej)
            #print(self.epsilon)
            self.epsilon = max(self.epsilon - self.eps_dec,self.eps_min)
        else:
            #jak sie nie uczymy bierz zachlannie
            state = T.tensor([observation]).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        return action