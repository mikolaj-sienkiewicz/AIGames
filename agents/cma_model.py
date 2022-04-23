import torch
from torch import nn
import numpy as np

class Model:
    def __init__(self, obs_space, num_outputs):
        self.model = nn.Sequential(
            nn.Linear(obs_space.shape[0], 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, num_outputs),
            nn.Softmax(dim=0)
        )
        self.num_parameters: int = 0
        x=0
        for p in self.model.parameters():
            p.requires_grad = False
            #print("STOP: ", p.data.shape, np.prod(np.array(p.data.shape)))
            self.num_parameters += np.prod(np.array(p.data.shape))
            x+=1
        #print("WOWOWOW: ",x)

    @classmethod
    def from_weights(cls, obs_space, num_outputs, weights):
        model = cls(obs_space, num_outputs)
        idx: int = 0
        for param in model.model.parameters():
            layer_shape: np.ndarray = np.array(param.data.shape)
            length: int = int(np.prod(layer_shape))
            #print("EY: ",weights.shape,layer_shape)
            param.data = torch.from_numpy(np.array(weights)[idx:idx + length].reshape(layer_shape)).float()
            idx += length
        return model

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out: np.ndarray = self.model(torch.from_numpy(observation).float())
        return np.array(out)