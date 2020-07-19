import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SeparateA2CModel(nn.Module):
    def __init__(self, V_model, pi_model):
        super(self.__class__, self).__init__()
        
        self._V_model = V_model
        self._pi_model = pi_model
    #def
    
    def forward(self, x):
        return self.forward_V(x), self.forward_pi(x)
    #def

    def forward_V(self, x):
        return self._V_model.forward(x)
    #def

    def forward_pi(self, x):
        return self._pi_model.forward(x)
    #def
#class


class SharedA2CModel(nn.Module):
    def __init__(self, model):
        super(self.__class__, self).__init__()

        self._model = model
    #def

    def forward(self, x):
        return self._model.forward(x)
    #def

    def forward_V(self, x):
        return self._model.forward_V(x)
    #def

    def forward_pi(self, x):
        return self._model.forward_pi(x)
    #def
#def
