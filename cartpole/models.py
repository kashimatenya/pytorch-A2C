import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

#
class ValueNet(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        
        self._layers =  nn.Sequential(         
                                nn.Linear(4, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 1)
                            )
    #def

    def forward(self, x):
        v = self._layers(x)
        return v
    #def
#class


class PolicyNet(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

        self._layers =   nn.Sequential(         
                                nn.Linear(4, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 2),
                                nn.Softmax(dim=-1)
                            )
    #def

    def forward(self, x):
        p = self._layers(x)
        return distributions.Categorical(p)
    #def
#class



class DualNet(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        
        self._shared_layers =  nn.Sequential(         
                                nn.Linear(4, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 64),
                                nn.ReLU(inplace=True),
                            )

        self._value_layers = nn.Sequential(
                                nn.Linear(64, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 1)
                            )


        self._policy_layers = nn.Sequential(
                                nn.Linear(64, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 2),
                                nn.Softmax(dim=-1)
                            )
    #def


    def forward(self, x):
        branch = self._shared_layers(x)
        V = self._V_head(branch)
        pi = self._pi_head(branch)
        return V, pi
    #def


    def forward_V(self, x):
        branch = self._shared_layers(x)
        V = self._V_head(branch)
        return V
    #def


    def forward_pi(self, x):
        branch = self._shared_layers(x)
        pi = self._pi_head(branch)
        return pi
    #def


    def _V_head(self, branch):
        V = self._value_layers(branch)
        return V
    #def


    def _pi_head(self, branch):
        p = self._policy_layers(branch)
        return distributions.Categorical(p)
    #def
#class
