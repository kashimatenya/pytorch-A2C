import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions


class ValueNet(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

        self._layers = nn.Sequential(         
                                nn.Linear(3, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 1)
                            )
    #def


    def forward(self, x):
        V = self._layers(x)        
        return V
    #def
#class


class PolicyNet(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        
        self._layers = nn.Sequential(         
                                nn.Linear(3, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 128),
                                nn.ReLU(inplace=True),
                                nn.Linear(128, 64),
                                nn.ReLU(inplace=True),
                                nn.Linear(64, 2),
                                nn.Tanh()
                            )

        self._scale = torch.tensor((2, 1)).reshape(1, 2)
        self._bias  = torch.tensor((0, 1+1e-3)).reshape(1, 2)
    #def


    def forward(self, x):
        
        theta = self._layers(x)
        theta = self._scale*theta+self._bias

        mu = theta[:, 0]
        sigma = theta[:, 1]

        return distributions.Normal(mu, sigma)
    #def
    
#class


class DualNet(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

        self._shared_layers = nn.Sequential( 
                                    nn.Linear(3, 256),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256, 128),
                                    nn.ReLU(inplace=True),
                                )

        self._value_layers = nn.Sequential(
                                    nn.Linear(128, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 1),
                                )

        self._policy_layers = nn.Sequential(
                                    nn.Linear(128, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, 2),
                                    nn.Tanh()
                                )

        self._theta_scale = torch.tensor((2, 1)).reshape(1, 2)
        self._theta_bias  = torch.tensor((0, 1+1e-3)).reshape(1, 2)
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
        theta  = self._policy_layers(branch)
            
        theta = self._theta_scale*theta+self._theta_bias

        mu = theta[:,0]
        sigma = theta[:,1]

        pi = distributions.Normal(mu, sigma)    
        return pi
    #def
#class
