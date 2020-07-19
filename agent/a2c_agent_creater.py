import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from agent.a2c_agent import A2CAgent
from agent.scheduler_functions import ExponentialDecayWithWarmUp

from math import exp, log


class A2CAgentCreater:
    def __init__(self, filename):
        with open(filename, 'r') as file :
            self._parameters = json.load(file)
    #def


    def create(self, model, device=torch.device("cpu")):
        parameters = self._parameters

        gamma = parameters["gamma"]
        n_steps = parameters["n_steps"]

        #coefficient in calcurating loss
        V_loss_coef = parameters["V_loss_coef"]
        pi_loss_coef = parameters["pi_loss_coef"]
        entropy_coef = parameters["entropy_coef"]

        #optimizer
        optimizer = optim.AdamW(model.parameters(), **parameters["optimizer"])
        
        #learning rate scheduler
        if parameters["lr_scheduler"] is not None:
            scheduler_function = ExponentialDecayWithWarmUp(**parameters["lr_scheduler"])
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, scheduler_function)
        else:
            lr_scheduler = None
        
        gradient_Clipping = parameters["gradient_clipping"]

        batch_size= parameters["batch_size"]

        agent = A2CAgent(
                        gamma,
                        n_steps,
                        model,
                        device,
                        V_loss_coef, 
                        pi_loss_coef, 
                        entropy_coef,
                        optimizer,
                        lr_scheduler,
                        gradient_Clipping,
                        batch_size
                    )
 
        return agent
    #def
#class
