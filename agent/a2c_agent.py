import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions


class A2CAgent:
    def __init__(
                self,
                gamma,
                n_steps,          
                model,   
                device,
                V_loss_coef,    
                pi_loss_coef,   
                entropy_coef,   
                optimizer,
                lr_scheduler,   
                gradient_clipping,
                batch_size
            ):

        self._gamma = gamma
        self._n_steps = n_steps

        self._model = model
        self._device = device

        self._model.to(self._device)
        self._model.eval()
            
        self._V_loss_coef = V_loss_coef
        self._pi_loss_coef = pi_loss_coef
        self._entropy_coef = entropy_coef


        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._gradient_clipping = gradient_clipping

        self._batch_size = batch_size

        self._latest_training_loss = self._get_loss_dict(None, None, None)
    #def


    def action(self, state, deterministic=False):
        with torch.no_grad():
            pi = self._model.forward_pi(torch.tensor(state, device=self._device)) 
        
        if deterministic:
            if type(pi) == distributions.Categorical:
                return (pi.probs.argmax(dim=-1)).numpy()
            #if
            if type(pi) == distributions.Normal:
                return pi.loc.numpy()
            #if

            raise Exception( type(pi) + "is not supported")
        #if
        
        return pi.sample().to("cpu").numpy()
    #def


    def train(self, experiences):
        V_loss_coef =self._V_loss_coef
        pi_loss_coef = self._pi_loss_coef
        entropy_coef = self._entropy_coef
    
        model = self._model.train()

        (V_loss, pi_loss, pi_entropy) = self._get_loss(experiences)        
        loss = V_loss_coef*V_loss +pi_loss_coef*(pi_loss -entropy_coef*pi_entropy)
        
        model.zero_grad()
        loss.backward()

        #gradient_clipping 
        if self._gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self._gradient_clipping)
    
        self._optimizer.step()
    
        #update stats
        losses = (V_loss, pi_loss, pi_entropy)
        self._latest_training_loss = self._get_loss_dict(*map(float, losses))
        
        self._model = model.eval()
        return
    #def

    #set using device
    def to(self, device):
        self._device = device
        self._model.to(device)
        return
    #def


    #estimate loss from memory
    def estimate(self, experiences): 
        losses = self._get_loss(experiences)
        return self._get_loss_dict( *map(float, losses) ) 
    #def


    def step_scheduler(self):
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        return
    #def

    def save(self, model_filename):
        self._model.to(torch.device("cpu"))
        torch.save(self._model.state_dict(), model_filename)
        self._model.to(self._device)
        return
    #def


    def load(self, model_filename):
        self._model.load_state_dict(torch.load(model_filename))
        self._model.to(self._device)
        return
    #def

    @property
    def stats(self):
        training_loss = self._latest_training_loss
        learning_rate = self._optimizer.param_groups[0]["lr"]
        return {"training_loss":training_loss, "learning_rate":learning_rate}
    #def


    @property
    def n_steps(self):
        return self._n_steps
    #def

    @property
    def batch_size(self):
        return self._batch_size
    #def

    #calculate losses
    def _get_loss(self, experiences):
        n_steps = self._n_steps
        model = self._model
        device = self._device
        
        V_loss = torch.zeros((1,), dtype=torch.float32, device=device)
        pi_loss = torch.zeros((1,), dtype=torch.float32, device=device)
        pi_entropy = torch.zeros((1,), dtype=torch.float32, device=device)
        count = torch.zeros((1,), dtype=torch.float32, device=device)

        #calculate loss in every sequence
        for State_, Action_, Reward_, end_state_, is_terminal_ in experiences:


            State  = torch.tensor(State_, device=device)
            Action = torch.tensor(Action_, device=device)
            Reward = torch.tensor(np.expand_dims(Reward_, axis=-1), device=device)
            is_terminal = is_terminal_

            (Value, Pi) = model.forward(State)

            if is_terminal:
                value_at_end_state = torch.zeros((1,1), dtype=torch.float32, device=device)
            else:
                with torch.no_grad():
                    end_state = torch.tensor(end_state_.reshape(1,-1), device=device)
                    value_at_end_state = model.forward_V(end_state)
            #if-else

            LogProb = Pi.log_prob(Action)            
            Entropy = Pi.entropy()

            experience_count = State.shape[0]

            if experience_count > n_steps:
                (V_loss_, pi_loss_, pi_entropy_, count_) = self._get_loss_fragment(Value, Reward, LogProb, Entropy)
                V_loss += V_loss_
                pi_loss += pi_loss_
                pi_entropy += pi_entropy_
                count += count_
    
            for k in reversed(range(1, min(n_steps, experience_count)+1)):
                (V_loss_, pi_loss_, pi_entropy_, count_) = self._get_loss_fragment_at_last_k_steps(Value, value_at_end_state, Reward, LogProb, Entropy, k)
                V_loss += V_loss_
                pi_loss += pi_loss_
                pi_entropy += pi_entropy_
                count += count_
    
        #for experience in experiences:

        V_loss /= count
        pi_loss /=count
        pi_entropy /= count

        return V_loss, pi_loss, pi_entropy
    #def


    def _get_loss_fragment(self, Value, Reward, LogProb, Entropy):
        gamma = self._gamma
        n_steps = self._n_steps

        advantage_ = pow(gamma, n_steps)*(Value[n_steps:].detach())
        for k in reversed(range(n_steps)):
            advantage_ += pow(gamma, k)*Reward[k:-n_steps+k]

        advantage_ -= Value[:-n_steps]
        advantage = advantage_.squeeze(dim=-1)

        V_loss = (advantage.pow(2)).sum(dim=0)
        pi_loss = (-LogProb[:-n_steps]*(advantage.detach())).sum(dim=0)
        pi_entropy = Entropy[:-n_steps].sum(dim=0)
        count = advantage.shape[0]

        return V_loss, pi_loss, pi_entropy, count
    #def


    def _get_loss_fragment_at_last_k_steps(self, Value, value_at_end_state, Reward, LogProb, Entropy, k):
        gamma = self._gamma

        advantage_ = pow(gamma, k)*(value_at_end_state.detach())
        for l in reversed(range(k)):
            advantage_ += pow(gamma, l)*Reward[-k+l]

        advantage_ -= Value[-k]
        advantage = advantage_.squeeze(dim=-1)

        V_loss = (advantage.pow(2)).sum(dim=0)
        pi_loss = (-LogProb[-k]*(advantage.detach())).sum(dim=0)
        pi_entropy = Entropy[-k].sum(dim=0)
        count = advantage.shape[0]

        return V_loss, pi_loss, pi_entropy, count
    #def


    def _get_loss_dict(self, V_loss, pi_loss, pi_entropy):
        return {"V":{"loss":V_loss}, "pi":{"loss":pi_loss, "entropy":pi_entropy}  }
    #def
#class
