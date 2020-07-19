from datetime import datetime

import gym
from gym import wrappers

import numpy as np

#wrapped enviroment of CartPole-v0
class CartPoleV0:
    def __init__(self, monitor=False):
        env = gym.make('CartPole-v0')
        
        if monitor:
            directory = "./replay/log/" + datetime.now().strftime('%Y%m%d%H%M%S')
            env = wrappers.Monitor(env, directory, video_callable = lambda x:True)
        
        self._env = env
        self._env_state = None

        self._steps = 0
        self._max_steps = 200

        self._reward_scaling = 0.05
        self._state_scaling = np.array((1.0/2.4, 1.0, 1.0/12, 1.0), dtype=np.float32)
        self._state_bias = np.array((0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    #def


    def reset(self,):
        self._env_state = self._env.reset()
        self._steps=0
        return self.observe()
    #def


    def step(self, action):
        (state_dash_, reward_, done_, _) = self._env.step(int(action))

        self._env_state = state_dash_
        self._steps +=1

        reward = self._reward_scaling*reward_
        
        state_dash = self.observe()

        info = {}

        if done_ :
            #distinguish wherther the pole is crushed or not 
            if self._steps == self._max_steps:
                done = True
                info["is_terminal"] = False
            else:
                done = True
                info["is_terminal"] = True
        else:
            done = False


        return (state_dash, reward, done, info)
    #def


    def observe(self, normalize=True):
        state_ = self._env_state.astype(dtype=np.float32)

        if normalize == True:
            state = self._state_scaling*state_ + self._state_bias
        else:
            state = state_ 

        return state
    #def


    def render(self):
        self._env.render()
        return
    #def


    def close(self):
        self._env.close()
    #def
    

    @property
    def max_steps(self):
        return self._max_steps
    #def

    @property
    def steps(self):
        return self._steps
    #def
#class
