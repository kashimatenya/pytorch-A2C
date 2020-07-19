from datetime import datetime

import gym
from gym import wrappers

import numpy as np

#wrapped environment of Penddulum-v0
class PendulumV0:
    def __init__(self, monitor=False):
        env = gym.make("Pendulum-v0")

        if monitor:
            directory = "./replay/log/" + datetime.now().strftime('%Y%m%d%H%M%S')
            env = wrappers.Monitor(env, directory, video_callable = lambda x:True)

        self._env = env
        self._env_state = None

        self._steps = 0
        self._max_steps = 200

        self._reward_scaling = 0.2
        self._state_scaling = np.array((1.0, 1.0, 1.0/8.0), dtype=np.float32)
        self._state_bias = np.array((0.0, 0.0, 0.0), dtype=np.float32)
    #def


    def reset(self,):
        self._env_state = self._env.reset()
        self._steps=0

        return self.observe()
    #def


    def step(self, action):
        (state_dash_, reward_, done_, _) = self._env.step(np.array(action).reshape(1))

        self._env_state = state_dash_
        self._steps +=1

        #scaling reward
        reward =  self._reward_scaling*reward_

        state_dash = self.observe()

        done = done_

        info = {}
        if done:
            info["is_terminal"] = False

        return (state_dash, reward, done, info)
    #def


    def observe(self, normalize=True):
        state_ = self._env_state.astype(np.float32)

        if normalize:
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
    def steps(self):
        return self._steps
    #def

    @property
    def max_steps(self):
        return self._max_steps
    #def
#class
