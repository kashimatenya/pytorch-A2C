import numpy as np

#memory for experiences 
#experiences are set of series of (State, action, reward) and accompanying information
class SequentialMemory:
    def __init__(self, size, transform=None):
        self._State = None
        self._Action = np.empty((size,), dtype=np.float32)
        self._Reward = np.empty((size,), dtype=np.float32)

        self._count = 0
        self._max_len = size

        self._end_state = None
        self._is_terminal = None

        self._transform = transform
    #def


    def append(self, state, action, reward):
        if self._count == self._max_len:
            raise Exception("memory overflow")

        assert not self.is_end, "append after termination"

        if self._State is None:
            self._State = np.empty( (self._max_len,)+state.shape, dtype=state.dtype)
        
        self._State[self._count] = state
        self._Action[self._count] = action
        self._Reward[self._count] = reward

        self._count +=1
        return
    #def


    def end_sequence(self, end_sate, is_terminal):
        if not self.is_end:
            self._end_state = end_sate
            self._is_terminal = is_terminal

        return
    #def


    def refer(self):
        assert self.is_end, "refer without termination"

        count = self._count

        State = self._State[:count]
        Action = self._Action[:count]
        Reward = self._Reward[:count]

        end_state = self._end_state
        is_terminal = self._is_terminal

        if self._transform is None:
            experiences = (State, Action, Reward, end_state, is_terminal)
        else:    
            experiences = self._transform(State, Action, Reward, end_state, is_terminal)
            
        return experiences
    #def


    def clear(self):
        self._count = 0
        self._state_dash = None
        self._is_terminal = None
        return
    #def


    @property
    def max_len(self):
        return self._max_len
    #def


    @property
    def is_end(self):
        return self._is_terminal is not None
    #def


    def __len__(self):
        return self._count
    #def
#class
