#wappe some enviroments as if one enviroments
class MultipleEnvironments:
    def __init__(self, environemtns):
        self._environments = environemtns
    #def

    def reset(self):
        State = []
        for enviroment in self._environments:
            state = enviroment.reset()
            State.append(state)        

        return State
    #def


    def step(self, Action):
        State_dash = []
        Reward = []
        Done = []
        Info = []

        for enviroment, action  in zip(self._environments, Action):
            (state_dash, reward, done, info) = enviroment.step(action)

            if done:
                info["end_state"] = state_dash
                state_dash = enviroment.reset()

            State_dash.append(state_dash)
            Reward.append(reward)
            Done.append(done)
            Info.append(info)

        return (State_dash, Reward, Done, Info)
    #def


    def observe(self, normalize=True):
        State = []
        for enviroment in self._environments:
            state = enviroment.observe(normalize)
            State.append(state) 

        return State
    #def


    def close(self):
        for enviroment in self._environments:
            enviroment.close()

        return
    #def

    def __len__(self):
        return len(self._environments)
    #def
    
#class