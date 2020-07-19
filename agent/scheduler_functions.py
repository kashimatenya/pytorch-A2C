from math import pow, exp, log

class ExponentialDecayWithWarmUp:
    def __init__(self, period, lower, delay=0, warm_up_epochs=0.0, initial_ratio=1.0):
        assert warm_up_epochs<=delay, 'invalid combination of delay and warm_up_epochs'

        attenuation=lower
        decay = exp(log(attenuation)/period)

        self._decay = decay
        self._lower = lower
        self._delay = delay

        self._warm_up_epochs = warm_up_epochs

        if warm_up_epochs >0:
            self._warm_up_slope = (1-initial_ratio)/self._warm_up_epochs
        else:
            self._warm_up_slope = None
    #def
    
    def __call__(self, epoch):  
        #warm-up
        if epoch < self._warm_up_epochs:
            return 1.0 - self._warm_up_slope*(self._warm_up_epochs - epoch)

        
        if epoch < self._delay:
            return 1.0

        #decay learning rate in exponentially
        return  max(pow(self._decay, epoch - self._delay), self._lower)       
    #def
#class
