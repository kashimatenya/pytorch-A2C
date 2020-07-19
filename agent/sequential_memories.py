from agent.sequential_memory import SequentialMemory

#set of SequencaialMemory
class SequentialMemories:
    def __init__(self, shape, transfrom = None):
        (max_count, size) = shape

        self._memories = [ SequentialMemory(size, transfrom) for _ in range(max_count) ]
        self._count = 1
    #def


    def append(self, state, action, reward):
        memory = self._memories[self._count-1]

        if memory.is_end:
            if self._count == len(self._memories):
                raise Exception("memory overflow")

            self._count +=1
            memory = self._memories[self._count-1]
        #if memory.is_end:

        memory.append(state, action, reward)
        return
    #def


    def end_sequence(self, end_state, is_terminal):
        memory = self._memories[self._count-1]
        memory.end_sequence(end_state, is_terminal)
        return
    #def


    def refer(self):
        memories = self._memories[:self._count]
        return list(map(lambda x:x.refer(), memories))
    #def


    def clear(self):
        memories = self._memories[:self._count]
        for memory in memories:
            memory.clear()

        self._count = 1
        return
    #def

    @property
    def is_end(self):
        memory = self._memories[self._count-1]
        return memory.is_end
    #def


    def __len__(self):
        memories = self._memories[:self._count]
        return sum(map(len, memories))
    #def
#class
