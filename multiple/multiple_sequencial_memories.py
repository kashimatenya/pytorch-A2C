#wappe some objects of SequencaialMemories as if one obuject
class MultipleSequentialMemories:
    def __init__(self, memories):
        self._memories = memories
    #def

    def append(self, State, Action, Reward, Done, Info):
        memories = self._memories
        for memory, state, action, reward, done, info in zip(memories, State, Action, Reward, Done, Info):
            memory.append(state, action, reward)
        
            if done:
                end_state = info["end_state"]
                is_terminal = info["is_terminal"]
                memory.end_sequence(end_state, is_terminal)

        return
    #def


    def end_sequence(self, EndState):
        memories = self._memories
        for memory, end_state in zip(memories, EndState):
            memory.end_sequence(end_state, False)

        return
    #def


    def refer(self):
        memories = self._memories
        experiences = []
        for memory in memories:
            experiences.extend(memory.refer())        

        return experiences
    #def


    def clear(self):
        memories = self._memories
        for memory in memories:
            memory.clear()

        return
    #def


    def __len__(self):
        memories = self._memories
        return sum(map(len, memories))
    #def
#class
