import random


"""
The Memory class stores a list of past transitions, adds entries,
    and returns queries to the list
"""
class Memory():
    def __init__(self, max_entries=50000):
        self.memory = []
        self.max_entries = max_entries
        self.entry_index = 0


    # Appends a transition to the end of memory
    def append_transition(self, transition):
        if len(self.memory) < self.max_entries:
            self.memory.append(transition)
        else:
            # This causes the memory to loop back around
            self.memory[self.entry_index] = transition
            self.entry_index += 1

            if self.entry_index >= self.max_entries:
                self.entry_index = 0


    # Gets n random transitions from the memory
    def query_memory(self, n):
        transition_list = []
        for i in range(n):
            rand = random.randint(0, len(self.memory) - 1)
            transition_list.append(self.memory[rand])

        return transition_list
    
