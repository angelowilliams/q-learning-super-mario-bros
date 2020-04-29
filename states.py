import cv2
import numpy as np


# process_state turns the frame grayscale, 84x84, and adds a channel
def process_state(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state, (84, 84))
    state = np.expand_dims(state, axis=2)

    return state

"""
The Transition class holds all the data needed after an action is performed
"""
class Transition():
    def __init__(self, state, action, reward, outcome, done):
        self.state = state
        self.reward = reward
        self.action = action
        self.outcome = process_state(outcome)
        self.done = done

"""
The StackedTransition class holds the same data as the Transition
    except that instead of 1 state, there are a list of transitions
"""
class StackedTransition():
    def __init__(self, transition, history):
        self.transition_list = [x for x in history]
        self.transition_list.append(transition)
        
        self.states = self.getNumpyArray()
        self.action = transition.action
        self.reward = transition.reward
        self.outcome = transition.outcome

    # Converts the transition_list to a numpy array that is compatible with keras
    def getNumpyArray(self):
        state_list = [x.state.reshape(1, 84, 84) for x in self.transition_list]
        
        states = np.stack(state_list, 1)
        
        return states


