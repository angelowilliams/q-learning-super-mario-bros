import random
import numpy as np
import sys

from network import DQNetwork
from memory import Memory
from states import process_state, Transition, StackedTransition
from environment import Environment


# The Agent class contains all other objects and is the main object in
#  charge of selecting actions and playing the game. All input and output
#  with the environment goes through an Agent object
class Agent():
    def __init__(self):
        self.network = DQNetwork() 
        # Increase max_entries to the maximum your system can handle to improve
        #  training
        self.memory = Memory(max_entries=30000)
        self.env = Environment(level_mode=0)
        self.num_actions = 5
        # Keeps track of the last four frames, used to make stacked transitions
        self.buffer = []
        self.epsilon = 0.9999975
        self.scaling_factor = self.epsilon
        self.minEpsilon = 0.1
        self.gamma = 0.9
        
        self.step_count = 1

        # The live network is copied over to the target network every copy_count
        #  frames. Also controls how often the model's performance is output into
        #  the terminal and how often the model is saved
        self.copy_count = 25000

        # Blurin plays copy_count frames before starting to train the model
        self.blurin = True

        # The model is trained on train_batch transitions every train_count frames
        self.train_count = 100
        self.train_batch = 25
        
        # Performs an empty input to get an initial state
        oldState, _, _, _, _ = self.perform_action(0)
        oldState = process_state(oldState)
        state, action, reward, done, info = self.perform_action(0)
        self.transition = Transition(oldState, action, reward, state, done)

        # The info log is used to determine when Mario gets stuck on a pipe
        self.info_log = [info]

        # The following are used to record the model's performance
        self.best_level = 0
        self.best_x_pos = 0
        self.total_reward_in_training_cycle = 0
        self.reward_list = []
        self.output_file = open('results.txt', 'w+')


    """
    Checks if Mario has been at the same x-position for 60 frames (1 second)
    Return True if so
    """
    def check_if_stuck(self):
        # We must generate a random number. In some cases, if we constantly say
        #   Mario is stuck, then the jump button will be held in, preventing Mario
        #   from jumping. Randomly returning False lets go of the jump button
        rand = random.random()
        if rand > 0.03:
            return self.info_log[0]["x_pos"] == self.info_log[-1]["x_pos"]
        else:
            return False


    """ 
    Chooses either a random action based on epsilon or chooses the 
    best action according to the live network. Returns the chosen action
    """
    def choose_action(self, stacked_transition):
        # Scales down epsilon every step
        if self.epsilon > self.minEpsilon:
            self.epsilon *= self.scaling_factor
       
        # If the model is stuck on a pipe, then return ['A', 'B', 'right']
        if self.check_if_stuck():
            return 4

        rand = random.random()
        if rand < self.epsilon:
            # Random action
            action = random.randint(0, self.num_actions - 1)
        else:
            # Best action
            action = np.argmax(self.network.predict(stacked_transition.states))

        return action


    """
    Performs an input action (integer 0 to 4) and returns:
        state: the resulting state
        action: the action performed (returned to be consistent with other functions)
        reward: the reward given for performing the input action
        done: a boolean representing if Mario is out of lives
        info: a dictionary giving additional information about the agent
    """
    def perform_action(self, action):
        state, reward, done, info = self.env.input_action(action)

        return state, action, reward, done, info


    """
    Performs a random action and returns the same as perform_action()
    """
    def perform_random_action(self):
        rand = random.randint(0, self.num_actions - 1)

        # Checks if Mario is stuck on a pipe
        if self.check_if_stuck():
            rand = 4

        state, reward, done, info = self.env.input_action(rand)

        return state, rand, reward, done, info

    
    """
    Performs one step of the game
    This entails the following major steps:
        1. Performing an action
        2. If it has been four frames since the last stack of frames was
            added to the buffer, stacks the last 4 and adds them to the buffer
        3. If it is time to copy the live network to the target network, does so
        4. If it is time to train the live network, does so
    """
    def step(self):
        # Performs action
        if self.blurin:
            state, action, reward, done, info = self.perform_random_action()
        else:
            stacked_transition = StackedTransition(self.transition, self.buffer[1:4])
            action = self.choose_action(stacked_transition)
            state, action, reward, done, info = self.perform_action(action)
        self.transition = Transition(self.transition.state, action, reward, state, done)
        
        self.total_reward_in_training_cycle += reward
       
        # Adds info dict to the log and trims the length down to 60
        self.info_log.append(info)
        if len(self.info_log) > 60:
            del self.info_log[0]

        # Resets the environment if we get a 'Game Over'
        if self.transition.done:
            self.env.reset()

        # Create stacked state and adds to memory
        if (self.step_count % 4 == 0) and (self.step_count > 20):
            stacked_transition = StackedTransition(self.transition, self.buffer[1:4])
            self.memory.append_transition(stacked_transition)
           
            # Keeps track of the farthest x position stats
            if info['x_pos'] > self.best_x_pos and info['stage'] >= self.best_level:
                self.best_level = info['stage']
                self.best_x_pos = info['x_pos']

        self.buffer.append(self.transition)
        if len(self.buffer) > 4:
            del self.buffer[0]

        # Copies weights from live model to target model and saves the model
        if self.step_count % self.copy_count == 0:
            self.network.copy_to_target() 
            self.network.save_weights()
            
            if self.blurin:
                self.blurin = False

            self.reward_list.append(self.total_reward_in_training_cycle)
            print(f" best_level = {self.best_level}, best x = {self.best_x_pos}, total rewards = {self.total_reward_in_training_cycle}")
            self.output_file.write(f"{self.best_level}-{self.best_x_pos}-{self.total_reward_in_training_cycle}\n")
            self.total_reward_in_training_cycle = 0
            self.best_level = 1
            self.best_x_pos = 0
 
        # Trains network
        elif (self.step_count % self.train_count == 0) and (not self.blurin):
            # Gets a random batch of stacked transitions
            training_batch = self.memory.query_memory(self.train_batch)
            state_list = []
            q_true = []

            for stacked_transition in training_batch:
                # Prepares the data to be fed into the neural network
                state_list.append(stacked_transition.states)
                q_list = self.network.predict(stacked_transition.states)[0]
                q_val = stacked_transition.reward
                q_val += self.gamma * max(self.network.predict(stacked_transition.states,
                                            use_target_model=True)[0])
                q_list[stacked_transition.action] = q_val
                q_true.append(q_list)
             
            state_list = np.stack(state_list, axis=1)[0]

            # Trains the network
            self.network.train(state_list, np.array(q_true))
        
        # !!!
        # Comment the following line to stop rendering the game. This will
        #   result in a substantial decrease in the amount of time spent per
        #   training cycle
        # !!!
        self.env.render()

        self.step_count += 1

