import pickle
import random


class QTable():
    def __init__(self, numActions=7, learningRate=0.1, 
                 discountRate=0.5, epsilon=0.1, pickleFile=False):
        if pickleFile:
            # Load File
            data = pickle.load(open(pickleFile, "rb"))
            self.table = data[0]
            numActions = data[1]
            learningRate = data[2]
            discountRate = data[3]
            epsilon = data[4]
        else:
            # Create Q-Table
            self.table = {}

        self.numActions = numActions
        self.learningRate = learningRate
        self.discountRate = discountRate
        self.epsilon = epsilon
        self.actionMap = [0, 1, 2, 3, 4, 5, 6]

    
    def getAction(self, state):
        if state in self.table:
            actionList = self.table[state]
        else:
            actionList = [0] * self.numActions
            self.table[state] = actionList

        rand = random.random()
        if rand < self.epsilon:
            action = random.choice(range(self.numActions))
        else:
            action = actionList.index(max(actionList))

        return action


    def updateTable(self, lastState, lastAction, reward, state):
        lastActionIndex = self.actionMap.index(lastAction)
        value = self.table[lastState][lastActionIndex]
       
        bestPossibleReward = self.discountRate * max(self.table[state])
        value += self.learningRate * (reward + bestPossibleReward - value)
        
        self.table[lastState][lastActionIndex] = value


    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setDiscountRate(self, discountRate):
        self.discountRate = discountRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def saveTable(self, pickleFile):
        data = [self.table, self.numActions, 
                self.learningRate, self.discountRate,
                self.epsilon]

        pickle.dump(data, open(pickleFile, "wb"))
     

def get_action(state, reward, info):
    """
    get_action returns the chosen action for any given state
    params: state: an array representing the current state
            reward: the reward for the current state
            info: additional information about the current state
    return: an integer representing the action to take in this state

    Possible actions:
        0: ['NOOP'],
        1: ['right'],
        2: ['right', 'A'],
        3: ['right', 'B'],
        4: ['right', 'A', 'B'],
        5: ['A'],
        6: ['left']

    Currently this funciton is just a placeholder, so it always returns 1
    """

    return 1
