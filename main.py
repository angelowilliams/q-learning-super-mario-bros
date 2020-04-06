from load_environment import load_environment
from q_learning import QTable
import numpy as np


def main():
    # Loads Super Mario Bros. Gym environment
    env = load_environment()
    table = QTable()

    done = True
    for step in range(5000):
        # For the first step
        if done:
            env.reset()
            state, reward, done, info = env.step(0)
            state = state.tostring()
        
        # Gets an action by Q-learning
        action = table.getAction(state)
        
        # Stores the current state
        lastState = state

        # Performs selected action
        state, reward, done, info = env.step(action)
        state = state.tostring()

        # Updates the table
        table.updateTable(lastState, action, reward, state)

        env.render()

    env.close()


if __name__ == '__main__':
    main()
