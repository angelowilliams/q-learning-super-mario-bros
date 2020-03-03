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
