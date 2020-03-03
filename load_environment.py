import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def load_environment():
    """
    load_environment prepares a Super Mario Bros. Gym environment
    params: none
    return: a ready instance of the gym environment
    """

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
   
    return env



