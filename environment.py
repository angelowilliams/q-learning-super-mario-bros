import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY


"""
The Environment class controls data flow to and from the Super Mario Bros game
"""
class Environment():
    def __init__(self, input_mode=RIGHT_ONLY, level_mode=0):
        # inputMode = RIGHT_ONLY, SIMPLE_MOVEMENT, or COMPLEX_MOVEMENT
        # levelMode = SuperMarioBros-vX
        self.env = gym_super_mario_bros.make(f"SuperMarioBros-v{level_mode}")
        self.env = JoypadSpace(self.env, input_mode)
        self.env.reset()

    
    # Performs action and returns result
    def input_action(self, action):
        return self.env.step(action)

    # Renders the environment
    def render(self):
        self.env.render()

    # Resets the environment
    def reset(self):
        self.env.reset()
   
    # Closes the environment
    def close(self):
        self.env.close()

