from load_environment import load_environment
import q_learning


def main():
    # Loads Super Mario Bros. Gym environment
    env = load_environment()
    
    done = True
    for step in range(5000):
        # For the first step
        if done:
            env.reset()
            state, reward, done, info = env.step(0)
           
        # Gets best action by Q-learning
        action = q_learning.get_action(state, reward, info)
        # Performs best action
        state, reward, done, info = env.step(action)
        
        env.render()

    env.close()


if __name__ == '__main__':
    main()
