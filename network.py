import keras


"""
DQNetwork class controls input/output to the model
The paratermers are:
    learning_rate - learning rate for the Adam optimizer
    input_dim - size (n) of nxn input image. Must be square
    output_dim - number of actions
    frame_stack - number of frames per stacked transition
"""
class DQNetwork():
    def __init__(self, learning_rate=0.1, input_dim=84, output_dim=5, frame_stack=4):
        self.learning_rate = learning_rate

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.frame_stack = frame_stack
        
        # Creates the models and populates self.target_model and self.live_model
        self.initialize_model()

    
    def initialize_model(self):
        self.live_model = keras.models.Sequential()
        self.target_model = keras.models.Sequential()
        
        # Creates mirror networks
        for model in [self.live_model, self.target_model]:
            # 3 convolutional layers with 64 filters
            # Max pooling layers are not used because the position of the objects matter
            model.add(keras.layers.Conv2D(64, 3, padding="same", activation='relu', input_shape=(4, 84, 84)))
            model.add(keras.layers.Conv2D(64, 3, padding="same", activation='relu'))
            model.add(keras.layers.Conv2D(64, 3, padding="same", activation='relu'))

            model.add(keras.layers.Flatten())
            
            # Two fully-connected layers
            model.add(keras.layers.Dense(512, activation='relu'))
            model.add(keras.layers.Dense(5, activation='relu'))

            model.compile('adam', 'mse')


    # Copies the weights of the live network to the target network
    def copy_to_target(self):
        self.target_model.set_weights(self.live_model.get_weights())

    # Trains the model with parameters:
    #   state_list: a list of n stacked transitions 
    #   q_true: a list of n lists of q-values num_actions long
    def train(self, state_list, q_true):
        self.live_model.fit(state_list, q_true, verbose=0)

    # Makes a prediction
    # state - a stacked transition
    # use_target_model - if True, uses the target model instead of the live model
    # Outputs a list of Q-values for the given state
    def predict(self, state, use_target_model=False):
        if use_target_model:
            return self.target_model.predict(state)
        else:
            return self.live_model.predict(state)


    # Saves the weights of the model to ./model.h5
    def save_weights(self):
        keras.models.save_model(self.live_model, "model.h5")
        
