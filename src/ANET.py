import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, Add, Input, BatchNormalization
from constants import *
import random
"""
One-hot encode the board before entering it into the NN

Create a convolutionary 2D network with 2d input;
    1. One Hot Encoded board for player 1
    2. One Hot Encoded board for player 2
"""


class Legal_Move_Layer(tf.keras.layers.Layer):

    def __init__(self, a=0.5, **kwargs):
        super().__init__(**kwargs)
        self.a = a

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        a, x = inputs
        board_size = a.shape[1]
        # reshape x to board like tensor, eg. 4,4,1
        x = tf.reshape(x, (-1, board_size, board_size, 1))

        a_zero_mask = tf.math.reduce_all(a == 0, axis=-1, keepdims=True)
        result = tf.where(a_zero_mask, x, tf.zeros_like(x))
        result = tf.reshape(result, (-1, board_size**2))
        return result

    def get_config(self):
        config = super().get_config()
        config.update({'a': self.a})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class ANET:

    def __init__(self, epochs):
        self.model = Sequential()
        self.epochs = epochs
        pass

    def initialize(
            self,
            board_shape: tuple[int, int],
            filters: list[int] = FILTERS,
            kernel_sizes: list[tuple[int, int]] = KERNEL_SIZES,
            fully_connected_layers: list[int] = DENSE_UNITS,
            activation: str = ACTIVATION,
            optimizer: tf.keras.optimizers.Optimizer = OPTIMIZER,
            padding: str = PADDING,  # input size = output size
            lossfunction: str = LOSS):
        """
        board_shape: (nRows, nCols)
        filters: Conv2D filters
        kernel_sizes: Conv2D kernel sizes
        fully_connected_layers: number of nodes in each dense layer
        activation_function: string of Keras activation function
        optimizer: string of Keras optimizer
        padding: string of Keras padding, uncertain about what sending an int would do
        lossfunction string identifier of built-in Keras loss function
        """

        # Add convolutional layers
        input_shape = (board_shape[0], board_shape[1], 2)
        inputs = Input(shape=input_shape)
        x = inputs

        # Add convolutional layers with residual connections and batch normalization
        for index, num_filters in enumerate(filters):
            # Convolutional layer
            x = Conv2D(num_filters,
                        kernel_size=kernel_sizes[index],
                        padding=padding)(x)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)

            # Another convolutional layer
            # x1 = Conv2D(num_filters,
            #             kernel_size=kernel_sizes[index],
            #             padding=padding)(x1)
            # x1 = BatchNormalization()(x1)

            # # Residual connection
            # # if index > 0 and filters[index] != filters[index - 1]:  # change in the number of filters
            # x = Conv2D(num_filters, kernel_size=(1, 1), padding=padding)(x)
            # x1 = Add()([x1, x])
            # x = Activation(activation)(x1)

        x = Flatten()(x)

        # Add dense layers with batch normalization
        for _, layer_size in enumerate(fully_connected_layers):
            x = Dense(layer_size)(x)
            x = BatchNormalization()(x)
            x = Activation(activation)(x)

        # Add output layer
        outputs = Dense(board_shape[0] * board_shape[1])(x)

        # Remove illegal moves from the output
        outputs = Legal_Move_Layer()([inputs, outputs])

        outputs = Activation('softmax')(outputs)

        self.model = Model(inputs=inputs, outputs=outputs)

        self.model.compile(loss=lossfunction,
                           optimizer=optimizer,
                           metrics=['accuracy'])

    def get_action(
            self,
            state: tuple[np.ndarray, int],
            legal_actions: list[tuple[tuple[int, int], int]],
            use_stochastic_moves: bool = False) -> tuple[tuple[int, int], int]:
        '''
        state: tuple(board, player)
        legal_actions: list of legal actions
        use_stochastic_moves: boolean, if true, play based on probability distribution, not always the highest
        '''
        board, player = state

        ohe_boards = np.array([self.one_hot_encode(state)])
        stack = tf.stack(ohe_boards)
        prediction = self.model(stack).numpy()
        prediction = self._decode(prediction.reshape(board.shape), player)

        # Play based on probability distribution, not always the highest
        if use_stochastic_moves:
            prediction = {a: prediction[a[0]] for a in legal_actions}
            action = (random.choices(population=list(prediction.keys()),
                                     weights=list(prediction.values()),
                                     k=1)[0])
        # Play the move with the highest probability
        else:
            max_pred = 0
            move = None
            for a in legal_actions:
                if prediction[a[0]] > max_pred:
                    max_pred = prediction[a[0]]
                    move = a[0]
            action = (move, player)

        return action

    def train(
            self, minibatch: list[tuple[tuple[np.ndarray, int],
                                        np.ndarray]]) -> None:
        """
        minibatch: list of (state, target) tuples for training the neural network
        """
        # Extract the states from the minibatch and one-hot encode them
        X = np.stack([self.one_hot_encode(state) for state, _ in minibatch])

        # Extract the targets from the minibatch, decode them, and flatten them into a 1D arra
        y = np.stack([
            self._decode(target, state[1]).flatten()
            for state, target in minibatch
        ])

        # Fit the neural network to the training data
        self.model.fit(X,
                       y,
                       batch_size=len(minibatch),
                       epochs=self.epochs,
                       verbose=1)

    """
    ------------------------ Model loading and saving ------------------------
    """

    def save_model(self, filepath: str) -> None:
        self.model.save(filepath=filepath)

    def load_model(self, filepath: str) -> None:
        '''
        load model with weights from file
        filepath: path to the model
        '''
        self.model = keras.models.load_model(filepath=filepath)

    def set_model(self, model: keras.models.Model) -> None:
        '''
        set model to a given model
        model: keras model
        '''
        self.model = model

    """
    ------------------------ Helper functions ------------------------
    """

    def _decode(self, prediction: np.ndarray, player: int) -> np.ndarray:
        """
        Decode the neural network output to match the board perspective of the player
        :param prediction: The output of the neural network
        :param player: Which player is making the prediction
        :return: The decoded prediction
        """
        if player == 1:
            return prediction
        elif player == 2:
            return prediction.T
        else:
            raise ValueError("Invalid player value. Must be 1 or 2.")

    def one_hot_encode(self, state: tuple[np.ndarray, int]) -> np.ndarray:
        """
        :param state on format tuple(game_board, player)
        :return one-hot-encoded state on the form [[bin 2D-array of the current players pegs],
        [bin 2D-array of the other players pegs]]
 
        """
        board, player = state
        player_1_perspective = (board == 1).astype(int)
        player_2_perspective = (board == 2).astype(int)

        encoded = np.zeros(shape=(board.shape[0], board.shape[1], 2))
        if player == 1:
            encoded = np.stack((player_1_perspective, player_2_perspective),
                               axis=-1,
                               dtype=float)
        else:
            encoded = np.stack(
                (player_2_perspective.T, player_1_perspective.T),
                axis=-1,
                dtype=float)
        return encoded
