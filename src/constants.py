import tensorflow as tf

"""
------------------------ General consts ------------------------
"""
BOARD_SIZE = 4
BOARD_SHAPE = (BOARD_SIZE, BOARD_SIZE)
"""
------------------------ Actor related consts ------------------------
"""

EPOCHS = 3
RBUF_LIMIT = 1500
EXPLORATION_CONST = 1.5
NUM_GAMES = 1000
NUM_SIMULATIONS = 10000
SAVE_INTERVAL = 50
MINIBATCH_SIZE = 150
FILEPATH = "models"
VISUALIZE_ACTOR = False
"""
------------------------ TOPP related consts ------------------------
"""

MODEL_PATHS = [
    "4x4_models/model_10",
    "4x4_models/model_50",
    "4x4_models/model_100",
    "4x4_models/model_500",
    "4x4_models/model_1000",
]
NUM_TOPP_GAMES = 10
VISUALIZE = False
"""
------------------------ ANET related consts ------------------------
"""

FILTERS = [64, 128, 128]
KERNEL_SIZES = [(5, 5), (3, 3), (2, 2)]
DENSE_UNITS = [64, 128, 128]
ACTIVATION = "relu"
LEARNING_RATE = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=300, decay_rate=0.9
)
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
PADDING = "same"
LOSS = "categorical_crossentropy"
"""
------------------------ MCTS related consts ------------------------
"""

TIME = 2.5
