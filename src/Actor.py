import tensorflow as tf
from ANET import ANET
from MCTS import MCTS
from collections import deque
import pickle
from utils import *
from Node import Node
import numpy as np
from hex_engine import Hex_Engine, switch_player
from constants import *


class Actor:
    """
    Actor for training the ANET
    """

    def __init__(self,
                 hex_engine: Hex_Engine,
                 anet: ANET,
                 rbuf_lim: int = 100,
                 exploration_const: float = 1.5):
        self.hex_engine = hex_engine
        self.anet = anet
        self.rbuf_lim = rbuf_lim
        self.replay_buffer = deque([], rbuf_lim)
        self.anet_parameters = []
        self.exploration_const = exploration_const

    def train(self, n_games: int, num_simulations: int,
              utility_function: callable, save_interval: int,
              minibatch_size: int, filepath: str) -> None:
        """
        Train the ANET by simulating games and periodically saving the model and replay buffer.

        Args:
            n_games (int): Number of games to simulate.
            num_simulations (int): Number of simulations per move.
            utility_function (Callable): Utility function for MCTS.
            save_interval (int): Interval at which to save the model and replay buffer.
            minibatch_size (int): Size of the minibatch for training.
            filepath (str): Path for saving the models and replay buffers.
        """

        # Save initial model
        model_path = f"{filepath}/initial_model"
        self.anet.save_model(model_path)

        # Loop through the specified number of games
        for i in range(1, n_games + 1):
            print(f"Game {i}")
            Node.clear_memo()

            # Play a game and add the resulting data to the replay buffer
            self.simulate_game(utility_function=utility_function,
                               num_simulations=num_simulations,
                               minibatch_size=minibatch_size,
                               game_num=i)

            # Save the model and replay buffer periodically
            if i % save_interval == 0:
                model_path = f"{filepath}/model_{i}"
                self.anet.save_model(model_path)

        # Save the final model after all games are played
        model_path = f"{filepath}/final_model"
        self.anet.save_model(model_path)

        print("Training successfully finished!")

    def simulate_game(self, utility_function, num_simulations: int,
                      minibatch_size: int, game_num: int) -> None:
        """
        Simulate a game, updating the replay buffer and training the ANET.

        Args:
            utility_function (Callable): Utility function for MCTS.
            num_simulations (int): Number of simulations per move.
            minibatch_size (int): Size of the minibatch for training.
        """
        # Initialize the game state
        board = self.hex_engine.get_start_board()
        player = 1
        state = (board, player)
        root_node = None
        winner = None
        dir_path = f'images/actor/game{game_num}'
        move = 0

        # Loop until the game is over or there are no legal moves left
        while len(self.hex_engine.get_legal_actions(
                state)) > 0 and winner is None:
            # Use MCTS to get the best move
            mcts = MCTS(hex_engine=self.hex_engine,
                        num_simulations=num_simulations,
                        default_policy=self.anet,
                        utility_function=utility_function,
                        state=state,
                        root_node=root_node,
                        exploration_const=self.exploration_const)

            best_move, prob_distribution, child_node = mcts.get_best_move()

            # Add to replay_buffer if buffer is smaller than limit, else remove oldest elements
            self.replay_buffer.append((state, prob_distribution))

            # Get the child state and check if it's a winning state
            child_state = self.hex_engine.get_child_board(state, best_move)
            if self.hex_engine.check_if_winning(child_state):
                _, next_player = child_state
                winner = switch_player(next_player)

            state = child_state
            root_node = child_node

            if VISUALIZE_ACTOR:
                pretty_print_state(state, f'{dir_path}/move_{move}.png')
            move += 1

        print(f"Winner: {winner}")

        # Train ANET on a random minibatch of cases from the replay buffer
        minibatch_size = min(minibatch_size, len(self.replay_buffer))
        minibatch = random.sample(self.replay_buffer, minibatch_size)
        self.anet.train(minibatch)


if __name__ == "__main__":
    hex_engine = Hex_Engine(BOARD_SIZE)
    anet = ANET(epochs=EPOCHS)
    anet.initialize(board_shape=(BOARD_SIZE, BOARD_SIZE))

    actor = Actor(hex_engine,
                  anet,
                  rbuf_lim=RBUF_LIMIT,
                  exploration_const=EXPLORATION_CONST)
    actor.train(n_games=NUM_GAMES,
                num_simulations=NUM_SIMULATIONS,
                utility_function=upper_confidence_trees,
                save_interval=SAVE_INTERVAL,
                minibatch_size=MINIBATCH_SIZE,
                filepath=FILEPATH)
