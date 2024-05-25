from ANET import ANET
import numpy as np
from itertools import combinations
from hex_engine import Hex_Engine
from utils import *
from constants import *


class TOPP:

    def __init__(self, board_size: int, model_paths: str, num_games: int):
        self.board_size = board_size
        self.anets = [self.init_anet(path) for path in model_paths]
        self.num_games = num_games

    def init_anet(self, path: str) -> ANET:
        anet = ANET(epochs=None)
        anet.load_model(path)
        return anet

    def run_tournament(self, print_games=False) -> np.ndarray:
        # create pairs of matches between all the players
        match_pairs = combinations(range(len(self.anets)), 2)
        score_table = np.zeros_like(self.anets)

        hex_engine = Hex_Engine(self.board_size)
        match = 0

        for p1, p2 in match_pairs:
            p1_score, p2_score = 0, 0

            def p1_move(state, **_):
                legal_actions = hex_engine.get_legal_actions(state)
                anet_action = self.anets[p1].get_action(
                    state, legal_actions, use_stochastic_moves=False)
                return anet_action, None

            def p2_move(state, **_):
                legal_actions = hex_engine.get_legal_actions(state)
                anet_action = self.anets[p2].get_action(
                    state, legal_actions, use_stochastic_moves=False)
                return anet_action, None

            for i in range(self.num_games):
                # alternate who goes first
                if i % 2 == 0:
                    winner, state_history = play_game(hex_engine, p1_move,
                                                      p2_move)
                else:
                    winner, state_history = play_game(hex_engine, p2_move,
                                                      p1_move)

                if print_games:
                    for index, state in enumerate(state_history):
                        pretty_print_state(
                            state,
                            f'images/topp/match{match}/game_{i}/move_{index}.png'
                        )

                # Score based on who won, given alternation of who goes first
                if i % 2 == 0:
                    if winner == 1:
                        p1_score += 1
                    elif winner == 2:
                        p2_score += 1
                    else:
                        raise Exception(
                            'PLAYER 1 OR PLAYER 2 SHOULD HAVE WON!')
                else:
                    if winner == 1:
                        p2_score += 1
                    elif winner == 2:
                        p1_score += 1
                    else:
                        raise Exception(
                            'PLAYER 1 OR PLAYER 2 SHOULD HAVE WON!')

            score_table[p1] += p1_score
            score_table[p2] += p2_score
            match += 1

        return score_table


if __name__ == '__main__':
    model_paths = MODEL_PATHS
    topp = TOPP(
        BOARD_SIZE,
        model_paths,
        NUM_TOPP_GAMES,
    )
    scores = topp.run_tournament(print_games=VISUALIZE)
    for model, score in zip(model_paths, scores):
        print(f'{model}: {score}')
