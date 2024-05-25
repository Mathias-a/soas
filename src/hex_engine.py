from typing import List, Tuple
import numpy as np


class Hex_Engine:
    '''
    Engine for the game of Hex.
    
    '''

    def __init__(self, board_size: int):
        self.board_size = board_size

    def get_start_board(self) -> np.ndarray:
        """
        Returns the initial empty board.
        """
        return np.zeros(shape=(self.board_size, self.board_size), dtype=int)

    def get_legal_actions(
            self, state: Tuple[np.ndarray,
                               int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Returns a list of legal moves for the current player.
        Todo: Remove player, I dont think it is needed.
        """
        board, player = state
        return [(move, player) for move in zip(*np.where(board == 0))]

    def get_score(self, state: Tuple[np.ndarray, int]) -> float:
        '''
        Returns the score of the game.
        NOTE: Should only be called when the game is over.
        '''
        if self.check_if_winning(state):
            return 1
        else:
            raise Exception('Game not scored')

    def get_child_board(
            self, state: Tuple[np.ndarray, int],
            action: Tuple[Tuple[int, int], int]) -> Tuple[np.ndarray, int]:
        """
        action = ((row, col), player)
        Returns the child state that is reached when applying action to state.
        """
        board, _ = state
        child_board = np.copy(board)
        (row, col), player = action
        child_board[row][col] = player
        return (child_board, switch_player(player))

    def check_if_winning(self, state):
        board, next_player = state
        player = switch_player(next_player)

        if player == 1:
            start_nodes = [(0, i) for i, x in enumerate(board[0])
                           if x == player]
        elif player == 2:
            start_nodes = [(i, 0) for i, x in enumerate(board[:, 0])
                           if x == player]
        else:
            raise Exception("player must be either 1 or 2")

        for start_node in start_nodes:
            if self._find_path(board, start_node, [start_node], player):
                return True

        return False

    def _find_path(self, board, node, path, player):
        if node[0] == self.board_size - 1 and player == 1:
            return True
        if node[1] == self.board_size - 1 and player == 2:
            return True

        for neighbor in self._get_neighbors(node):
            if neighbor not in path and board[neighbor] == player:
                path.append(neighbor)
                if self._find_path(board, neighbor, path, player):
                    return True
        return False

    def _get_neighbors(self, node):
        row, col = node
        neighbors = [
            (row + dr, col + dc)
            for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, 1), (1, -1)]
        ]
        return [(r, c) for r, c in neighbors
                if 0 <= r < self.board_size and 0 <= c < self.board_size]


def switch_player(player):
    return 3 - player
