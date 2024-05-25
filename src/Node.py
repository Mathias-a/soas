from collections import defaultdict
from typing import Dict, Tuple
import threading
from hex_engine import Hex_Engine


class Node:
    """
    MCTS Node class
    
    Hashing nodes to static dictionary memo, in order to reuse them at later states
    """
    memo: Dict[bytes, 'Node'] = defaultdict(lambda: None)

    @staticmethod
    def clear_memo():
        Node.memo.clear()

    def __init__(self, state: Tuple):
        self.state = state
        self.lock = threading.Lock()

        # children = {action: Node(child_state)}
        # action = ((row, col), player)
        self.children: dict[Tuple[Tuple[int, int], int], 'Node'] = {}
        # edges = {action: (value, visit_count)}
        self.edges: dict[Tuple[Tuple[int, int], int], Tuple[int, int]] = {}
        self.visit_count = 1

    def get_child(self, action: Tuple[Tuple[int, int], int]) -> 'Node':
        return self.children[action]

    def set_child_states(self, actions: list[Tuple[Tuple[int, int], int]],
                         state_manager: Hex_Engine) -> None:
        """
        Used when expanding nodes.
        If any child nodes have been generated at an earlier stage in the game, they are reloaded
        """
        for action in actions:
            child_state = state_manager.get_child_board(self.state, action)
            hash_ = hash(child_state[0].tobytes())
            if Node.memo[hash_]:
                self.children[action] = Node.memo[hash_]
            else:
                node = Node(child_state)
                Node.memo[hash_] = node
                self.children[action] = node

            # Set edge_visits initially to 1 to avoid division by zero
            self.edges[action] = self.edges.get(action, (0, 1))

    def is_leaf(self) -> bool:
        return not self.children

    def visit(self, action: Tuple[Tuple[int, int], int], value: int) -> None:
        with self.lock:
            self.visit_count += 1
            edge_value, edge_visits = self.edges.get(action, (0, 1))
            self.edges[action] = (edge_value + value, edge_visits + 1)

    def get_value(self, action: Tuple[Tuple[int, int], int]) -> float:
        edge_value, edge_visits = self.edges[action]
        return (1 / edge_visits) * edge_value
