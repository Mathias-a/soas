import time
from Node import Node
import numpy as np
from typing import Tuple
from hex_engine import Hex_Engine
from ANET import ANET
from constants import *


class MCTS:

    def __init__(self,
                 hex_engine: Hex_Engine,
                 num_simulations: int,
                 default_policy: ANET,
                 utility_function,
                 state: Tuple[np.ndarray, int],
                 root_node: Node = None,
                 exploration_const: float = 1.5):
        '''
        Args:
        hex_engine:                     Engine for Hex game
        num_simulations (int):          Number of simulations
        default_policy (ANET):          Default policy for rollout       default_policy(state, action)
        utility_function (callable):    Used for evaluating moves
        state (tuple(int, int), int):   Current game state (game_board, player)
        root_node (Node):               Root node when tree is pruned
        exploration_const (float):      Exploratory constant for utility function
        '''

        self.hex_engine = hex_engine
        self.num_simulations = num_simulations
        self.default_policy = default_policy
        self.utility_function = utility_function
        self.state = state
        self.root_node = root_node
        self.exploration_const = exploration_const

    def get_best_move(
            self
    ) -> Tuple[Tuple[int, int], dict[Tuple[int, int], float], Node]:
        """
        Runs the MCTS algorithm for a specified number of simulations and returns the best move to make,
        the probability distribution over all possible moves, and the child node associated with the best move.
        return: best_move, distribution, child_node
        """
        # get the root node of the search tree
        root_node = self.get_root_node()

        # Create timer to stop simulations after a certain amount of time
        start_time = time.time()
        max_time = TIME

        # Run simulations based on input
        for _ in range(self.num_simulations):

            # stop the loop if the elapsed time is greater than 3 seconds
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time:
                break

            # Tree Search
            node_action_list = self.traverse(root_node)

            # Set node to be expanded
            next_node = self.get_next_node(node_action_list, root_node)

            # Expand selected node and choose a child node using default policy
            node_action = self.expand(next_node)
            if node_action is not None:
                node_action_list.append(node_action)

            # Back-propagate the score from the leaf node up to the root
            score = self.get_score_from_node_action_list(node_action_list)
            self.back_propagate(node_action_list, score)

        # Return the best move and its associated child node, as well as the probability distribution over all possible moves
        best_move, child_node = self.select_best_move_and_child(root_node)
        prob_distribution = self._get_prob_distribution(root_node)
        return best_move, prob_distribution, child_node

    def get_root_node(self) -> Node:
        """
        Return the root node of the search tree. 
        If a root node has not been set, create a new node with the current game state.
        """
        if self.root_node is None:
            return Node(self.state)
        else:
            return self.root_node

    def get_next_node(self, node_action_list: list[Node, Tuple[int, int]],
                      root_node: Node) -> Node:
        """
        Given a list of node-action pairs and the root node of the search tree, 
        return the next node in the search tree to explore.
        """
        if len(node_action_list) > 0:
            last_node, last_action = node_action_list[-1]
            return last_node.children[last_action]
        else:
            return root_node

    def get_score_from_node_action_list(
            self, node_action_list: list[Node, Tuple[int, int]]) -> float:
        """
        Given a list of node-action pairs, 
        return the score associated with the last action in the list (i.e., the score of the game after making that move).
        """
        if len(node_action_list) > 0:
            return self.rollout(node_action_list[-1])
        else:
            return self.hex_engine.get_score(self.state)

    def select_best_move_and_child(
            self, root_node: Node) -> Tuple[Tuple[int, int], Node]:
        """
        Given the root node of the search tree, 
        return the best move to make and the child node associated with that move.
        """
        best_move = max(root_node.edges, key=(lambda k: root_node.edges[k][1]))
        child_node = root_node.children[best_move]
        return best_move, child_node

    def traverse(self, node: Node) -> list[Tuple[Node, Tuple]]:
        '''
        Traverse down the search tree based on the tree policy (i.e., a policy that balances exploration and exploitation) 
        until a leaf node is reached.
        Return a list of node-action pairs representing the path from the root node to the leaf node.
        '''
        node_action_list = []

        #Traversing down to the bottom of the existing tree
        while len(node.children) > 0:
            best_action = self._choose_action(node)
            node_action_list.append((node, best_action))
            node = node.get_child(best_action)

        return node_action_list

    def expand(self, node: Node) -> Tuple[Node, Tuple[int, int]]:
        '''
        node: root node of the traversal
        return: node, best_action

        Given a node in the search tree, generate its children 
        (moves from the current state) 
        return the child node associated with the best move according to the default policy.
        '''

        #No expansion if the node is an end state
        if self.hex_engine.check_if_winning(node.state):
            return

        #Expanding the node
        actions = self.hex_engine.get_legal_actions(node.state)
        node.set_child_states(actions, self.hex_engine)

        #Using the default policy to choose the best move from the expansion
        best_action = self.default_policy.get_action(node.state,
                                                     actions,
                                                     use_stochastic_moves=True)

        return node, best_action

    def rollout(self, node_action: list[Node, Tuple[int, int]]) -> float:
        '''
        param: node_action: tuple(node, action)
        return: score

        Given a node-action pair representing a leaf node in the search tree, perform a rollout.
        Use the default policy until an end state is reached (i.e., one player wins).
        Return the score associated with the end state.
        '''
        node, action = node_action
        state = self.hex_engine.get_child_board(node.state, action)

        #If the incoming child_state is winning
        if self.hex_engine.check_if_winning(state):
            score = self.hex_engine.get_score(state)
            return score

        #Else - using counter to get the correct value of the score (+/-)
        winning_state = None
        counter = 0

        #Rollout until winning state
        while True:
            if self.hex_engine.check_if_winning(state):
                winning_state = child_state
                break

            legal_actions = self.hex_engine.get_legal_actions(state)

            best_action = self.default_policy.get_action(
                state, legal_actions, use_stochastic_moves=True)
            child_state = self.hex_engine.get_child_board(state, best_action)

            state = child_state
            counter += 1

        #Adjusting the score depending on the number of steps until the root node
        score = self.hex_engine.get_score(winning_state) * (-1)**counter
        return score

    def back_propagate(self, node_action_list: list[Node, Tuple[tuple[int,
                                                                      int],
                                                                int]],
                       score: float) -> None:
        '''
        Backpropagating the score through the visited node-action pairs
        '''
        for i, (node, action) in enumerate(reversed(node_action_list)):
            #Alternating (+/-) score depending on which players turn it is
            node.visit(action, score * (-1)**i)

    def _choose_action(self, node: Node):
        '''
        Given a node in the search tree, choose the best action to take 
        based on the current Q-value estimates and the exploration constant.
        :return argmax(q_sa + u_sa)
        '''
        actions = node.children.keys()
        n_s = node.visit_count
        best_action = max(
            actions,
            key=lambda a: self._score_move(node=node, n_s=n_s, action=a))
        return best_action

    def _score_move(self, node: Node, n_s, action: Tuple[int, int]):
        '''
        Given a node in the search tree, a state, a visit count, and an action, 
        compute the score for that action using the current Q-value estimates and the exploration constant.
        :return (q_sa + u_sa)
        '''
        # visit count
        n_sa = node.edges[action][1]

        u_sa = self.utility_function(n_s=n_s,
                                     n_sa=n_sa,
                                     exploration_const=self.exploration_const)
        q_sa = node.get_value(action)
        return q_sa + u_sa

    def _get_prob_distribution(self, node: Node) -> np.ndarray:
        '''
        Given a node in the search tree, 
        compute the normalized probability distribution over all possible moves 
        based on the visit counts of the child nodes.
        '''
        visit_count = np.zeros_like(self.state[0])
        w = np.zeros_like(self.state[0])
        for key in node.edges.keys():
            move = key[0]
            visit_count[move] = node.edges[key][1]
            w[move] = node.edges[key][0]

        prob_distribution = visit_count / sum(sum(visit_count))
        return prob_distribution
