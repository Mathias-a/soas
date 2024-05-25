from hex_engine import Hex_Engine
import random
import numpy as np
import os

import cairo
from PIL import Image

"""
------------------------------ Hex game functions ------------------------------
"""


def play_game(
    hex_engine: Hex_Engine, p1_policy: callable, p2_policy: callable
) -> tuple[int, list]:
    """
    Play a game of Hex.

    Args:
        hex_engine: The Hex engine to use.
        p1_policy: The policy for player 1.
        p2_policy: The policy for player 2.

    Returns:
        The winner of the game and the state history.
    """
    board = hex_engine.get_start_board()
    player = 1
    state = (board, player)
    winner = None

    state_history = []

    while len(hex_engine.get_legal_actions(state)) > 0 and winner is None:
        pretty_print_state(state, f"images/input/move{len(state_history)}.png")
        state_history.append(state)
        _, player = state
        if player == 1:
            move, _ = p1_policy(state, hex_engine=hex_engine)
        else:
            move, _ = p2_policy(state, hex_engine=hex_engine)

        child_state = hex_engine.get_child_board(state, move)

        if hex_engine.check_if_winning(child_state):
            winner = player
            state_history.append(child_state)

        # set state as the board state and the other player
        state = child_state
    return winner, state_history


def random_player(
    state: tuple[np.ndarray, int], **kwargs
) -> tuple[tuple[int, int], int]:
    """
    Choose a random move for a player.

    Args:
        state: The current state of the game.

    Returns:
        The move to make and None.
    """
    # Get hex engine from kwargs, or create a new one if it doesn't exist
    hex_engine = kwargs.get("hex_engine")
    if hex_engine is None:
        hex_engine = Hex_Engine(len(state[0]))

    legal_actions = hex_engine.get_legal_actions(state)
    return random.choice(legal_actions), None


def input_player(
    state: tuple[np.ndarray, int], **kwargs
) -> tuple[tuple[int, int], int]:
    """
    Get a move from a human player.

    Args:
        state: The current state of the game.

    Returns:
        The move to make and None.
    """
    hex_engine = kwargs.get("hex_engine")
    if hex_engine is None:
        hex_engine = Hex_Engine(len(state[0]))

    curr_board, player = state
    print(curr_board)
    while True:
        input_move = input("place piece at coordinate i.e: 2,0 >> ")
        try:
            x, y = input_move.strip(" ").split(",")
            move = ((int(x), int(y)), player)
            if move in hex_engine.get_legal_actions(state):
                return move, None
            else:
                print("Invalid move")
        except ValueError:
            print("Invalid input")


def pretty_print_state(state, path):
    """
    Draw the Hex board and display it in a window.
    p1 is blue, p2 is red, empty is white.
    Red is playing from left to right, blie is playing from top to bottom.

    Args:
        state: The current state of the game.
    """
    # Create a surface
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 640, 480)

    # Create a context
    ctx = cairo.Context(surface)

    # Define the hexagon properties
    hex_radius = 20
    hex_side = 2 * hex_radius * np.sin(np.pi / 3)
    hex_width = 2 * hex_radius
    hex_height = 2 * hex_radius * np.cos(np.pi / 6)

    # Define the starting position of the first hexagon
    x_start = 40
    y_start = 40

    # Loop over the board and draw hexagons
    for i in range(len(state[0])):
        for j in range(len(state[0])):
            # Calculate the center of the current hexagon
            x_center = x_start + i * hex_width + j * hex_radius
            y_center = y_start + j * hex_height

            # Draw the hexagon
            ctx.move_to(
                x_center + hex_radius * np.cos(np.pi / 6),
                y_center + hex_radius * np.sin(np.pi / 6),
            )
            ctx.line_to(
                x_center + hex_radius * np.cos(np.pi / 6 + np.pi / 3),
                y_center + hex_radius * np.sin(np.pi / 6 + np.pi / 3),
            )
            ctx.line_to(
                x_center + hex_radius * np.cos(np.pi / 6 + 2 * np.pi / 3),
                y_center + hex_radius * np.sin(np.pi / 6 + 2 * np.pi / 3),
            )
            ctx.line_to(
                x_center + hex_radius * np.cos(np.pi / 6 + 3 * np.pi / 3),
                y_center + hex_radius * np.sin(np.pi / 6 + 3 * np.pi / 3),
            )
            ctx.line_to(
                x_center + hex_radius * np.cos(np.pi / 6 + 4 * np.pi / 3),
                y_center + hex_radius * np.sin(np.pi / 6 + 4 * np.pi / 3),
            )
            ctx.line_to(
                x_center + hex_radius * np.cos(np.pi / 6 + 5 * np.pi / 3),
                y_center + hex_radius * np.sin(np.pi / 6 + 5 * np.pi / 3),
            )
            ctx.close_path()

            # Set the fill color of the hexagon
            if state[0][i, j] == 1:
                ctx.set_source_rgb(1, 0, 0)
            elif state[0][i, j] == 2:
                ctx.set_source_rgb(0, 0, 1)
            else:
                ctx.set_source_rgb(1, 1, 1)

            # Fill the hexagon
            ctx.fill()

    # Write the surface to a file
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    surface.write_to_png(path)


"""
------------------------------------ Utility ---------------------------------------
"""


def upper_confidence_trees(**kwargs):
    n_s = kwargs.get("n_s")
    n_sa = kwargs.get("n_sa", n_s)
    exploration_const = kwargs.get("exploration_const", 1)
    return exploration_const * np.sqrt((np.log(n_s)) / (1 + n_sa))


def increment_directory_name(directory_path):
    num = 1
    while os.path.exists(os.path.join(directory_path, f"folder_{num}")):
        num += 1
    new_dir = os.path.join(directory_path, f"folder_{num}")
    os.makedirs(new_dir)
    return new_dir


if __name__ == "__main__":
    play_game(Hex_Engine(4), random_player, random_player)
