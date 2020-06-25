import logging

from kaggle_environments import make
from kaggle_environments.envs.halite.halite import *

logging.basicConfig(level=logging.WARNING)
env = make("halite", debug=True)
DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']

CONFIG = None
SIZE = 21

ME = None

PLANNED_MOVES = list()  # a list of positions where our ships will be in the next step


def handle_special_steps(board: Board):
    step = board.step
    if step == 0:
        # Immediately spawn a shipyard
        ME.ships[0].next_action = ShipAction.CONVERT
        return True
    return False


def handle_halite_deposits(board: Board):
    pass


def spawn_ships(board: Board):
    pass


def move_ships(board: Board):
    pass


@board_agent
def agent(board: Board):
    obs = board.observation
    global CONFIG
    if CONFIG is None:
        CONFIG = board.configuration
        global SIZE
        SIZE = CONFIG.size

    global ME
    ME = board.current_player

    logging.debug("Begin step " + str(board.step))

    if handle_special_steps(board):
        return  # don't execute the functions below
