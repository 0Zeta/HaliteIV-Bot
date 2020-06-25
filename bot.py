import logging

from kaggle_environments import make
from kaggle_environments.envs.halite.halite import *

logging.basicConfig(level=logging.WARNING)
env = make("halite", debug=True)
DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']

PLANNED_MOVES = list()  # a list of positions where our ships will be in the next step


def handle_special_steps(board: Board):
    pass


def handle_halite_deposits(board: Board):
    pass


def spawn_ships(board: Board):
    pass


def move_ships(board: Board):
    pass


@board_agent
def agent(board: Board):
    obs = board.observation
    logging.debug("Begin step " + str(obs.step))
