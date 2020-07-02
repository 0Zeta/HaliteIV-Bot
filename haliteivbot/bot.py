import logging

from kaggle_environments import make
from kaggle_environments.envs.halite.halite import *

logging.basicConfig(level=logging.WARNING)
env = make("halite", debug=True)
DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']

BOT = None


@board_agent
def agent(board: Board):
    global BOT
    if BOT is None:
        BOT = HaliteBot(board)

    logging.debug("Begin step " + str(board.step))
    BOT.step(board)


class HaliteBot(object):

    def __init__(self, board: Board):
        self.config = board.configuration
        self.size = self.config.size
        self.me = board.current_player

        self.planned_moves = list()  # a list of positions where our ships will be in the next step

    def step(self, board: Board):
        obs = board.observation
        if self.handle_special_steps(board):
            return  # don't execute the functions below

    def handle_special_steps(self, board: Board):
        step = board.step
        if step == 0:
            # Immediately spawn a shipyard
            self.me.ships[0].next_action = ShipAction.CONVERT
            return True
        return False

    def handle_halite_deposits(self, board: Board):
        pass

    def spawn_ships(self, board: Board):
        pass

    def move_ships(self, board: Board):
        pass
