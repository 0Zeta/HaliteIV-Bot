from kaggle_environments.envs.halite.halite import *
from kaggle_environments.envs.halite.helpers import Point, Cell

DIRECTIONS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
NEIGHBOURS = [Point(0, -1), Point(0, 1), Point(-1, 0), Point(1, 0)]


def navigate(source: Point, target: Point, size: int):
    possible_moves = list()

    if source.x != target.x:
        if abs(source.x - target.x) < (size - abs(source.x - target.x)):
            ew = ShipAction.EAST if source.x < target.x else ShipAction.WEST
        else:
            ew = ShipAction.WEST if source.x < target.x else ShipAction.EAST
        possible_moves.append(ew)
    if source.y != target.y:
        if abs(source.y - target.y) < (size - abs(source.y - target.y)):
            ns = ShipAction.NORTH if source.y < target.y else ShipAction.SOUTH
        else:
            ns = ShipAction.SOUTH if source.y < target.y else ShipAction.NORTH
        possible_moves.append(ns)
    return possible_moves


def calculate_distance(source: Point, target: Point, size):
    """
    Compute the Manhattan distance between two positions.
    :param source: The source from where to calculate
    :param target: The target to where calculate
    :param size:   The size of the board
    :return: The distance between the two positions
    """
    delta_x = min(abs(source.x - target.x), size - abs(source.x - target.x))
    delta_y = min(abs(source.y - target.y), size - abs(source.y - target.y))
    return delta_x + delta_y


def get_neighbours(cell: Cell):
    return [cell.neighbor(point) for point in NEIGHBOURS]
