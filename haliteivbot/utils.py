import scipy.optimize
from kaggle_environments.envs.halite.halite import *
from kaggle_environments.envs.halite.helpers import Point, Cell

DIRECTIONS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
NEIGHBOURS = [Point(0, -1), Point(0, 1), Point(-1, 0), Point(1, 0)]
DISTANCES = None
SIZE = 21


def create_optimal_mining_steps_matrix(alpha, gamma):
    # The optimal amount of turns spent mining on a cell based on it's distance and the distance to the nearest friendly shipyard
    def score(n1, n2, m, H):
        return gamma ** (n1 + m) * (1 - .75 ** m) * 1.02 ** (n1 + m) * H / (n1 + alpha * n2 + m)

    matrix = []
    for n1 in range(20):
        n_opt = []
        for n2 in range(20):
            def h(mine):
                return -score(n1, n2, mine, 500)

            res = scipy.optimize.minimize_scalar(h, bounds=(1, 15), method='Bounded')
            n_opt.append(res.x)
        matrix.append(n_opt)
    return matrix


def create_distance_list(size):
    """taken from https://www.kaggle.com/jpmiller/fast-distance-calcs by JohnM"""

    def dist_1d(a1, a2):
        amin = np.fmin(a1, a2)
        amax = np.fmax(a1, a2)
        adiff = amax - amin
        adist = np.fmin(adiff, size - adiff)
        return adist

    base = np.arange(size ** 2)
    idx1 = np.repeat(base, size ** 2)
    idx2 = np.tile(base, size ** 2)

    rowdist = dist_1d(idx1 // size, idx2 // size)
    coldist = dist_1d(idx1 % size, idx2 % size)
    dist_matrix = (rowdist + coldist).reshape(size ** 2, -1)

    global DISTANCES
    DISTANCES = dist_matrix.tolist()


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


def calculate_distance(source: Point, target: Point):
    """
    Compute the Manhattan distance between two positions.
    :param source: The source from where to calculate
    :param target: The target to where calculate
    :return: The distance between the two positions
    """
    return DISTANCES[source.to_index(SIZE)][target.to_index(SIZE)]


def get_distance(source: int, target: int):
    return DISTANCES[source][target]


def get_neighbours(cell: Cell):
    return [cell.neighbor(point) for point in NEIGHBOURS]
