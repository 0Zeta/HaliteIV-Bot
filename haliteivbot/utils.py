import scipy.optimize
from kaggle_environments.envs.halite.halite import *
from kaggle_environments.envs.halite.helpers import Point, Cell
from scipy.ndimage import gaussian_filter

DIRECTIONS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
NEIGHBOURS = [Point(0, -1), Point(0, 1), Point(-1, 0), Point(1, 0)]
DISTANCES = None
NAVIGATION = None
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


def get_blurred_halite_map(halite, sigma, multiplier=1, size=21):
    halite_map = np.array(halite).reshape((size, -1))
    blurred_halite_map = gaussian_filter(halite_map, sigma, mode='wrap')
    return multiplier * blurred_halite_map.reshape((size ** 2,))


def create_navigation_lists(size):
    """distance list taken from https://www.kaggle.com/jpmiller/fast-distance-calcs by JohnM"""
    base = np.arange(size ** 2)
    idx1 = np.repeat(base, size ** 2)
    idx2 = np.tile(base, size ** 2)

    idx_to_action = {
        0: None,
        1: ShipAction.WEST,
        2: ShipAction.EAST,
        4: ShipAction.NORTH,
        8: ShipAction.SOUTH
    }

    idx_to_action_list = dict()
    for int_a, action_a in [(i, idx_to_action[i]) for i in range(3)]:
        for int_b, action_b in [(j, idx_to_action[j]) for j in (0, 4, 8)]:
            action_list = []
            if action_a is not None:
                action_list.append(action_a)
            if action_b is not None:
                action_list.append(action_b)
            idx_to_action_list[int_a + int_b] = action_list

    def calculate(a1, a2, smaller_val, greater_val):
        amin = np.fmin(a1, a2)
        amax = np.fmax(a1, a2)
        adiff = amax - amin
        adist = np.fmin(adiff, size - adiff)
        wrap_around = np.not_equal(adiff, adist)
        directions = np.zeros((len(a1),), dtype=np.int)
        greater = np.greater(a2, a1)
        smaller = np.greater(a1, a2)
        directions[greater != wrap_around] = greater_val
        directions[smaller != wrap_around] = smaller_val
        return adist, directions

    c1 = calculate(idx1 // size, idx2 // size, 4, 8)
    c2 = calculate(idx1 % size, idx2 % size, 1, 2)
    rowdist = c1[0]
    coldist = c2[0]
    dist_matrix = (rowdist + coldist).reshape(size ** 2, -1)

    direction_x = c2[1]
    direction_y = c1[1]
    dir_matrix = (direction_x + direction_y).reshape(size ** 2, -1)

    global DISTANCES
    DISTANCES = dist_matrix.tolist()

    global NAVIGATION
    NAVIGATION = [[idx_to_action_list[a] for a in b] for b in dir_matrix]


def navigate(source: Point, target: Point, size: int):
    return NAVIGATION[source.to_index(size)][target.to_index(size)]


def nav(source: int, target: int):
    return NAVIGATION[source][target]


def get_direction_to_neighbour(source: int, target: int) -> ShipAction:
    return NAVIGATION[int][int][0]


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
