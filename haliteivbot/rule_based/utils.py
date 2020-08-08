import scipy.optimize
from kaggle_environments.envs.halite.halite import *
from kaggle_environments.envs.halite.helpers import Point, Cell
from scipy.ndimage import gaussian_filter

DIRECTIONS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
NEIGHBOURS = [Point(0, -1), Point(0, 1), Point(-1, 0), Point(1, 0)]
DISTANCES = None
NAVIGATION = None
FARTHEST_DIRECTIONS_IDX = None
FARTHEST_DIRECTIONS = None
POSITIONS_IN_REACH = None
POSITIONS_IN_SMALL_RADIUS = None
POSITIONS_IN_MEDIUM_RADIUS = None
SIZE = 21
TO_INDEX = {Point.from_index(index, SIZE): index for index in range(SIZE ** 2)}


def create_optimal_mining_steps_tensor(alpha, beta, gamma):
    # The optimal amount of turns spent mining on a cell based on it's distancel, the CHratio and the distance to the nearest friendly shipyard
    # Adapted from https://www.kaggle.com/solverworld/optimal-mining-with-carried-halite
    chrange = 11

    def score(n1, n2, m, H, C):
        return gamma ** (n1 + m) * (beta * C + (1 - .75 ** m) * 1.02 ** (n1 + m) * H) / (n1 + alpha * n2 + m)

    tensor = []
    for n1 in range(20):
        n_opt = []
        for n2 in range(20):
            ch_opt = []
            for ch in range(chrange):
                if ch == 0:
                    CHratio = 0
                else:
                    CHratio = math.exp((ch - 5) / 2.5)

                def h(mine):
                    return -score(n1, n2, mine, 500, CHratio * 500)

                res = scipy.optimize.minimize_scalar(h, bounds=(1, 15), method='Bounded')
                ch_opt.append(round(res.x))
            n_opt.append(ch_opt)
        tensor.append(n_opt)
    return tensor


def compute_positions_in_reach():
    def get_in_reach(position: int):
        point = Point.from_index(position, SIZE)
        return (
            point,
            (point + NEIGHBOURS[0]) % SIZE,
            (point + NEIGHBOURS[1]) % SIZE,
            (point + NEIGHBOURS[2]) % SIZE,
            (point + NEIGHBOURS[3]) % SIZE
        )

    global POSITIONS_IN_REACH
    POSITIONS_IN_REACH = {Point.from_index(pos, SIZE): get_in_reach(pos) for pos in range(SIZE ** 2)}
    return POSITIONS_IN_REACH


def get_blurred_halite_map(halite, sigma, multiplier=1, size=21):
    halite_map = np.array(halite).reshape((size, -1))
    blurred_halite_map = gaussian_filter(halite_map, sigma, mode='wrap')
    return multiplier * blurred_halite_map.reshape((size ** 2,))


def get_blurred_conflict_map(me, enemies, alpha, sigma, zeta, size=21):
    fight_map = np.full((size, size), fill_value=1, dtype=np.float)
    max_halite = [max(ship.halite for ship in player.ships) if len(player.ships) > 0 else 0 for player in
                  (enemies + [me])]
    if len(max_halite) == 0:
        return
    max_halite = max(max_halite)
    if max_halite <= 0:
        return fight_map.reshape((size ** 2,))
    player_maps = [gaussian_filter(_get_player_map(player, max_halite, size), sigma, mode='wrap') for player in
                   [me] + enemies]
    max_value = max([np.max(player_map) for player_map in player_maps])
    for player_index, player_map in enumerate(player_maps):
        player_map = (player_map / max_value) * zeta + 1
        if player_index == 0:
            player_map *= alpha
        fight_map = np.multiply(fight_map, player_map)
    return fight_map.reshape((size ** 2,))


def _get_player_map(player, max_halite, size=21):
    player_map = np.ndarray((size ** 2,), dtype=np.float)
    for ship in player.ships:
        player_map[TO_INDEX[ship.position]] = ship.halite / max_halite
    for shipyard in player.shipyards:
        player_map[TO_INDEX[shipyard.position]] = max_halite / 2
    return player_map.reshape((size, size))


def get_dominance_map(me, opponents, sigma, radius, halite_clip, size=21):
    dominance_map = np.zeros((size ** 2,), dtype=np.float)
    if radius == 'small':
        radius_map = POSITIONS_IN_SMALL_RADIUS
    elif radius == 'medium':
        radius_map = POSITIONS_IN_MEDIUM_RADIUS
    else:
        raise Exception('Invalid radius type: ', radius)

    for ship in me.ships:
        dominance_map[radius_map[TO_INDEX[ship.position]]] += clip(halite_clip - ship.halite, 0,
                                                                   halite_clip) / halite_clip
    for shipyard in me.shipyards:
        dominance_map[radius_map[TO_INDEX[shipyard.position]]] += 1
    for player in opponents:
        for ship in player.ships:
            dominance_map[radius_map[TO_INDEX[ship.position]]] -= 1
        for shipyard in player.shipyards:
            dominance_map[radius_map[TO_INDEX[shipyard.position]]] -= 1

    blurred_dominance_map = gaussian_filter(dominance_map.reshape((size, size)), sigma=sigma, mode='wrap')
    return blurred_dominance_map.reshape((-1,))


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
    rowdist = c2[0]
    coldist = c1[0]
    dist_matrix = (rowdist + coldist).reshape(size ** 2, -1)

    direction_x = c2[1]
    direction_y = c1[1]
    dir_matrix = (direction_x + direction_y).reshape(size ** 2, -1)

    global DISTANCES
    DISTANCES = dist_matrix

    global NAVIGATION
    NAVIGATION = [[idx_to_action_list[a] for a in b] for b in dir_matrix]

    farthest_directions = np.zeros((size ** 4), dtype=np.int)
    farthest_directions[coldist < rowdist] += direction_x[coldist < rowdist]
    farthest_directions[coldist > rowdist] += direction_y[coldist > rowdist]
    farthest_directions[coldist == rowdist] += direction_x[coldist == rowdist] + direction_y[coldist == rowdist]

    global FARTHEST_DIRECTIONS_IDX
    FARTHEST_DIRECTIONS_IDX = farthest_directions.reshape((size ** 2, size ** 2))

    global FARTHEST_DIRECTIONS
    FARTHEST_DIRECTIONS = [[idx_to_action_list[a] for a in b] for b in FARTHEST_DIRECTIONS_IDX]


def create_radius_lists(small_radius, medium_radius):
    global POSITIONS_IN_SMALL_RADIUS
    global POSITIONS_IN_MEDIUM_RADIUS
    POSITIONS_IN_SMALL_RADIUS = _create_radius_list(small_radius)
    POSITIONS_IN_MEDIUM_RADIUS = _create_radius_list(medium_radius)
    return POSITIONS_IN_SMALL_RADIUS, POSITIONS_IN_MEDIUM_RADIUS


def _create_radius_list(radius):
    radius_list = []
    for i in range(SIZE ** 2):
        radius_list.append(np.argwhere(DISTANCES[i] <= radius).reshape((-1,)).tolist())
    return radius_list


def navigate(source: Point, target: Point, size: int):
    return NAVIGATION[TO_INDEX[source]][TO_INDEX[target]]


def nav(source: int, target: int):
    return NAVIGATION[source][target]


def get_inefficient_directions(directions):
    return [dir for dir in DIRECTIONS if dir not in directions]


def get_direction_to_neighbour(source: int, target: int) -> ShipAction:
    return NAVIGATION[source][target][0]


def calculate_distance(source: Point, target: Point):
    """
    Compute the Manhattan distance between two positions.
    :param source: The source from where to calculate
    :param target: The target to where calculate
    :return: The distance between the two positions
    """
    return DISTANCES[TO_INDEX[source]][TO_INDEX[target]]


def get_distance(source: int, target: int):
    return DISTANCES[source][target]


def get_distance_matrix():
    return DISTANCES


def get_farthest_directions_matrix():
    return FARTHEST_DIRECTIONS_IDX


def get_farthest_directions_list():
    return FARTHEST_DIRECTIONS


def get_neighbours(cell: Cell):
    return [cell.neighbor(point) for point in NEIGHBOURS]


def clip(a, minimum, maximum):
    if a <= minimum:
        return minimum
    if a >= maximum:
        return maximum
    return a


class imdict(dict):
    def __hash__(self):
        return id(self)

    def _immutable(self, *args, **kws):
        raise TypeError('object is immutable')

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    update = _immutable
    setdefault = _immutable
    pop = _immutable
    popitem = _immutable
