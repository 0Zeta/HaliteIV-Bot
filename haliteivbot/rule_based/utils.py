import math

import numpy as np
import scipy.optimize
from kaggle_environments.envs.halite.helpers import Point, Cell, ShipAction
from scipy.ndimage import gaussian_filter

DIRECTIONS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]
NEIGHBOURS = [Point(0, -1), Point(0, 1), Point(-1, 0), Point(1, 0)]
NEIGHBOURS2 = [
    Point(0, -1),
    Point(0, 1),
    Point(-1, 0),
    Point(1, 0),
    Point(-1, -1),
    Point(1, 1),
    Point(-1, 1),
    Point(1, -1),
]
DISTANCES = None
NAVIGATION = None
FARTHEST_DIRECTIONS_IDX = None
FARTHEST_DIRECTIONS = None
POSITIONS_IN_REACH = None
POSITIONS_IN_REACH_INDICES = None
POSITIONS_IN_SMALL_RADIUS = None
POSITIONS_IN_MEDIUM_RADIUS = None
SIZE = 21
TO_INDEX = {Point.from_index(index, SIZE): index for index in range(SIZE ** 2)}


def create_optimal_mining_steps_tensor(alpha, beta, gamma):
    # The optimal amount of turns spent mining on a cell based on it's distancel, the CHratio and the distance to the nearest friendly shipyard
    # Adapted from https://www.kaggle.com/solverworld/optimal-mining-with-carried-halite
    chrange = 15

    def score(n1, n2, m, H, C):
        return (
            gamma ** (n1 + m)
            * (beta * C + (1 - 0.75 ** m) * 1.02 ** (n1 + m) * H)
            / (n1 + alpha * n2 + m)
        )

    tensor = []
    for n1 in range(22):
        n_opt = []
        for n2 in range(22):
            ch_opt = []
            for ch in range(chrange):
                if ch == 0:
                    CHratio = 0
                else:
                    CHratio = math.exp((ch - 5) / 2.5)

                def h(mine):
                    return -score(n1, n2, mine, 500, CHratio * 500)

                res = scipy.optimize.minimize_scalar(
                    h, bounds=(1, 15), method="Bounded"
                )
                ch_opt.append(int(round(res.x)))
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
            (point + NEIGHBOURS[3]) % SIZE,
        )

    def get_in_reach_indices(position: int):
        point = Point.from_index(position, SIZE)
        return np.array(
            [
                TO_INDEX[point],
                TO_INDEX[(point + NEIGHBOURS[0]) % SIZE],
                TO_INDEX[(point + NEIGHBOURS[1]) % SIZE],
                TO_INDEX[(point + NEIGHBOURS[2]) % SIZE],
                TO_INDEX[(point + NEIGHBOURS[3]) % SIZE],
            ]
        )

    global POSITIONS_IN_REACH, POSITIONS_IN_REACH_INDICES
    POSITIONS_IN_REACH = {
        Point.from_index(pos, SIZE): get_in_reach(pos) for pos in range(SIZE ** 2)
    }
    POSITIONS_IN_REACH_INDICES = np.ndarray((SIZE ** 2, 5), dtype=np.int)
    for pos in range(SIZE ** 2):  # really sad, but it's 4 am
        POSITIONS_IN_REACH_INDICES[pos] = get_in_reach_indices(pos)
    return POSITIONS_IN_REACH, POSITIONS_IN_REACH_INDICES


def get_max_distance(points):
    max_distance = 0
    for i in range(len(points)):
        pos1 = TO_INDEX[points[i]]
        for j in range(i + 1, len(points)):
            pos2 = TO_INDEX[points[j]]
            distance = get_distance(pos1, pos2)
            if distance > max_distance:
                max_distance = distance
    return max_distance


def get_blurred_halite_map(halite, sigma, multiplier=1, size=21):
    halite_map = np.array(halite).reshape((size, -1))
    blurred_halite_map = gaussian_filter(halite_map, sigma, mode="wrap")
    return multiplier * blurred_halite_map.reshape((size ** 2,))


def get_blurred_conflict_map(me, enemies, alpha, sigma, zeta, size=21):
    fight_map = np.full((size, size), fill_value=1, dtype=np.float)
    max_halite = [
        max(ship.halite for ship in player.ships) if len(player.ships) > 0 else 0
        for player in (enemies + [me])
    ]
    if len(max_halite) == 0:
        return
    max_halite = max(max_halite)
    if max_halite <= 0:
        return fight_map.reshape((size ** 2,))
    player_maps = [
        gaussian_filter(_get_player_map(player, max_halite, size), sigma, mode="wrap")
        for player in [me] + enemies
    ]
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


def get_cargo_map(ships, shipyards, halite_norm, size=21):
    cargo_map = np.zeros((size ** 2,), dtype=np.float)
    for ship in ships:
        cargo_map[TO_INDEX[ship.position]] += ship.halite / halite_norm
    for shipyard in shipyards:
        cargo_map[TO_INDEX[shipyard.position]] += 700 / halite_norm
    return 30 * gaussian_filter(
        cargo_map.reshape((SIZE, SIZE)), sigma=2.5, mode="wrap"
    ).reshape((-1,))


def get_hunting_matrix(ships):
    hunting_matrix = np.full(shape=(SIZE ** 2,), fill_value=99999, dtype=np.int)
    for ship in ships:
        for position in POSITIONS_IN_REACH_INDICES[TO_INDEX[ship.position]]:
            if hunting_matrix[position] > ship.halite:
                hunting_matrix[position] = ship.halite
        # hunting_matrix[hunting_matrix > ship.halite][POSITIONS_IN_REACH_INDICES[TO_INDEX[ship.position]]] = ship.halite
    return hunting_matrix


def get_dominance_map(me, opponents, sigma, factor, halite_clip, size=21):
    dominance_map = np.zeros((SIZE ** 2), dtype=np.float)
    for ship in me.ships:
        dominance_map[TO_INDEX[ship.position]] += (
            clip(halite_clip - ship.halite, 0, halite_clip) / halite_clip
        )
    for shipyard in me.shipyards:
        dominance_map[TO_INDEX[shipyard.position]] += 1.5
    for player in opponents:
        for ship in player.ships:
            dominance_map[TO_INDEX[ship.position]] -= (
                clip(halite_clip - ship.halite, 0, halite_clip) / halite_clip
            )
        for shipyard in player.shipyards:
            dominance_map[TO_INDEX[shipyard.position]] -= 1.8
    blurred_dominance_map = gaussian_filter(
        dominance_map.reshape((size, size)), sigma=sigma, mode="wrap"
    )
    return factor * blurred_dominance_map.reshape((-1,))


def get_new_dominance_map(players, sigma, factor, halite_clip, size=21):
    dominance_regions = np.zeros((4, size ** 2), dtype=np.float)
    for player in players:
        player_id = player.id
        dominance_map = np.zeros((size ** 2,), dtype=np.float)
        for ship in player.ships:
            dominance_map[TO_INDEX[ship.position]] = (
                clip(halite_clip - ship.halite, 0, halite_clip) / halite_clip
            )
        for shipyard in player.shipyards:
            dominance_map[TO_INDEX[shipyard.position]] += 1.5
        dominance_regions[player_id] = factor * gaussian_filter(
            dominance_map.reshape((size, size)), sigma=sigma, mode="wrap"
        ).reshape((-1,))

    maxima = np.zeros((4, size ** 2))
    for i in range(4):
        maxima[i] = np.max(
            dominance_regions[[j for j in range(4) if j != i], :], axis=0
        )
    dominance_regions -= maxima
    return dominance_regions


def get_regions(players, sigma, halite_clip, threshold=0.1, size=21):
    dominance_map = get_new_dominance_map(players, sigma, 50, halite_clip, size)
    regions = np.full((SIZE ** 2,), fill_value=-1, dtype=np.int)
    for i in range(4):
        regions[dominance_map[i] >= threshold] = i
    return regions


def get_borders(positions):
    borders = []
    for pos in positions:
        for pos2 in get_neighbouring_positions(Point.from_index(pos, SIZE)):
            if TO_INDEX[pos2] not in positions:
                borders.append(pos)
                break
    return borders


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
        8: ShipAction.SOUTH,
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
    farthest_directions[coldist == rowdist] += (
        direction_x[coldist == rowdist] + direction_y[coldist == rowdist]
    )

    global FARTHEST_DIRECTIONS_IDX
    FARTHEST_DIRECTIONS_IDX = farthest_directions.reshape((size ** 2, size ** 2))

    global FARTHEST_DIRECTIONS
    FARTHEST_DIRECTIONS = [
        [idx_to_action_list[a] for a in b] for b in FARTHEST_DIRECTIONS_IDX
    ]


def dist(a, b):
    diff = abs(a - b)
    return min(diff, SIZE - diff)


def get_axis(direction):
    if direction == ShipAction.NORTH or direction == ShipAction.SOUTH:
        return "y"
    else:
        return "x"


def get_triangles(positions, min_distance, max_distance):
    triangles = []
    if len(positions) < 3:
        return triangles
    for p1 in range(len(positions)):
        A = positions[p1]
        for p2 in range(p1 + 1, len(positions)):
            B = positions[p2]
            for p3 in range(p2 + 1, len(positions)):
                C = positions[p3]
                if is_triangle(A, B, C, min_distance, max_distance):
                    triangles.append((A, B, C))
    return triangles


def is_triangle(A, B, C, min_distance, max_distance):
    if (A.x == B.x == C.x) or (A.y == B.y == C.y):
        return False
    distances = [
        calculate_distance(A, B),
        calculate_distance(A, C),
        calculate_distance(B, C),
    ]
    if any(
        [distance < min_distance or distance > max_distance for distance in distances]
    ):
        return False
    if max(dist(A.x, B.x), dist(A.x, C.x), dist(B.x, C.x)) < 3:
        return False
    if max(dist(A.y, B.y), dist(A.y, C.y), dist(B.y, C.y)) < 3:
        return False
    return True


def create_radius_lists(small_radius, medium_radius):
    global POSITIONS_IN_SMALL_RADIUS
    global POSITIONS_IN_MEDIUM_RADIUS
    POSITIONS_IN_SMALL_RADIUS = create_radius_list(small_radius)
    POSITIONS_IN_MEDIUM_RADIUS = create_radius_list(medium_radius)
    return POSITIONS_IN_SMALL_RADIUS, POSITIONS_IN_MEDIUM_RADIUS


def create_radius_list(radius):
    radius_list = []
    for i in range(SIZE ** 2):
        radius_list.append(np.argwhere(DISTANCES[i] <= radius).reshape((-1,)).tolist())
    return radius_list


def group_ships(ships, max_group_size, max_distance):
    position_to_ship = {TO_INDEX[ship.position]: ship for ship in ships}
    groups = group_positions(
        [TO_INDEX[ship.position] for ship in ships], max_group_size, max_distance
    )
    return [[position_to_ship[position] for position in group] for group in groups]


def group_positions(positions, max_group_size, max_distance):
    groups = [[position] for position in positions]
    current_distance = 1
    while current_distance <= max_distance:
        if len(groups) <= 1:
            break
        unfinished_groups = [group for group in groups if len(group) < max_group_size]
        if len(unfinished_groups) == 0:
            break
        if min([len(group) for group in unfinished_groups]) >= math.ceil(
            max_group_size / 2
        ):
            break
        changed = True
        while changed:
            changed = False
            unfinished_groups = [
                group for group in groups if len(group) < max_group_size
            ]
            unfinished_positions = [
                position for group in unfinished_groups for position in group
            ]
            in_range = {
                position: [
                    pos2
                    for pos2 in unfinished_positions
                    if DISTANCES[position][pos2] == current_distance
                ]
                for position in unfinished_positions
            }
            position_to_group = {
                position: group_id
                for group_id, group in enumerate(groups)
                for position in group
            }
            for position, positions_in_range in in_range.items():
                if len(positions_in_range) == 0:
                    continue
                group1 = position_to_group[position]
                current_group_size = len(groups[group1])
                for pos2 in positions_in_range:
                    group2 = position_to_group[pos2]
                    if group1 == group2:
                        continue
                    if current_group_size + len(groups[group2]) <= max_group_size:
                        # merge the two groups
                        groups[group1].extend(groups[group2])
                        del groups[group2]
                        changed = True
                        break
                if changed:
                    break
        current_distance += 1
    return groups


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


def get_neighbouring_positions(point):
    return [(point + neighbour) % SIZE for neighbour in NEIGHBOURS]


def get_adjacent_positions(point):
    return [TO_INDEX[(point + neighbour) % SIZE] for neighbour in NEIGHBOURS2]


def get_hunting_proportion(players, halite_threshold=0):
    return [
        sum([1 for ship in player.ships if ship.halite <= halite_threshold])
        / len(player.ships)
        if len(player.ships) > 0
        else -1
        for player in players
    ]


class Vector(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        else:
            return Vector(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        else:
            return Vector(self.x - other, self.y - other)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other)

    def __rmul__(self, other):
        return Vector(self.x * other, self.y * other)

    def __abs__(self):
        return abs(self.x) + abs(self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __mod__(self, other):
        return Vector(self.x % other, self.y % other)

    def __str__(self):
        return "(%g, %g)" % (self.x, self.y)

    def __ne__(self, other):
        return not self.__eq__(other)  # reuse __eq__


def get_excircle_midpoint(A: Point, B: Point, C: Point):
    assert A != B != C
    AB = get_vector(A, B)
    AC = get_vector(A, C)
    r = get_orthogonal_vector(AB)
    v = get_orthogonal_vector(AC)
    M1, M2 = 0.5 * Vector(AB.x, AB.y), 0.5 * Vector(AC.x, AC.y)
    a = r.y * v.x - r.x * v.y
    if a == 0:
        BC = get_vector(B, C)
        AB_abs = abs(AB)
        AC_abs = abs(AC)
        BC_abs = abs(BC)
        abs_max = max(AB_abs, AC_abs, BC_abs)
        if AB_abs == abs_max:
            return C
        elif AC_abs == abs_max:
            return B
        else:
            return A
    yololon = (M1.x * v.y - M2.x * v.y - M1.y * v.x + M2.y * v.x) / a
    Q = Vector(A.x, A.y) + M1 + yololon * r
    return Point(round(Q.x), round(Q.y)) % SIZE


def get_vector(A: Point, B: Point):
    def calculate_component(a1, a2):
        amin = min(a1, a2)
        amax = max(a1, a2)
        adiff = amax - amin
        adist = min(adiff, SIZE - adiff)
        if adiff == adist:
            return adiff if a2 == amax else -adiff
        else:
            return -adist if a2 == amax else adist

    return Vector(calculate_component(A.x, B.x), calculate_component(A.y, B.y))


def get_orthogonal_vector(v: Vector):
    return Vector(-v.y, v.x)


def clip(a, minimum, maximum):
    if a <= minimum:
        return minimum
    if a >= maximum:
        return maximum
    return a
