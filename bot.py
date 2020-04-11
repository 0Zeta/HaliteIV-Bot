from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.halite import Board, get_to_pos, get_col_row
from random import choice, shuffle
from collections import defaultdict, Counter
from math import ceil, floor
import numpy as np


env = make("halite", debug=True)
DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']
# compute neighbouring fields for every position
NEIGHBOURS = None
POSITIONS = list()
SHIPYARD_POSITIONS = list()  # our own shipyards
ENEMY_SHIPYARD_POSITIONS = list()
SIZE = 15
HYPERPARAMETERS = {
    'return_time_penalty_factor': 1.1,
    'mining_absolute_halite_threshold': 25,
    'return_absolute_halite': 2000,
    'max_halite': 4000,
    'halite_score_factor': 0.05,
    'score_preferences_length': 20,
    'score_preferences_stay': 5,
    'max_score_optimization_depth': 100
}

MAX_HALITE = 0  # the maximum amount of halite on one field
HALITE_SUM = 0  # the sum of all halite on the map
HALITE_MAP = None

PLANNED_MOVES = list()  # a list of positions where our ships will be in the next step


def compute_neighbours():
    global POSITIONS, NEIGHBOURS
    POSITIONS = list(range(SIZE**2))
    NEIGHBOURS = dict()
    for position in POSITIONS:
        NEIGHBOURS[position] = [get_to_pos(SIZE, position, direction) for direction in DIRECTIONS]


def handle_special_steps(obs, config, board: Board):
    ships_items = list(obs.players[obs.player][2].items())
    if obs.step == 1 and len(ships_items) > 0:  # Immediately create a shipyard on the first move
        ship_id, ship = ships_items[0]
        board.convert(ship_id)
        # TODO: add shipyard to list of shipyards


def spawn_ships(obs, config, board: Board):
    player = obs.player
    opponent = 1 if player == 0 else 0  # Adapt to work with multiple opponents
    player_halite = obs.players[player][0]
    shipyards = obs.players[player][1]

    player_ship_count = len(obs.players[player][2])
    opponent_ship_count = len(obs.players[opponent][2])
    for uid, pos in shipyards.items():
        if board.ships[pos] is None and player_halite >= config.spawnCost and player_ship_count < (opponent_ship_count + 3):
            board.spawn(uid)
            PLANNED_MOVES.append(pos)
            player_ship_count += 1


def move_ships(obs, config, board: Board):
    size = config.size
    ships = list(obs.players[obs.player][2].items())
    if len(ships) == 0:
        return
    mining_ships = ships.copy()
    targets = dict()
    preferences = dict()
    preference_depth = defaultdict(lambda: 0)
    prevs = dict()

    for uid, ship in ships:
        if uid in board.action.keys() and board.action[uid] is not None:
            mining_ships.remove((uid, ship))
            continue
        pos, ship_halite = ship
        scores, prev = get_scores(obs, config, pos, ship_halite)
        preferences[uid] = np.argsort(scores)[:HYPERPARAMETERS['score_preferences_length']]
        prevs[uid] = prev
        if pos in preferences[uid][:HYPERPARAMETERS['score_preferences_stay']]:
            PLANNED_MOVES.append(pos)
            targets[uid] = pos

    for uid, pref in preferences.items():
        targets[uid] = pref[0]  # Assign every ship the position with the best score

    # Assign  ship targets
    i = 0
    while len(mining_ships) > len(set(targets.values())) and i < HYPERPARAMETERS['max_score_optimization_depth']:
        i += 1
        c = Counter(targets.values())
        for pos in [k for k in c.keys() if c[k] > 1]:
            conflicting_ships = [(uid, ship[0]) for uid, ship in ships if targets[uid] == pos]
            ship_with_min_distance = None
            min_distance = float('inf')
            for uid, ship_position in conflicting_ships:
                distance = calculate_distance(ship_position, pos)
                if distance < min_distance:
                    ship_with_min_distance = (uid, ship_position)
                    min_distance = distance
            conflicting_ships.remove(ship_with_min_distance)
            for uid, ship_position in conflicting_ships:
                if preference_depth[uid] >= HYPERPARAMETERS['score_preferences_length'] - 1:
                    PLANNED_MOVES.append(ship_position)  # TODO: add behaviour for ships not getting any of their preferences
                    targets[uid] = ship_position
                else:
                    preference_depth[uid] += 1
                    targets[uid] = preferences[uid][preference_depth[uid]]

    for uid, ship in mining_ships:
        pos = ship[0]
        target = targets[uid]
        p = target

        while prevs[uid][p] != pos:
            p = prevs[uid][p]
        if p in PLANNED_MOVES:
            PLANNED_MOVES.append(pos)  # do nothing TODO: check for an alternative route
            continue

        if p == pos:
            PLANNED_MOVES.append(pos)
            continue

        source_x, source_y = get_col_row(SIZE, pos)
        p_x, p_y = get_col_row(SIZE, p)
        if source_y == p_y:
            move = 'EAST' if (source_x + 1) % SIZE == p_x else 'WEST'
        else:
            move = 'SOUTH' if (source_y + 1) % SIZE == p_y else 'NORTH'
        PLANNED_MOVES.append(p)
        board.move(uid, move)


def calculate_distance(source, target):
    """
    Compute the Manhattan distance between two positions.
    :param source: The source from where to calculate
    :param target: The target to where calculate
    :return: The distance between the two positions
    """
    source_x, source_y = get_col_row(SIZE, source)
    target_x, target_y = get_col_row(SIZE, target)
    delta_x = min(abs(source_x - target_x), SIZE - abs(source_x - target_x))
    delta_y = min(abs(source_y - target_y), SIZE - abs(source_y - target_y))
    return delta_x + delta_y


def get_nearest_shipyard_position(pos):
    min_distance = float('inf')
    nearest_shipyard = None
    for shipyard_position in SHIPYARD_POSITIONS:
        distance = calculate_distance(pos, shipyard_position)
        if distance < min_distance:
            min_distance = distance
            nearest_shipyard = shipyard_position
    return nearest_shipyard, min_distance


def get_move_costs(obs, config, pos, start_halite):
    """implements Djikstra's algorithm"""
    halite_map = obs.halite
    move_cost = np.full(shape=(len(POSITIONS),), fill_value=999999, dtype=np.int)
    ship_halite = np.full(shape=(len(POSITIONS),), fill_value=0.0, dtype=np.float)
    prev = defaultdict(lambda: None)
    Q = POSITIONS.copy()
    move_cost[pos] = 0
    prev[pos] = pos
    ship_halite[pos] = start_halite

    while len(Q) > 0:
        u = sorted(Q, key=lambda p: move_cost[p])[0]
        Q.remove(u)
        for neighbour in NEIGHBOURS[u]:
            if u in ENEMY_SHIPYARD_POSITIONS:
                continue
            u_halite = halite_map[u] * (1 + config.regenRate) ** move_cost[u]  # halite on the position u
            must_stay = u_halite * config.moveCost > ship_halite[u]
            alternative = move_cost[u] + (1 if not must_stay else 2)
            # TODO: consider enemies and own ships
            if alternative < move_cost[neighbour]:
                move_cost[neighbour] = alternative
                prev[neighbour] = u
                if neighbour in SHIPYARD_POSITIONS:
                    ship_halite[neighbour] = 0
                elif must_stay:
                    ship_halite[neighbour] = ship_halite[u] + ceil(u_halite * config.collectRate) - (floor(u_halite * (1 - config.collectRate)) * (1 + config.regenRate) * config.moveCost)
                else:
                    ship_halite[neighbour] = ship_halite[u] - u_halite * config.moveCost

    return move_cost, prev, ship_halite


def get_scores(obs, config, pos, start_halite):
    move_costs, prev, ship_halite = get_move_costs(obs, config, pos, start_halite)
    return_distances = np.asarray([get_nearest_shipyard_position(p)[1] for p in POSITIONS])
    scores = HYPERPARAMETERS['return_time_penalty_factor'] * return_distances + move_costs - HYPERPARAMETERS['halite_score_factor'] * HALITE_MAP
    return scores, prev


def agent(obs, config):
    player = obs.player
    step = obs.step
    size = config.size
    board = Board(obs, config)
    player_halite, shipyards, ships = obs.players[obs.player]

    global SHIPYARD_POSITIONS, ENEMY_SHIPYARD_POSITIONS, HALITE_MAP, MAX_HALITE, HALITE_SUM, PLANNED_MOVES, FIRST_MAP
    PLANNED_MOVES = list()
    for uid, pos in shipyards.items():
        if pos not in SHIPYARD_POSITIONS:
            SHIPYARD_POSITIONS.append(pos)
    for pos in POSITIONS:
        if board.shipyards[pos] != -1 and board.shipyards[pos] != player and pos not in ENEMY_SHIPYARD_POSITIONS:
            ENEMY_SHIPYARD_POSITIONS.append(pos)
    HALITE_MAP = np.asarray(obs.halite)
    MAX_HALITE = np.max(HALITE_MAP)
    HALITE_SUM = np.sum(HALITE_MAP)

    if NEIGHBOURS is None:
        global SIZE
        SIZE = size
        compute_neighbours()
    get_scores(obs, config, 0, 20)
    handle_special_steps(obs, config, board)
    spawn_ships(obs, config, board)
    move_ships(obs, config, board)
    return board.action
