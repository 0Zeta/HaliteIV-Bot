from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.halite import Board, get_to_pos, get_col_row
from random import choice, shuffle
from collections import defaultdict, Counter
from math import ceil, floor
import numpy as np
import logging


logging.basicConfig(level=logging.WARNING)
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
    'return_absolute_halite': 500,
    'max_halite': 4000,
    'halite_score_factor': 0.04,
    'score_preferences_length': 20,
    'score_preferences_stay': 5,
    'max_score_optimization_depth': 100
}

MAX_HALITE = 0  # the maximum amount of halite on one field
HALITE_SUM = 0  # the sum of all halite on the map
HALITE_MAP = None

PLANNED_MOVES = list()  # a list of positions where our ships will be in the next step
RETURNING_SHIPS = list()
DEPOSITING_SHIPS = list()


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


def handle_halite_deposits():
    for uid, ship in RETURNING_SHIPS:
        pos, ship_halite = ship
        if pos in SHIPYARD_POSITIONS:
            RETURNING_SHIPS.remove((uid, ship))
            PLANNED_MOVES.append(pos)
            DEPOSITING_SHIPS.append(uid)
            logging.debug("DEPOSIT: " + uid + " stays at " + str(pos))


def spawn_ships(obs, config, board: Board):
    player = obs.player
    player_halite = obs.players[player][0]
    shipyards = obs.players[player][1]

    player_ship_count = len(obs.players[player][2])
    for uid, pos in shipyards.items():
        if board.ships[pos] is None and player_halite >= config.spawnCost and obs.step < 230 and pos not in PLANNED_MOVES:
            board.spawn(uid)
            PLANNED_MOVES.append(pos)
            player_ship_count += 1
            logging.debug("SPAWN: " + uid + " spawns ship at " + str(pos))


def move_ships(obs, config, board: Board):
    size = config.size
    ships = list(obs.players[obs.player][2].items())
    logging.debug("ship count: " + str(len(ships)))
    if len(ships) == 0:
        return
    mining_ships = ships.copy()
    targets = dict()
    preferences = dict()
    preference_depth = defaultdict(lambda: 0)
    prevs = dict()

    for uid, ship in ships:
        if (uid in board.action.keys() and board.action[uid] is not None) or uid in DEPOSITING_SHIPS:
            mining_ships.remove((uid, ship))
            continue
        pos, ship_halite = ship
        if ship_halite > HYPERPARAMETERS['return_absolute_halite'] or (uid, ship) in RETURNING_SHIPS:
            mining_ships.remove((uid, ship))
            if (uid, ship) not in RETURNING_SHIPS:
                RETURNING_SHIPS.append((uid, ship))
            nearest_shipyard, _ = get_nearest_shipyard_position(pos)
            possible_moves = navigate_to(pos, nearest_shipyard)
            moved = False
            for direction in possible_moves:
                to_pos = get_to_pos(SIZE, pos, direction)
                if to_pos in PLANNED_MOVES or to_pos in ENEMY_SHIPYARD_POSITIONS:
                    continue
                board.move(uid, direction)
                PLANNED_MOVES.append(to_pos)
                logging.debug("RETURN: " + uid + " moves to " + str(to_pos))
                moved = True
                break

            if not moved:
                PLANNED_MOVES.append(pos)
                logging.debug("RETURN: " + uid + " stays at " + str(pos))
            continue

        scores, prev = get_scores(obs, config, pos, ship_halite)
        ship_prefs = np.argsort(scores)
        preferences[uid] = ship_prefs[~np.isin(ship_prefs, SHIPYARD_POSITIONS+ENEMY_SHIPYARD_POSITIONS)][:HYPERPARAMETERS['score_preferences_length']]
        prevs[uid] = prev

    for uid, pref in preferences.items():
        targets[uid] = pref[0]  # Assign every ship the position with the best score

    # Assign  ship targets
    i = 0
    while len(mining_ships) > len(set(targets.values())) and i < HYPERPARAMETERS['max_score_optimization_depth']:
        i += 1
        c = Counter(targets.values())
        conflicting_positions = [k for k in c.keys() if c[k] > 1]
        while len(conflicting_positions) > 0:
            pos = conflicting_positions[0]
            conflicting_ships = [(uid, ship[0]) for uid, ship in mining_ships if targets[uid] == pos]
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
                    logging.debug("MINING (no target): " + uid + " stays at " + str(pos))
                    targets[uid] = ship_position
                else:
                    preference_depth[uid] = preference_depth[uid] + 1
                    targets[uid] = preferences[uid][preference_depth[uid]]
            c = Counter(targets.values())
            conflicting_positions = [k for k in c.keys() if c[k] > 1]

    for uid, ship in mining_ships:
        pos = ship[0]
        target = targets[uid]
        p = target

        while prevs[uid][p] != pos:
            p = prevs[uid][p]
        if p in PLANNED_MOVES:
            moved = False
            p_moves = navigate_to(pos, target)
            for possible_move in p_moves:
                p2 = get_to_pos(SIZE, pos, possible_move)
                if p2 not in PLANNED_MOVES:
                    PLANNED_MOVES.append(p2)
                    board.move(uid, possible_move)
                    logging.debug("MINING (alternative route): " + uid + " moves to " + str(p2))
                    moved = True
                    break
            if not moved:
                for direction in DIRECTIONS:  # redundant
                    p3 = get_to_pos(SIZE, pos, direction)
                    if p3 not in PLANNED_MOVES:
                        PLANNED_MOVES.append(p3)
                        board.move(uid, direction)
                        logging.debug("MINING (avoiding crash): " + uid + " moves to " + str(p3))
                        moved = True
                        break
            if not moved:
                logging.warning("MINING: crash of " + uid + " on position " + str(pos))
            continue

        if p == pos:
            PLANNED_MOVES.append(pos)
            logging.debug("MINING (reached target): " + uid + " stays at " + str(pos))
            continue

        source_x, source_y = get_col_row(SIZE, pos)
        p_x, p_y = get_col_row(SIZE, p)
        if source_y == p_y:
            move = 'EAST' if (source_x + 1) % SIZE == p_x else 'WEST'
        else:
            move = 'SOUTH' if (source_y + 1) % SIZE == p_y else 'NORTH'
        PLANNED_MOVES.append(p)
        board.move(uid, move)
        logging.debug("MINING (on the way): " + uid + " moves to " + str(p))


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


def navigate_to(source, target):
    source_x, source_y = get_col_row(SIZE, source)
    target_x, target_y = get_col_row(SIZE, target)
    possible_moves = list()

    if source_x != target_x:
        if abs(source_x - target_x) < (SIZE - abs(source_x - target_x)):
            ew = 'EAST' if source_x < target_x else 'WEST'
        else:
            ew = 'WEST' if source_x < target_x else 'EAST'
        possible_moves.append(ew)
    if source_y != target_y:
        if abs(source_y - target_y) < (SIZE - abs(source_y - target_y)):
            ns = 'SOUTH' if source_y < target_y else 'NORTH'
        else:
            ns = 'NORTH' if source_y < target_y else 'SOUTH'
        possible_moves.append(ns)
    return possible_moves


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
            alternative = move_cost[u] + 1
            # TODO: consider enemies and own ships
            if alternative < move_cost[neighbour]:
                move_cost[neighbour] = alternative
                prev[neighbour] = u
                if neighbour in SHIPYARD_POSITIONS:
                    ship_halite[neighbour] = 0  # only if the ship stays
                else:
                    ship_halite[neighbour] = ship_halite[u] * (1 - config.moveCost)

    return move_cost, prev, ship_halite


def get_scores(obs, config, pos, start_halite):
    move_costs, prev, ship_halite = get_move_costs(obs, config, pos, start_halite)
    return_distances = np.asarray([get_nearest_shipyard_position(p)[1] for p in POSITIONS])
    scores = HYPERPARAMETERS['return_time_penalty_factor'] * return_distances + move_costs - HYPERPARAMETERS['halite_score_factor'] * HALITE_MAP
    return scores, prev


def agent(obs, config):
    logging.debug("Begin step " + str(obs.step))
    player = obs.player
    step = obs.step
    size = config.size
    board = Board(obs, config)
    player_halite, shipyards, ships = obs.players[obs.player]

    global SHIPYARD_POSITIONS, ENEMY_SHIPYARD_POSITIONS, HALITE_MAP, MAX_HALITE, HALITE_SUM, PLANNED_MOVES, DEPOSITING_SHIPS
    PLANNED_MOVES = list()
    DEPOSITING_SHIPS = list()
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

    handle_special_steps(obs, config, board)
    handle_halite_deposits()
    spawn_ships(obs, config, board)
    move_ships(obs, config, board)
    return board.action
