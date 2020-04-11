from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.halite import Board, get_to_pos, get_col_row
from random import choice, shuffle
from collections import defaultdict
from math import ceil, floor
import numpy as np


env = make("halite", debug=True)
DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']
# compute neighbouring fields for every position
NEIGHBOURS = None
POSITIONS = list()
SHIPYARD_POSITIONS = list()  # our own shipyards
SIZE = 15
HYPERPARAMETERS = {
    'return_time_penalty_factor': 1.1,
    'mining_absolute_halite_threshold': 25,
    'return_absolute_halite': 2000,
    'max_halite': 4000
}

MAX_HALITE = 0  # the maximum amount of halite on one field
HALITE_MAP = None


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
            player_ship_count += 1


def move_ships(obs, config, board: Board):
    size = config.size

    for uid, ship in list(obs.players[obs.player][2].items()):
        if uid in board.action.keys() and board.action[uid] is not None:
            continue
        pos, ship_halite = ship
        move_choices = [None]
        for direction in ["NORTH", "EAST", "SOUTH", "WEST"]:
            to_pos = get_to_pos(size, pos, direction)
            # Enemy shipyard present.
            if board.shipyards[to_pos] != obs.player and board.shipyards[to_pos] != -1:
                continue
            # Larger ship most likely staying in place.
            if board.ships[to_pos] is not None and board.ships[to_pos]["halite"] >= ship_halite:
                continue
            # Weigh the direction based on number of possible larger ships that could be present.
            weight = 6
            if board.ships[to_pos] is not None and board.ships[to_pos]["player_index"] == obs.player:
                weight -= 1
            for s in board.possible_ships[to_pos].values():
                if s["halite"] > ship_halite:
                    weight -= 1
            move_choices += [direction] * weight
        move = choice(move_choices)
        if move is not None:
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
    ship_halite[pos] = start_halite

    while len(Q) > 0:
        u = sorted(Q, key=lambda p: move_cost[p])[0]
        Q.remove(u)
        for neighbour in NEIGHBOURS[u]:
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
    move_costs, _, ship_halite = get_move_costs(obs, config, pos, start_halite)
    return_distances = np.asarray([get_nearest_shipyard_position(p)[1] for p in POSITIONS])
    scores = HYPERPARAMETERS['return_time_penalty_factor'] * return_distances + move_costs * (HYPERPARAMETERS['max_halite'] - start_halite) / (HALITE_MAP + 1e-4)  # avoid division by zero
    return scores


def agent(obs, config):
    player = obs.player
    step = obs.step
    size = config.size
    board = Board(obs, config)
    player_halite, shipyards, ships = obs.players[obs.player]

    global SHIPYARD_POSITIONS, HALITE_MAP, MAX_HALITE
    for uid, pos in shipyards.items():
        if pos not in SHIPYARD_POSITIONS:
            SHIPYARD_POSITIONS.append(pos)
    HALITE_MAP = np.asarray(obs.halite)
    MAX_HALITE = np.max(HALITE_MAP)

    if NEIGHBOURS is None:
        global SIZE
        SIZE = size
        compute_neighbours()
    get_scores(obs, config, 0, 20)
    handle_special_steps(obs, config, board)
    spawn_ships(obs, config, board)
    move_ships(obs, config, board)
    return board.action
