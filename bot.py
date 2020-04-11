from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.halite import Board, get_to_pos
from random import choice, shuffle
from collections import defaultdict


env = make("halite", debug=True)
DIRECTIONS = ['NORTH', 'EAST', 'SOUTH', 'WEST']
# compute neighbouring fields for every position
NEIGHBOURS = None
POSITIONS = list()


def compute_neighbours(size):
    global POSITIONS, NEIGHBOURS
    POSITIONS = list(range(size**2))
    NEIGHBOURS = dict()
    for position in POSITIONS:
        NEIGHBOURS[position] = [get_to_pos(size, position, direction) for direction in DIRECTIONS]


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


def get_move_costs(obs, config, pos, start_halite):
    """implements Djikstra's algorithm"""
    halite_map = obs.halite
    move_cost = defaultdict(lambda: float('inf'))
    ship_halite = defaultdict(lambda: 0)
    prev = defaultdict(lambda: None)
    Q = POSITIONS.copy()
    move_cost[pos] = 0
    ship_halite[pos] = start_halite

    while len(Q) > 0:
        u = sorted(Q, key=lambda p: move_cost[p])[0]
        Q.remove(u)
        for neighbour in NEIGHBOURS[u]:
            must_stay = halite_map[u] * config.moveCost > ship_halite[u]
            alternative = move_cost[u] + (1 if not must_stay else 2)
            # TODO: consider enemies
            if alternative < move_cost[neighbour]:
                move_cost[neighbour] = alternative
                prev[neighbour] = u
                if must_stay:
                    ship_halite[neighbour] = ship_halite[u] + halite_map[u] * config.collectRate - halite_map[u] * (1 - config.collectRate) * config.moveCost
                else:
                    ship_halite[neighbour] = ship_halite[u] - halite_map[u] * config.moveCost

    return move_cost, prev


def agent(obs, config):
    player = obs.player
    step = obs.step
    size = config.size
    board = Board(obs, config)
    player_halite, shipyards, ships = obs.players[obs.player]

    if NEIGHBOURS is None:
        compute_neighbours(size)

    handle_special_steps(obs, config, board)
    spawn_ships(obs, config, board)
    move_ships(obs, config, board)
    return board.action
