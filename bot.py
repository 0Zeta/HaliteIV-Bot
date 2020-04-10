from kaggle_environments import evaluate, make
from kaggle_environments.envs.halite.halite import Board, get_to_pos
from random import choice, shuffle


env = make("halite", debug=True)


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


def agent(obs, config):
    player = obs.player
    step = obs.step
    size = config.size
    board = Board(obs, config)
    player_halite, shipyards, ships = obs.players[obs.player]

    handle_special_steps(obs, config, board)
    spawn_ships(obs, config, board)
    print(board.ships_by_uid)
    move_ships(obs, config, board)
    return board.action
