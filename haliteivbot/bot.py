import logging
from enum import Enum

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Shipyard, Ship

from haliteivbot.utils import *

logging.basicConfig(level=logging.WARNING)

env = make("halite", debug=True)

PARAMETERS = {
    'spawn_till': 275,
    'spawn_step_multiplier': 3,
    'min_ships': 22,
    'ship_spawn_threshold': 0.7056203930791999,
    'shipyard_conversion_threshold': 1,
    'ships_shipyards_threshold': 0.3074067755476633,
    'shipyard_stop': 318,
    'min_shipyard_distance': 6,
    'min_mining_halite': 10,
    'convert_when_attacked_threshold': 304,
    'max_halite_attack_shipyard': 56,
    'mining_score_alpha': 0.95,
    'mining_score_beta': 0.95,
    'mining_score_gamma': 0.9894561554371855,
    'hunting_threshold': 0.79,
    'hunting_halite_threshold': 1,
    'disable_hunting_till': 7,
    'hunting_score_gamma': 0.8,
    'return_halite': 825,
    'max_ship_advantage': 3,
    'map_blur_sigma': 0.5779976863701278,
    'map_blur_gamma': 0.41473115809497146,
    'max_deposits_per_shipyard': 3,
    'end_return_extra_moves': 8,
    'end_start': 385,
    'ending_halite_threshold': 10,
    'cell_score_enemy_halite': 0.3,
    'cell_score_neighbour_discount': 0.6,
    'move_preference_base': 200,
    'move_preference_return': 210,
    'move_preference_mining': 250,
    'move_preference_hunting': 150,
    'cell_score_ship_halite': 0.0006
}

BOT = None


class ShipType(Enum):
    MINING = 1
    RETURNING = 2
    HUNTING = 3
    GUARDING = 4
    CONVERTING = 5
    ENDING = 6


class HaliteBot(object):

    def __init__(self, parameters):
        self.parameters = parameters
        self.config = None
        self.size = 21
        self.me = None
        self.player_id = 0

        self.halite = 5000
        self.ship_count = 1
        self.shipyard_count = 0
        self.shipyard_positions = []
        self.blurred_halite_map = None
        self.average_halite_per_cell = 0

        self.planned_moves = list()  # a list of positions where our ships will be in the next step
        self.planned_shipyards = list()
        self.ship_position_preferences = None
        self.ship_types = dict()
        self.mining_targets = dict()
        self.deposit_targets = dict()
        self.friendly_neighbour_count = dict()

        self.enemies = list()

        self.returning_ships = list()
        self.mining_ships = list()
        self.hunting_ships = list()
        self.endangered_ships = list()

        self.optimal_mining_steps = create_optimal_mining_steps_tensor(self.parameters['mining_score_alpha'],
                                                                       self.parameters['mining_score_beta'],
                                                                       self.parameters['mining_score_gamma'])
        create_navigation_lists(self.size)
        self.positions_in_reach_list = compute_positions_in_reach()

    def step(self, board: Board):
        if self.me is None:
            self.player_id = board.current_player_id
            self.config = board.configuration
            self.size = self.config.size

        self.me = board.current_player
        self.halite = self.me.halite
        self.ship_count = len(self.me.ships)
        self.shipyard_count = len(self.me.shipyards)
        self.friendly_neighbour_count = {
            ship.cell.position.to_index(self.size): self.get_friendly_neighbour_count(ship.cell) for ship in
            self.me.ships}

        self.average_halite_per_cell = sum([halite for halite in board.observation['halite']]) / self.size ** 2

        self.blurred_halite_map = get_blurred_halite_map(board.observation['halite'], self.parameters['map_blur_sigma'])

        self.shipyard_positions = []
        for shipyard in self.me.shipyards:
            self.shipyard_positions.append(shipyard.position.to_index(self.size))

        # Compute distances to the next shipyard:
        if self.shipyard_count == 0:
            # There is no shipyard, but we still need to mine.
            self.shipyard_distances = [3] * self.size ** 2
        else:
            self.shipyard_distances = []
            for position in range(self.size ** 2):
                min_distance = float('inf')
                for shipyard in self.me.shipyards:  # TODO: consider planned shipyards
                    distance = get_distance(position, shipyard.position.to_index(self.size))
                    if distance < min_distance:
                        min_distance = distance
                self.shipyard_distances.append(min_distance)

        self.planned_moves.clear()
        self.positions_in_reach = []
        for ship in self.me.ships:
            self.positions_in_reach.extend(self.positions_in_reach_list[ship.position])
        self.positions_in_reach = list(set(self.positions_in_reach))
        if board.step > self.parameters['end_start']:
            for shipyard in self.me.shipyards:
                if shipyard.position in self.positions_in_reach:
                    self.positions_in_reach.extend([shipyard.position for _ in range(3)])
        nb_positions_in_reach = len(self.positions_in_reach)
        nb_shipyard_conversions = self.halite // self.config.convert_cost + 1  # TODO: Ships with enough halite can also convert to a shipyard
        self.available_shipyard_conversions = nb_shipyard_conversions - 1  # without cargo
        self.ship_to_index = {ship: ship_index for ship_index, ship in enumerate(self.me.ships)}
        self.position_to_index = dict()
        for position_index, position in enumerate(self.positions_in_reach):
            if position in self.position_to_index.keys():
                self.position_to_index[position].append(position_index)
            else:
                self.position_to_index[position] = [position_index]
        self.ship_position_preferences = np.full(
            shape=(self.ship_count, nb_positions_in_reach + nb_shipyard_conversions),
            fill_value=-999999)  # positions + convert "positions"
        for ship_index, ship in enumerate(self.me.ships):
            if ship.halite >= self.parameters['convert_when_attacked_threshold']:
                self.ship_position_preferences[ship_index,
                nb_positions_in_reach:nb_positions_in_reach + self.available_shipyard_conversions] = -self.parameters[
                    'convert_when_attacked_threshold']
            if (ship.halite + self.halite) // self.config.convert_cost > self.available_shipyard_conversions:
                self.ship_position_preferences[
                    ship_index, nb_positions_in_reach + self.available_shipyard_conversions] = -self.parameters[
                    'convert_when_attacked_threshold']
            for position in self.positions_in_reach_list[ship.position]:
                self.ship_position_preferences[
                    ship_index, self.position_to_index[position]] = self.calculate_cell_score(ship,
                                                                                              board.cells[position])

        self.planned_shipyards.clear()
        self.ship_types.clear()
        self.mining_targets.clear()
        self.deposit_targets.clear()
        self.enemies = [ship for player in board.players.values() for ship in player.ships if
                        player.id != self.player_id]

        if self.handle_special_steps(board):
            return  # don't execute the functions below
        self.guard_shipyards(board)
        self.move_ships(board)
        self.spawn_ships(board)
        return self.me.next_actions

    def handle_special_steps(self, board: Board) -> bool:
        step = board.step
        if step == 0:
            # Immediately spawn a shipyard
            self.convert_to_shipyard(self.me.ships[0])
            logging.debug("Ship " + str(self.me.ships[0].id) + " converts to a shipyard at the start of the game.")
            return True
        return False

    def spawn_ships(self, board: Board):
        # Spawn a ship if there are none left
        if len(self.me.ships) == 0:
            if len(self.me.shipyards) > 0:
                self.spawn_ship(self.me.shipyards[0])

        for shipyard in self.me.shipyards:
            if self.halite < self.config.spawn_cost:
                return
            if shipyard.position in self.planned_moves:
                continue
            if any(filter(lambda cell: cell.ship is not None and cell.ship.player_id != self.player_id,
                          get_neighbours(shipyard.cell))):
                # There is an enemy ship next to the shipyard.
                self.spawn_ship(shipyard)
                continue
            if self.halite < self.config.spawn_cost + board.step * self.parameters['spawn_step_multiplier']:
                continue
            if self.reached_spawn_limit(board):
                continue
            if self.ship_count >= self.parameters['min_ships'] and self.average_halite_per_cell / self.ship_count < \
                    self.parameters['ship_spawn_threshold']:
                continue
            if any(filter(lambda cell: cell.ship is None and cell.position not in self.planned_moves,
                          get_neighbours(shipyard.cell))):
                # Only spawn a ship if there are not too many own ships around the shipyard
                self.spawn_ship(shipyard)

    def reached_spawn_limit(self, board: Board):
        return board.step > self.parameters['spawn_till'] or ((self.ship_count >= max(
            [len(player.ships) for player in board.players.values() if player.id != self.player_id]) + self.parameters[
                                                                   'max_ship_advantage']) and self.ship_count >=
                                                              self.parameters['min_ships'])

    def move_ships(self, board: Board):
        if self.ship_count == 0:
            return
        # TODO: remove these lists
        self.returning_ships.clear()
        self.mining_ships.clear()
        self.hunting_ships.clear()

        if self.shipyard_count == 0:
            ship = max(self.me.ships, key=lambda ship: ship.halite)
            if ship.halite + self.halite >= self.config.convert_cost:
                self.convert_to_shipyard(ship)
                self.ship_types[ship.id] = ShipType.CONVERTING

        for ship in self.me.ships:
            ship_type = self.get_ship_type(ship, board)
            self.ship_types[ship.id] = ship_type
            if ship_type == ShipType.MINING:
                self.mining_ships.append(ship)
            elif ship_type == ShipType.RETURNING or ship_type == ShipType.ENDING:
                self.returning_ships.append(ship)

        ships = self.me.ships.copy()
        self.assign_ship_targets(board)  # also converts some ships tp hunting/returning ships
        while len(ships) > 0:
            ship = ships[0]
            ship_type = self.ship_types[ship.id]
            if ship_type == ShipType.MINING:
                self.handle_mining_ship(ship, board)
            elif ship_type == ShipType.RETURNING or ship_type == ShipType.ENDING:
                self.handle_returning_ship(ship, board)
            elif ship_type == ShipType.HUNTING:
                self.handle_hunting_ship(ship, board)

            ships.remove(ship)

        row, col = scipy.optimize.linear_sum_assignment(self.ship_position_preferences, maximize=True)
        for ship_index, position_index in zip(row, col):
            ship = self.me.ships[ship_index]
            if position_index >= len(self.positions_in_reach):
                # The ship wants to convert to a shipyard
                if self.ship_position_preferences[ship_index, position_index] > 5000:
                    # immediately convert
                    ship.next_action = ShipAction.CONVERT  # self.halite has already been reduced
                else:
                    ship.next_action = ShipAction.CONVERT
                    self.halite -= self.config.convert_cost
            else:
                target = self.positions_in_reach[position_index]
                if target != ship.position:
                    ship.next_action = get_direction_to_neighbour(ship.position.to_index(self.size),
                                                                  target.to_index(self.size))
                self.planned_moves.append(target)

    def assign_ship_targets(self, board: Board):
        # Adapted from https://www.kaggle.com/manavtrivedi/optimus-mine-agent
        if self.ship_count == 0:
            return

        ship_targets = {}
        mining_positions = []
        dropoff_positions = []

        id_to_ship = {ship.id: ship for ship in self.mining_ships}

        halite_map = board.observation['halite']
        for position, halite in enumerate(halite_map):
            if halite >= self.parameters['min_mining_halite']:
                mining_positions.append(position)

        for shipyard in self.me.shipyards:
            for _ in range(self.parameters['max_deposits_per_shipyard']):
                dropoff_positions.append(shipyard.position.to_index(self.size))

        mining_scores = np.zeros((len(self.mining_ships), len(mining_positions) + len(dropoff_positions)))
        for ship_index, ship in enumerate(self.mining_ships):
            for position_index, position in enumerate(mining_positions + dropoff_positions):
                mining_scores[ship_index, position_index] = self.calculate_mining_score(
                    ship.position.to_index(self.size), position, halite_map[position],
                    self.blurred_halite_map[position], ship.halite)

        row, col = scipy.optimize.linear_sum_assignment(mining_scores, maximize=True)
        target_positions = mining_positions + dropoff_positions

        for r, c in zip(row, col):
            if mining_scores[r][c] < self.parameters['hunting_threshold'] and board.step > self.parameters[
                'disable_hunting_till']:
                ship = self.mining_ships[r]
                if ship.halite < self.parameters['hunting_halite_threshold']:
                    self.hunting_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.HUNTING
                else:
                    self.returning_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.RETURNING
                continue
            ship_targets[self.mining_ships[r].id] = target_positions[c]

        # Convert indexed positions to points
        for ship_id, target_pos in ship_targets.items():
            ship = id_to_ship[ship_id]
            if target_pos in self.shipyard_positions:
                self.returning_ships.append(ship)
                self.ship_types[ship_id] = ShipType.RETURNING
                self.deposit_targets[ship_id] = Point.from_index(target_pos, self.size)
                logging.debug("Ship " + str(ship.id) + " returns.")
                continue
            self.mining_targets[ship_id] = Point.from_index(target_pos, self.size)
            logging.debug(
                "Assigning target " + str(Point.from_index(target_pos, self.size)) + " to ship " + str(ship.id))

    def get_ship_type(self, ship: Ship, board: Board) -> ShipType:
        if ship.id in self.ship_types.keys():
            return self.ship_types[ship.id]
        if board.step >= 380:
            if self.shipyard_distances[ship.position.to_index(self.size)] + board.step + self.parameters[
                'end_return_extra_moves'] >= 400 and ship.halite >= self.parameters['ending_halite_threshold']:
                return ShipType.ENDING
        if ship.halite >= self.parameters['return_halite']:
            return ShipType.RETURNING
        else:
            return ShipType.MINING

    def handle_returning_ship(self, ship: Ship, board: Board):
        if self.ship_types[ship.id] == ShipType.ENDING:
            destination = self.get_nearest_shipyard(ship.position)
            if destination is not None:
                destination = destination.position
        else:
            if ship.id in self.deposit_targets.keys():
                destination = self.deposit_targets[ship.id]
            else:
                destination = self.get_nearest_shipyard(ship.position)
                if destination is not None:
                    destination = destination.position
        if destination is None:
            if self.halite + ship.halite >= self.config.convert_cost:
                if self.shipyard_count == 0:
                    self.convert_to_shipyard(ship)
                    logging.debug("Returning ship " + str(
                        ship.id) + " has no shipyard and converts to one at position " + str(ship.position) + ".")
                    return
                else:
                    destination = board.cells[self.planned_shipyards[0]].position
            else:
                # TODO: mine
                logging.debug("Returning ship " + str(ship.id) + " has no shipyard to go to.")
                return
        if board.step <= self.parameters['shipyard_stop'] and calculate_distance(ship.position, destination) >= \
                self.parameters[
                    'min_shipyard_distance'] and self.halite >= self.config.convert_cost:  # TODO: consider ship cargo
            if self.average_halite_per_cell / self.shipyard_count >= self.parameters[
                'shipyard_conversion_threshold'] \
                    and self.shipyard_count / self.ship_count < self.parameters['ships_shipyards_threshold']:
                self.convert_to_shipyard(ship)
                logging.debug("Returning ship " + str(ship.id) + " converts to a shipyard at position " + str(
                    ship.position) + ".")
                return

        if self.ship_types[ship.id] == ShipType.ENDING:
            ship_pos = ship.position.to_index(self.size)
            destination_pos = destination.to_index(self.size)
            if get_distance(ship_pos, destination_pos) == 1:
                self.change_position_score(ship, destination, 9999)  # probably unnecessary
                logging.debug("Ending ship " + str(ship.id) + " returns to a shipyard at position " + str(destination))
                return

        preferred_moves = navigate(ship.position, destination, self.size)
        for move in preferred_moves:
            position = (ship.position + move.to_point()) % self.size
            self.change_position_score(ship, position, self.parameters['move_preference_return'])

    def handle_mining_ship(self, ship: Ship, board: Board):
        target = self.mining_targets[ship.id]
        if target != ship.position:
            preferred_moves = navigate(ship.position, target, self.size)
            for move in preferred_moves:
                position = (ship.position + move.to_point()) % self.size
                self.change_position_score(ship, position, self.parameters['move_preference_base'])
        else:
            self.change_position_score(ship, target, self.parameters['move_preference_mining'])

    def handle_hunting_ship(self, ship: Ship, board: Board):
        if len(self.enemies) > 0:
            target = max(self.enemies, key=lambda enemy: self.calculate_hunting_score(ship, enemy))
            if target.halite > ship.halite:
                preferred_moves = navigate(ship.position, target.position, self.size)
                for move in preferred_moves:
                    position = (ship.position + move.to_point()) % self.size
                    self.change_position_score(ship, position, self.parameters['move_preference_hunting'])

    def guard_shipyards(self, board: Board):
        for shipyard in self.me.shipyards:
            if shipyard.position in self.planned_moves:
                continue
            enemies = set(filter(lambda cell: cell.ship is not None and cell.ship.player_id != self.player_id,
                                 get_neighbours(shipyard.cell)))
            if len(enemies) > 0:
                if shipyard.cell.ship is not None:
                    self.ship_types[shipyard.cell.ship.id] = ShipType.GUARDING
                    # TODO: maybe attack the enemy ship
                    self.change_position_score(shipyard.cell.ship, shipyard.cell.position, 10000)
                    logging.debug("Exploring ship " + str(shipyard.cell.ship.id) + " stays at position " + str(
                        shipyard.position) + " to guard a shipyard.")
                else:
                    potential_guards = [neighbour.ship for neighbour in get_neighbours(shipyard.cell) if
                                        neighbour.ship is not None and neighbour.ship.id == self.player_id]
                    if len(potential_guards) > 0 and (
                            self.reached_spawn_limit(board) or self.halite < self.config.spawn_cost):
                        guard = sorted(potential_guards, key=lambda ship: ship.halite)[0]
                        self.change_position_score(guard, shipyard.position, 8000)
                        self.ship_types[guard.id] = ShipType.GUARDING
                        logging.debug("Ship " + str(guard.id) + " moves to position " + str(
                            shipyard.position) + " to protect a shipyard.")
                    elif self.halite > self.config.spawn_cost:
                        self.spawn_ship(shipyard)
                    else:
                        logging.debug("Shipyard " + str(shipyard.id) + " cannot be protected.")

    def calculate_mining_score(self, ship_position: int, cell_position: int, halite, blurred_halite,
                               ship_halite) -> float:
        # TODO: account for enemies
        distance_from_ship = get_distance(ship_position, cell_position)
        distance_from_shipyard = self.shipyard_distances[cell_position]
        halite_val = (1 - self.parameters['map_blur_gamma'] ** distance_from_ship) * blurred_halite + self.parameters[
            'map_blur_gamma'] ** distance_from_ship * halite
        if distance_from_shipyard > 20:
            # There is no shipyard.
            distance_from_shipyard = 20
        if ship_halite == 0:
            ch = 0
        elif halite_val == 0:
            ch = 10
        else:
            ch = int(math.log(ship_halite / halite_val) * 2.5 + 5.5)
            ch = np.clip(ch, 0, 10)
        mining_steps = self.optimal_mining_steps[distance_from_ship - 1][distance_from_shipyard - 1][ch]
        return self.parameters['mining_score_gamma'] ** (distance_from_ship + mining_steps) * (
                self.parameters['mining_score_beta'] * ship_halite + (1 - 0.75 ** mining_steps) * min(
            1.02 ** (distance_from_ship) * halite_val, 500) * 1.02 ** mining_steps) / (
                       distance_from_ship + mining_steps + self.parameters[
                   'mining_score_alpha'] * distance_from_shipyard)

    def calculate_hunting_score(self, ship: Ship, enemy: Ship) -> float:
        d_halite = enemy.halite - ship.halite
        distance = calculate_distance(ship.position, enemy.position)
        return self.parameters['hunting_score_gamma'] ** distance * d_halite / distance

    def calculate_cell_score(self, ship: Ship, cell: Cell) -> float:
        score = 0
        if cell.position in self.planned_moves:
            score -= 1500
            return score
        if cell.shipyard is not None and cell.shipyard.player_id != self.player_id:
            if ship.halite > self.parameters['max_halite_attack_shipyard']:
                score -= (500 + ship.halite)
            # TODO: reward attacking an enemy shipyard?
        if cell.ship is not None and cell.ship.player_id != self.player_id:
            if cell.ship.halite < ship.halite:
                score -= (500 + ship.halite)
            elif cell.ship.halite == ship.halite:
                score -= 500
            else:
                score += cell.ship.halite * self.parameters['cell_score_enemy_halite']
        neighbour_value = 0
        for neighbour in get_neighbours(cell):
            if neighbour.ship is not None and neighbour.ship.player_id != self.player_id:
                if neighbour.ship.halite < ship.halite:  # We really don't want to go to that cell unless it's necessary.
                    neighbour_value = -(500 + ship.halite) * self.parameters['cell_score_neighbour_discount']
                    break
                elif neighbour.ship.halite == ship.halite:
                    neighbour_value -= 500 * self.parameters['cell_score_neighbour_discount']
                else:
                    neighbour_value += neighbour.ship.halite * self.parameters['cell_score_enemy_halite'] * \
                                       self.parameters['cell_score_neighbour_discount']
        score += neighbour_value
        # TODO: consider cell halite?
        return score * (1 + self.parameters['cell_score_ship_halite'] * ship.halite)

    def change_position_score(self, ship: Ship, position: Point, delta: float):
        self.ship_position_preferences[self.ship_to_index[ship], self.position_to_index[position]] += delta

    def get_nearest_shipyard(self, pos: Point):
        min_distance = float('inf')
        nearest_shipyard = None
        for shipyard in self.me.shipyards:
            distance = calculate_distance(pos, shipyard.position)
            if distance < min_distance:
                min_distance = distance
                nearest_shipyard = shipyard
        return nearest_shipyard

    def get_friendly_neighbour_count(self, cell: Cell):
        return sum(1 for _ in
                   filter(lambda n: n.ship is not None and n.ship.player_id == self.player_id, get_neighbours(cell)))

    def convert_to_shipyard(self, ship: Ship):
        assert self.halite + ship.halite >= self.config.convert_cost
        # ship.next_action = ShipAction.CONVERT
        self.ship_position_preferences[self.ship_to_index[ship],
        len(self.position_to_index):self.available_shipyard_conversions] = 9999999
        self.halite += ship.halite
        self.halite -= self.config.convert_cost
        self.ship_count -= 1
        self.shipyard_count += 1
        self.planned_shipyards.append(ship.position)

    def spawn_ship(self, shipyard: Shipyard):
        assert self.halite >= self.config.spawn_cost
        shipyard.next_action = ShipyardAction.SPAWN
        self.planned_moves.append(shipyard.position)
        self.halite -= self.config.spawn_cost
        self.ship_count += 1
        logging.debug("Spawning ship on position " + str(shipyard.position) + " (shipyard " + str(shipyard.id) + ")")


@board_agent
def agent(board: Board):
    global BOT
    if BOT is None:
        BOT = HaliteBot(PARAMETERS)

    logging.debug("Begin step " + str(board.step))
    BOT.step(board)
