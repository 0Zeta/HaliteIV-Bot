import logging
from enum import Enum
from random import random

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Shipyard, Ship

from haliteivbot.utils import *

logging.basicConfig(level=logging.WARNING)

env = make("halite", debug=True)

PARAMETERS = {
    'cell_score_enemy_halite': 0.5,
    'cell_score_neighbour_discount': 0.6566392486200908,
    'cell_score_ship_halite': 0.0006103202270414903,
    'conflict_map_alpha': 1.5680459520099566,
    'conflict_map_sigma': 0.8,
    'conflict_map_zeta': 0.8970120401490058,
    'convert_when_attacked_threshold': 477,
    'disable_hunting_till': 78,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.07032581227654462,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.11623455541829122,
    'end_return_extra_moves': 6,
    'end_start': 382,
    'ending_halite_threshold': 29,
    'hunting_halite_threshold': 6,
    'hunting_score_alpha': 1.0605617511903758,
    'hunting_score_beta': 2.123061010819677,
    'hunting_score_delta': 0.6769838756696485,
    'hunting_score_gamma': 0.9479697908505702,
    'hunting_threshold': 5.116440638204147,
    'map_blur_gamma': 0.47762345321439664,
    'map_blur_sigma': 0.5434126106796429,
    'max_halite_attack_shipyard': 247,
    'max_hunting_ships_per_direction': 1,
    'max_ship_advantage': 2,
    'max_shipyard_distance': 13,
    'min_mining_halite': 43,
    'min_ships': 14,
    'min_shipyard_distance': 5,
    'mining_score_alpha': 0.99,
    'mining_score_beta': 0.9151352865019396,
    'mining_score_delta': 5.546253335531781,
    'mining_score_gamma': 0.9909440276439139,
    'move_preference_base': 106,
    'move_preference_hunting': 108,
    'move_preference_mining': 123,
    'move_preference_return': 116,
    'return_halite': 1489,
    'ship_spawn_threshold': 0.16096483652829469,
    'ships_shipyards_threshold': 0.12227383861386135,
    'shipyard_abandon_dominance': -1.8513695799207515,
    'shipyard_conversion_threshold': 7.371982550837711,
    'shipyard_guarding_attack_probability': 1.0,
    'shipyard_guarding_min_dominance': 6.0006368518077595,
    'shipyard_min_dominance': 6.980137883872445,
    'shipyard_stop': 277,
    'spawn_min_dominance': 3.7347995207486466,
    'spawn_step_multiplier': 1,
    'spawn_till': 327
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
        self.rank = 0

        self.planned_moves = list()  # a list of positions where our ships will be in the next step
        self.planned_shipyards = list()
        self.ship_position_preferences = None
        self.ship_types = dict()
        self.mining_targets = dict()
        self.deposit_targets = dict()
        self.hunting_targets = dict()
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
        create_radius_lists(self.parameters['dominance_map_small_radius'],
                            self.parameters['dominance_map_medium_radius'])
        self.positions_in_reach_list = compute_positions_in_reach()
        self.farthest_directions = get_farthest_directions_matrix()

    def step(self, board: Board, obs):
        if self.me is None:
            self.player_id = board.current_player_id
            self.config = board.configuration
            self.size = self.config.size
        self.observation = obs
        self.me = board.current_player
        self.opponents = board.opponents
        self.ships = self.me.ships
        self.halite = self.me.halite
        self.ship_count = len(self.ships)
        self.shipyard_count = len(self.me.shipyards)
        # self.friendly_neighbour_count = {
        #    TO_INDEX[ship.cell.position]: self.get_friendly_neighbour_count(ship.cell) for ship in
        #    self.me.ships}

        self.average_halite_per_cell = sum([halite for halite in self.observation['halite']]) / self.size ** 2

        self.blurred_halite_map = get_blurred_halite_map(self.observation['halite'], self.parameters['map_blur_sigma'])

        self.shipyard_positions = []
        for shipyard in self.me.shipyards:
            self.shipyard_positions.append(TO_INDEX[shipyard.position])

        ranking = np.argsort([self.calculate_player_score(player) for player in [self.me] + self.opponents])[::-1]
        self.rank = int(np.where(ranking == 0)[0])

        map_presence_ranking = np.argsort(
            [self.calculate_player_map_presence(player) for player in [self.me] + self.opponents])[::-1]
        self.map_presence_rank = int(np.where(map_presence_ranking == 0)[0])
        # print("Map presence rank: " + str(self.map_presence_rank) + " Player rank: " + str(self.rank))

        # Compute distances to the next shipyard:
        if self.shipyard_count == 0:
            # There is no shipyard, but we still need to mine.
            self.shipyard_distances = [3] * self.size ** 2
        else:
            self.shipyard_distances = []
            for position in range(self.size ** 2):
                min_distance = float('inf')
                for shipyard in self.me.shipyards:  # TODO: consider planned shipyards
                    distance = get_distance(position, TO_INDEX[shipyard.position])
                    if distance < min_distance:
                        min_distance = distance
                self.shipyard_distances.append(min_distance)

        if len(self.me.ships) > 0:
            # self.blurred_conflict_map = get_blurred_conflict_map(self.me, self.opponents, self.parameters['conflict_map_alpha'], self.parameters['conflict_map_zeta'], self.parameters['conflict_map_sigma'])
            self.small_dominance_map = get_dominance_map(self.me, self.opponents,
                                                         self.parameters['dominance_map_small_sigma'], 'small')
            self.medium_dominance_map = get_dominance_map(self.me, self.opponents,
                                                          self.parameters['dominance_map_medium_sigma'], 'medium')
            # if board.step % 25 == 0:
            #     display_matrix(self.medium_dominance_map.reshape((self.size, self.size)))

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

        self.handle_special_steps(board)
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
            dominance = self.medium_dominance_map[TO_INDEX[shipyard.position]]
            if any(filter(lambda cell: cell.ship is not None and cell.ship.player_id != self.player_id,
                          get_neighbours(shipyard.cell))) and dominance >= self.parameters[
                'shipyard_guarding_min_dominance']:
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
            if dominance < self.parameters['spawn_min_dominance'] and board.step > 25 and self.shipyard_count > 1:
                continue
            if any(filter(lambda cell: cell.ship is None and cell.position not in self.planned_moves,
                          get_neighbours(shipyard.cell))):
                # Only spawn a ship if there are not too many own ships around the shipyard
                self.spawn_ship(shipyard)

    def reached_spawn_limit(self, board: Board):
        return board.step > self.parameters['spawn_till'] or ((self.ship_count >= sum(
            [len(player.ships) for player in board.players.values() if player.id != self.player_id]) / 3 +
                                                               self.parameters[
                                                                   'max_ship_advantage']) and self.ship_count >=
                                                              self.parameters['min_ships'])

    def move_ships(self, board: Board):
        if len(self.me.ships) == 0:
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
        self.assign_ship_targets(board)  # also converts some ships to hunting/returning ships
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
                    self.planned_shipyards.append(ship.position)
            else:
                target = self.positions_in_reach[position_index]
                if target != ship.position:
                    ship.next_action = get_direction_to_neighbour(TO_INDEX[ship.position], TO_INDEX[target])
                self.planned_moves.append(target)

    def assign_ship_targets(self, board: Board):
        # Adapted from https://www.kaggle.com/manavtrivedi/optimus-mine-agent
        if self.ship_count == 0:
            return

        ship_targets = {}
        mining_positions = []
        dropoff_positions = set()

        id_to_ship = {ship.id: ship for ship in self.mining_ships}

        halite_map = self.observation['halite']
        for position, halite in enumerate(halite_map):
            if halite >= self.parameters['min_mining_halite']:
                mining_positions.append(position)

        for shipyard in self.me.shipyards:
            shipyard_pos = TO_INDEX[shipyard.position]
            # Maybe only return to safe shiypards
            # Add each shipyard once for each distance to a ship
            for ship in self.mining_ships:
                dropoff_positions.add(shipyard_pos + get_distance(TO_INDEX[ship.position], shipyard_pos) * 1000)
        dropoff_positions = list(dropoff_positions)

        mining_scores = np.zeros((len(self.mining_ships), len(mining_positions) + len(dropoff_positions)))
        for ship_index, ship in enumerate(self.mining_ships):
            ship_pos = TO_INDEX[ship.position]
            for position_index, position in enumerate(mining_positions + dropoff_positions):
                if position >= 1000:
                    distance_to_shipyard = position // 1000
                    position = position % 1000
                    if distance_to_shipyard != get_distance(ship_pos, position):
                        mining_scores[ship_index, position_index] = -999999
                        continue

                mining_scores[ship_index, position_index] = self.calculate_mining_score(
                    ship_pos, position, halite_map[position],
                    self.blurred_halite_map[position], ship.halite)

        row, col = scipy.optimize.linear_sum_assignment(mining_scores, maximize=True)
        target_positions = mining_positions + dropoff_positions

        assigned_scores = [mining_scores[r][c] for r, c in zip(row, col)]
        hunting_threshold = np.mean(assigned_scores) - np.std(assigned_scores) * self.parameters[
            'hunting_score_alpha'] if len(assigned_scores) > 0 else -1

        for r, c in zip(row, col):
            if (mining_scores[r][c] < self.parameters['hunting_threshold'] or (
                    mining_scores[r][c] < hunting_threshold and self.mining_ships[r].halite <= self.parameters[
                'hunting_halite_threshold']) and self.map_presence_rank <= 1) and board.step > self.parameters[
                'disable_hunting_till']:
                continue
            if target_positions[c] >= 1000:
                ship_targets[self.mining_ships[r].id] = target_positions[c] % 1000
            else:
                ship_targets[self.mining_ships[r].id] = target_positions[c]

        # Convert indexed positions to points
        for ship_id, target_pos in ship_targets.items():
            ship = id_to_ship[ship_id]
            if target_pos in self.shipyard_positions:
                self.returning_ships.append(ship)
                self.mining_ships.remove(ship)
                self.ship_types[ship_id] = ShipType.RETURNING
                self.deposit_targets[ship_id] = Point.from_index(target_pos, self.size)
                logging.debug("Ship " + str(ship.id) + " returns.")
                continue
            self.mining_targets[ship_id] = Point.from_index(target_pos, self.size)
            logging.debug(
                "Assigning target " + str(Point.from_index(target_pos, self.size)) + " to ship " + str(ship.id))

        for ship in self.mining_ships:
            if ship.id not in self.mining_targets.keys():
                if ship.halite <= self.parameters['hunting_halite_threshold']:
                    self.hunting_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.HUNTING
                else:
                    self.returning_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.RETURNING

        encoded_dirs = [1, 2, 4, 8]
        possible_enemy_targets = [(dir, ship) for dir in encoded_dirs for ship in self.enemies for _ in
                                  range(self.parameters['max_hunting_ships_per_direction'])]
        hunting_scores = np.zeros(
            (len(self.hunting_ships), len(self.enemies) * 4 * self.parameters['max_hunting_ships_per_direction']))
        for ship_index, ship in enumerate(self.hunting_ships):
            ship_pos = TO_INDEX[ship.position]
            for enemy_index, (direction, enemy_ship) in enumerate(possible_enemy_targets):
                farthest_dirs = self.farthest_directions[ship_pos][TO_INDEX[enemy_ship.position]]
                if farthest_dirs == direction or (farthest_dirs - direction) in encoded_dirs:
                    hunting_scores[ship_index, enemy_index] = self.calculate_hunting_score(ship, enemy_ship)
                else:
                    hunting_scores[ship_index, enemy_index] = -999999

        row, col = scipy.optimize.linear_sum_assignment(hunting_scores, maximize=True)
        for r, c in zip(row, col):
            self.hunting_targets[self.hunting_ships[r].id] = possible_enemy_targets[c][1]

    def get_ship_type(self, ship: Ship, board: Board) -> ShipType:
        if ship.id in self.ship_types.keys():
            return self.ship_types[ship.id]
        if board.step >= 380:
            if self.shipyard_distances[TO_INDEX[ship.position]] + board.step + self.parameters[
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
        nearest_shipyard = self.get_nearest_shipyard(ship.position)
        distance_to_nearest_shipyard = calculate_distance(ship.position,
                                                          nearest_shipyard.position) if nearest_shipyard is not None else 20
        if board.step <= self.parameters['shipyard_stop'] and self.parameters[
            'min_shipyard_distance'] <= distance_to_nearest_shipyard <= \
                self.parameters[
                    'max_shipyard_distance'] and self.halite + ship.halite >= self.config.convert_cost:
            if (self.shipyard_count == 1) and (25 < board.step < 60) and (2 < distance_to_nearest_shipyard < 6):
                self.convert_to_shipyard(ship)  # Force an early second shipyard
                return
            if self.average_halite_per_cell / self.shipyard_count >= self.parameters[
                'shipyard_conversion_threshold'] \
                    and self.shipyard_count / self.ship_count < self.parameters['ships_shipyards_threshold'] \
                    and self.medium_dominance_map[TO_INDEX[ship.position]] >= self.parameters[
                'shipyard_min_dominance']:
                self.convert_to_shipyard(ship)
                logging.debug("Returning ship " + str(ship.id) + " converts to a shipyard at position " + str(
                    ship.position) + ".")
                return

        if self.ship_types[ship.id] == ShipType.ENDING:
            ship_pos = TO_INDEX[ship.position]
            destination_pos = TO_INDEX[destination]
            if get_distance(ship_pos, destination_pos) == 1:
                self.change_position_score(ship, destination, 9999)  # probably unnecessary
                logging.debug("Ending ship " + str(ship.id) + " returns to a shipyard at position " + str(destination))
                return

        self.prefer_moves(ship, navigate(ship.position, destination, self.size),
                          self.parameters['move_preference_return'])

    def handle_mining_ship(self, ship: Ship, board: Board):
        if ship.id not in self.mining_targets.keys():
            logging.critical("Mining ship " + str(ship.id) + " has no valid mining target.")
            return
        target = self.mining_targets[ship.id]
        if target != ship.position:
            preferred_moves = navigate(ship.position, target, self.size)
            for move in preferred_moves:
                position = (ship.position + move.to_point()) % self.size
                self.change_position_score(ship, position, self.parameters['move_preference_base'])
            for move in get_inefficient_directions(preferred_moves):
                position = (ship.position + move.to_point()) % self.size
                self.change_position_score(ship, position, -self.parameters['move_preference_base'])
        else:
            self.change_position_score(ship, target, self.parameters['move_preference_mining'])
            for move in DIRECTIONS:
                position = (ship.position + move.to_point()) % self.size
                self.change_position_score(ship, position, -self.parameters['move_preference_mining'])

    def handle_hunting_ship(self, ship: Ship, board: Board):
        if len(self.enemies) > 0:
            if ship.id in self.hunting_targets.keys():
                target = self.hunting_targets[ship.id]
            else:
                target = max(self.enemies, key=lambda enemy: self.calculate_hunting_score(ship, enemy))
            if target.halite > ship.halite:
                self.prefer_moves(ship, navigate(ship.position, target.position, self.size),
                                  # TODO: Should we choose the direction with the longest distance?
                                  self.parameters['move_preference_hunting'])

    def guard_shipyards(self, board: Board):
        for shipyard in self.me.shipyards:
            if shipyard.position in self.planned_moves:
                continue
            enemies = set(filter(lambda cell: cell.ship is not None and cell.ship.player_id != self.player_id,
                                 get_neighbours(shipyard.cell)))
            if len(enemies) > 0:
                dominance = self.medium_dominance_map[TO_INDEX[shipyard.position]]
                # TODO: maybe don't move on the shipyard if the dominance score is too low
                if shipyard.cell.ship is not None:
                    self.ship_types[shipyard.cell.ship.id] = ShipType.GUARDING
                    if self.halite < self.config.spawn_cost or dominance < self.parameters[
                        'shipyard_guarding_min_dominance'] or random() > self.parameters[
                        'shipyard_guarding_attack_probability']:  # TODO: mabe add step check
                        if dominance > self.parameters['shipyard_abandon_dominance']:
                            self.change_position_score(shipyard.cell.ship, shipyard.cell.position, 10000)
                            logging.debug("Ship " + str(shipyard.cell.ship.id) + " stays at position " + str(
                                shipyard.position) + " to guard a shipyard.")
                    else:
                        self.spawn_ship(shipyard)
                        for enemy in enemies:
                            self.change_position_score(shipyard.cell.ship, enemy.position,
                                                       500)  # equalize to crash into the ship even if that means we also lose our ship
                            self.attack_position(
                                enemy.position)  # Maybe also do this if we don't spawn a ship, but can move one to the shipyard
                else:
                    # TODO: add max halite the guarding ship can have
                    potential_guards = [neighbour.ship for neighbour in get_neighbours(shipyard.cell) if
                                        neighbour.ship is not None and neighbour.ship.id == self.player_id]
                    if len(potential_guards) > 0 and (
                            self.reached_spawn_limit(board) or self.halite < self.config.spawn_cost):
                        guard = sorted(potential_guards, key=lambda ship: ship.halite)[0]
                        self.change_position_score(guard, shipyard.position, 8000)
                        self.ship_types[guard.id] = ShipType.GUARDING
                        logging.debug("Ship " + str(guard.id) + " moves to position " + str(
                            shipyard.position) + " to protect a shipyard.")
                    elif self.halite > self.config.spawn_cost and (dominance >= self.parameters[
                        'shipyard_guarding_min_dominance'] or board.step <= 25 or self.shipyard_count == 1):
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
            ch = clip(ch, 0, 10)
        mining_steps = self.optimal_mining_steps[distance_from_ship - 1][distance_from_shipyard - 1][ch]
        return self.parameters['mining_score_gamma'] ** (distance_from_ship + mining_steps) * (
                self.parameters['mining_score_beta'] * ship_halite + (1 - 0.75 ** mining_steps) * min(
            1.02 ** (distance_from_ship) * halite_val, 500) * 1.02 ** mining_steps) / (
                       distance_from_ship + mining_steps + self.parameters[
                   'mining_score_alpha'] * distance_from_shipyard)

    def calculate_hunting_score(self, ship: Ship, enemy: Ship) -> float:
        d_halite = enemy.halite - ship.halite
        distance = calculate_distance(ship.position, enemy.position)
        return self.parameters['hunting_score_gamma'] ** distance * d_halite * (
                self.parameters['hunting_score_delta'] + self.parameters['hunting_score_beta'] * clip(
            self.medium_dominance_map[TO_INDEX[enemy.position]] + 30, 0, 60) / 60)

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
        score += self.small_dominance_map[TO_INDEX[cell.position]]
        # TODO: consider cell halite?
        return score * (1 + self.parameters['cell_score_ship_halite'] * ship.halite)

    def calculate_player_score(self, player):
        return player.halite + len(player.ships) * 500 + sum(
            [ship.halite / 4 for ship in player.ships] if len(player.ships) > 0 else [0])

    def calculate_player_map_presence(self, player):
        return len(player.ships) + len(player.shipyards)

    def prefer_moves(self, ship, directions, weight):
        for dir in directions:
            position = (ship.position + dir.to_point()) % self.size
            self.change_position_score(ship, position, weight)
        for dir in get_inefficient_directions(directions):
            position = (ship.position + dir.to_point()) % self.size
            self.change_position_score(ship, position, -weight)

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
        self.ship_types[ship.id] = ShipType.CONVERTING
        self.ship_position_preferences[self.ship_to_index[ship],
        len(self.position_to_index):-1] = 9999999  # TODO: fix the amount of available shipyard conversions
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

    def attack_position(self, position: Point):
        self.ship_position_preferences[:, self.position_to_index[position]][
            self.ship_position_preferences[:, self.position_to_index[position]] > -50] += 900


def agent(obs, config):
    global BOT
    if BOT is None:
        BOT = HaliteBot(PARAMETERS)
    board = Board(obs, config)
    logging.debug("Begin step " + str(board.step))
    return BOT.step(board, obs)
