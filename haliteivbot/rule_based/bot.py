import logging
from enum import Enum
from math import floor, ceil
from random import random

from kaggle_environments.envs.halite.helpers import Shipyard, Ship

from haliteivbot.rule_based.utils import *

logging.basicConfig(level=logging.WARNING)

PARAMETERS = {
    'cargo_map_halite_norm': 197,
    'cell_score_dominance': 2.0690023066592538,
    'cell_score_enemy_halite': 0.3208283626314189,
    'cell_score_neighbour_discount': 0.7,
    'cell_score_ship_halite': 0.0006924718210075495,
    'convert_when_attacked_threshold': 520,
    'disable_hunting_till': 81,
    'dominance_map_halite_clip': 340,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.01,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.15815686031545878,
    'end_return_extra_moves': 7,
    'end_start': 377,
    'ending_halite_threshold': 9,
    'hunting_avg_halite_threshold': 25,
    'hunting_halite_threshold': 0.05,
    'hunting_min_ships': 16,
    'hunting_score_alpha': 0.6,
    'hunting_score_beta': 3,
    'hunting_score_cargo_clip': 2.434932143755778,
    'hunting_score_delta': 0.8709006820260277,
    'hunting_score_gamma': 0.9509334468781269,
    'hunting_score_iota': 0.5105732890493775,
    'hunting_score_kappa': 0.39357038462375626,
    'hunting_score_zeta': 4,
    'hunting_threshold': 12.12833619658105,
    'hunting_score_ship_bonus': 150,
    'hunting_score_halite_norm': 100,
    'map_blur_gamma': 0.6534115332552308,
    'map_blur_sigma': 0.7762017145865703,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 30,
    'max_shipyard_distance': 7,
    'max_shipyards': 7,
    'min_mining_halite': 32,
    'min_ships': 30,
    'min_shipyard_distance': 6,
    'mining_score_alpha': 1.2,
    'mining_score_beta': 0.85,
    'mining_score_dominance_clip': 4,
    'mining_score_dominance_norm': 1,
    'mining_score_gamma': 0.96,
    'mining_score_farming_penalty': 0.05,
    'move_preference_base': 100,
    'move_preference_block_shipyard': -100,
    'move_preference_hunting': 107,
    'move_preference_longest_axis': 10,
    'move_preference_mining': 130,
    'move_preference_return': 116,
    'move_preference_stay_on_shipyard': -125,
    'farming_end': 350,
    'return_halite': 1970,
    'ship_spawn_threshold': 1.4001702394113038,
    'ships_shipyards_threshold': 0.25,
    'shipyard_abandon_dominance': -45,
    'shipyard_conversion_threshold': 5,
    'shipyard_guarding_attack_probability': 0.1,
    'shipyard_guarding_min_dominance': -25,
    'shipyard_min_population': 0.7,
    'shipyard_min_dominance': 5,
    'shipyard_start': 45,
    'shipyard_stop': 260,
    'spawn_min_dominance': 3.528656727561098,
    'spawn_till': 260
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
        self.farming_radius_list = create_radius_list(ceil(self.parameters['max_shipyard_distance'] / 2))
        self.distances = get_distance_matrix()
        self.positions_in_reach_list = compute_positions_in_reach()
        self.farthest_directions_indices = get_farthest_directions_matrix()
        self.farthest_directions = get_farthest_directions_list()

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
        self.step_count = board.step
        self.ship_count = len(self.ships)
        self.shipyard_count = len(self.me.shipyards)

        self.average_halite_per_cell = sum([halite for halite in self.observation['halite']]) / self.size ** 2
        self.average_halite_population = sum(
            [1 if halite > 0 else 0 for halite in self.observation['halite']]) / self.size ** 2
        self.nb_cells_in_farming_radius = len(self.farming_radius_list[0])

        self.blurred_halite_map = get_blurred_halite_map(self.observation['halite'], self.parameters['map_blur_sigma'])

        self.shipyard_positions = []
        for shipyard in self.me.shipyards:
            self.shipyard_positions.append(TO_INDEX[shipyard.position])

        players = [self.me] + self.opponents
        ranking = np.argsort([self.calculate_player_score(player) for player in players])[::-1]
        self.rank = int(np.where(ranking == 0)[0])
        self.player_ranking = dict()

        map_presence_ranks = np.argsort(
            [self.calculate_player_map_presence(player) for player in players])[::-1]
        self.map_presence_rank = int(np.where(map_presence_ranks == 0)[0])
        self.map_presence_ranking = dict()

        halite_ranks = np.argsort([player.halite for player in players])[::-1]
        self.halite_ranking = dict()

        for i, player in enumerate(players):
            self.player_ranking[player.id] = int(np.where(ranking == i)[0])
            self.map_presence_ranking[player.id] = int(np.where(map_presence_ranks == i)[0])
            self.halite_ranking[player.id] = int(np.where(halite_ranks == i)[0])

        self.farming_positions = []
        # Compute distances to the next shipyard:
        if self.shipyard_count == 0:
            # There is no shipyard, but we still need to mine.
            self.shipyard_distances = [3] * self.size ** 2
        else:
            # Also compute farming positions
            required_in_range = min(3, max(2, len(self.shipyard_positions)))
            self.shipyard_distances = []
            for position in range(self.size ** 2):
                nb_in_farming_range = 0
                min_distance = float('inf')
                for shipyard_position in self.shipyard_positions:  # TODO: consider planned shipyards
                    distance = get_distance(position, shipyard_position)
                    if position not in self.shipyard_positions and distance <= self.parameters['max_shipyard_distance']:
                        nb_in_farming_range += 1
                        if nb_in_farming_range == required_in_range:
                            self.farming_positions.append(position)
                    if distance < min_distance:
                        min_distance = distance
                self.shipyard_distances.append(min_distance)

        if len(self.me.ships) > 0:
            self.small_dominance_map = get_dominance_map(self.me, self.opponents,
                                                         self.parameters['dominance_map_small_sigma'], 'small',
                                                         self.parameters['dominance_map_halite_clip'])
            self.medium_dominance_map = get_dominance_map(self.me, self.opponents,
                                                          self.parameters['dominance_map_medium_sigma'], 'medium',
                                                          self.parameters['dominance_map_halite_clip'])
            self.cargo_map = get_cargo_map(self.me.ships, self.me.shipyards, self.parameters['cargo_map_halite_norm'])

        self.planned_moves.clear()
        self.spawn_limit_reached = self.reached_spawn_limit(board)
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
            if ship.cell.shipyard is not None:
                self.ship_position_preferences[ship_index, self.position_to_index[ship.position]] += self.parameters[
                    'move_preference_stay_on_shipyard']

        self.planned_shipyards.clear()
        self.ship_types.clear()
        self.mining_targets.clear()
        self.deposit_targets.clear()
        self.enemies = [ship for player in board.players.values() for ship in player.ships if
                        player.id != self.player_id]
        enemy_cargo = sorted([ship.halite for ship in self.enemies])
        self.hunting_halite_threshold = enemy_cargo[
            floor(len(enemy_cargo) * self.parameters['hunting_halite_threshold'])] if len(enemy_cargo) > 0 else 0

        self.handle_special_steps(board)
        self.guard_shipyards(board)
        self.build_shipyards()
        self.move_ships(board)
        self.spawn_ships(board)
        return self.me.next_actions

    def handle_special_steps(self, board: Board) -> bool:
        step = board.step
        if step == 0:
            # Immediately spawn a shipyard
            self.convert_to_shipyard(self.me.ships[0])
            logging.debug("Ship " + str(self.me.ships[0].id) + " converts to a shipyard at the start of the game.")
            return False

    def build_shipyards(self):
        if self.parameters['shipyard_start'] > self.step_count or self.step_count > self.parameters['shipyard_stop']:
            return
        for ship in sorted(self.me.ships, key=lambda s: s.halite, reverse=True):
            if self.should_convert(ship):
                self.convert_to_shipyard(ship)
                return  # only build one shipyard per step

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
                'shipyard_guarding_min_dominance'] and self.step_count < 370:
                # There is an enemy ship next to the shipyard.
                self.spawn_ship(shipyard)
                continue
            if self.halite < self.config.spawn_cost + (0 if self.step_count < 70 else len(
                    self.me.shipyards) * self.config.spawn_cost):  # TODO: remove the halite reserve in favor of guarding ships
                continue
            if self.reached_spawn_limit(board):
                continue
            if self.ship_count >= self.parameters['min_ships'] and self.average_halite_per_cell / self.ship_count < \
                    self.parameters['ship_spawn_threshold']:
                continue
            if dominance < self.parameters['spawn_min_dominance'] and board.step > 75 and self.shipyard_count > 1:
                continue
            self.spawn_ship(shipyard)

    def reached_spawn_limit(self, board: Board):
        return board.step > self.parameters['spawn_till'] or ((self.ship_count >= max(
            [len(player.ships) for player in board.players.values() if player.id != self.player_id]) +
                                                               self.parameters[
                                                                   'max_ship_advantage']) and self.ship_count >=
                                                              self.parameters['min_ships'])

    def move_ships(self, board: Board):
        if len(self.me.ships) == 0:
            return
        self.returning_ships.clear()
        self.mining_ships.clear()
        self.hunting_ships.clear()

        if self.shipyard_count == 0:
            ship = max(self.me.ships, key=lambda ship: ship.halite)  # TODO: choose the ship with the safest position
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

        self.mining_score_beta = self.parameters['mining_score_beta'] if self.step_count >= 60 else 0.65 * \
                                                                                                    self.parameters[
                                                                                                        'mining_score_beta']  # Don't return too often early in the game
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
        logging.debug("assigned mining scores mean: {}".format(np.mean(assigned_scores)))
        hunting_threshold = np.mean(assigned_scores) - np.std(assigned_scores) * self.parameters[
            'hunting_score_alpha'] if len(assigned_scores) > 0 else -1
        hunting_enabled = board.step > self.parameters['disable_hunting_till'] and (self.ship_count >= self.parameters[
            'hunting_min_ships'] or board.step > self.parameters['spawn_till']) and self.average_halite_per_cell <= \
                          self.parameters['hunting_avg_halite_threshold']

        for r, c in zip(row, col):
            if (mining_scores[r][c] < self.parameters['hunting_threshold'] or (
                    mining_scores[r][c] < hunting_threshold and self.mining_ships[
                r].halite <= self.hunting_halite_threshold)) and hunting_enabled:
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
                if ship.halite <= self.hunting_halite_threshold:
                    self.hunting_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.HUNTING
                else:
                    self.returning_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.RETURNING

        encoded_dirs = [1, 2, 4, 8]
        possible_enemy_targets = [(dir, ship) for dir in encoded_dirs for ship in self.enemies for _ in
                                  range(self.parameters['max_hunting_ships_per_direction'])]
        logging.debug("Number of hunting ships: {}".format(len(self.hunting_ships)))
        hunting_scores = np.zeros(
            (len(self.hunting_ships), len(self.enemies) * 4 * self.parameters['max_hunting_ships_per_direction']))
        for ship_index, ship in enumerate(self.hunting_ships):
            ship_pos = TO_INDEX[ship.position]
            for enemy_index, (direction, enemy_ship) in enumerate(possible_enemy_targets):
                farthest_dirs = self.farthest_directions_indices[ship_pos][TO_INDEX[enemy_ship.position]]
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
        if board.step >= self.parameters['end_start']:
            if self.shipyard_distances[TO_INDEX[ship.position]] + board.step + self.parameters[
                'end_return_extra_moves'] >= 398 and ship.halite >= self.parameters['ending_halite_threshold']:
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

        ship_pos = TO_INDEX[ship.position]
        destination_pos = TO_INDEX[destination]
        if self.ship_types[ship.id] == ShipType.ENDING:
            if get_distance(ship_pos, destination_pos) == 1:
                self.change_position_score(ship, destination, 9999)  # probably unnecessary
                logging.debug("Ending ship " + str(ship.id) + " returns to a shipyard at position " + str(destination))
                return

        self.prefer_moves(ship, navigate(ship.position, destination, self.size),
                          self.farthest_directions[ship_pos][destination_pos],
                          self.parameters['move_preference_return'])

    def handle_mining_ship(self, ship: Ship, board: Board):
        if ship.id not in self.mining_targets.keys():
            logging.critical("Mining ship " + str(ship.id) + " has no valid mining target.")
            return
        target = self.mining_targets[ship.id]
        ship_pos = TO_INDEX[ship.position]
        target_pos = TO_INDEX[target]
        if target != ship.position:
            self.prefer_moves(ship, nav(ship_pos, target_pos), self.farthest_directions[ship_pos][target_pos],
                              self.parameters['move_preference_base'])
            if self.shipyard_distances[ship_pos] == 1:
                for neighbour in get_neighbours(ship.cell):
                    if neighbour.shipyard is not None and neighbour.shipyard.player_id == self.player_id:
                        if self.step_count <= 11 and self.halite >= self.config.spawn_cost:
                            # We really want to get our ships out
                            self.change_position_score(ship, neighbour.position,
                                                       5 * self.parameters['move_preference_stay_on_shipyard'])
                        else:
                            self.change_position_score(ship, neighbour.position,
                                                       self.parameters['move_preference_stay_on_shipyard'])

        else:
            self.change_position_score(ship, target, self.parameters['move_preference_mining'])
            self.prefer_moves(ship, [], [], self.parameters['move_preference_mining'])

    def handle_hunting_ship(self, ship: Ship, board: Board):
        if len(self.enemies) > 0:
            if ship.id in self.hunting_targets.keys():
                target = self.hunting_targets[ship.id]
            else:
                target = max(self.enemies, key=lambda enemy: self.calculate_hunting_score(ship, enemy))
            if target.halite > ship.halite:
                ship_position = ship.position
                target_position = target.position
                self.prefer_moves(ship, navigate(ship_position, target_position, self.size),
                                  self.farthest_directions[TO_INDEX[ship_position]][TO_INDEX[target_position]],
                                  self.parameters['move_preference_hunting'])

    def guard_shipyards(self, board: Board):
        for shipyard in self.me.shipyards:
            if shipyard.position in self.planned_moves:
                continue
            enemies = set(filter(lambda cell: cell.ship is not None and cell.ship.player_id != self.player_id,
                                 get_neighbours(shipyard.cell)))
            max_halite = min([cell.ship.halite for cell in enemies]) if len(enemies) > 0 else 500

            if len(enemies) > 0:
                dominance = self.medium_dominance_map[TO_INDEX[shipyard.position]]
                # TODO: maybe don't move on the shipyard if the dominance score is too low
                if shipyard.cell.ship is not None:
                    self.ship_types[shipyard.cell.ship.id] = ShipType.GUARDING
                    if self.halite < self.config.spawn_cost or (
                            self.step_count > self.parameters['spawn_till'] and self.shipyard_count > 1) or dominance < \
                            self.parameters[
                                'shipyard_guarding_min_dominance'] or random() > self.parameters[
                        'shipyard_guarding_attack_probability'] or self.step_count >= 365:
                        if dominance > self.parameters['shipyard_abandon_dominance']:
                            self.change_position_score(shipyard.cell.ship, shipyard.cell.position, 10000)
                            logging.debug("Ship " + str(shipyard.cell.ship.id) + " stays at position " + str(
                                shipyard.position) + " to guard a shipyard.")
                    else:
                        self.spawn_ship(shipyard)
                        for enemy in enemies:
                            logging.debug("Attacking a ship near our shipyard")
                            self.change_position_score(shipyard.cell.ship, enemy.position,
                                                       500)  # equalize to crash into the ship even if that means we also lose our ship
                            self.attack_position(
                                enemy.position)  # Maybe also do this if we don't spawn a ship, but can move one to the shipyard
                else:
                    potential_guards = [neighbour.ship for neighbour in get_neighbours(shipyard.cell) if
                                        neighbour.ship is not None and neighbour.ship.player_id == self.player_id and neighbour.ship.halite <= max_halite]
                    if len(potential_guards) > 0 and (
                            self.reached_spawn_limit(board) or self.halite < self.config.spawn_cost):
                        guard = sorted(potential_guards, key=lambda ship: ship.halite)[0]
                        self.change_position_score(guard, shipyard.position, 8000)
                        self.ship_types[guard.id] = ShipType.GUARDING
                        logging.debug("Ship " + str(guard.id) + " moves to position " + str(
                            shipyard.position) + " to protect a shipyard.")
                    elif self.halite > self.config.spawn_cost and (dominance >= self.parameters[
                        'shipyard_guarding_min_dominance'] or board.step <= 25 or self.shipyard_count == 1) and self.step_count < \
                            self.parameters['end_start']:
                        logging.debug("Shipyard " + str(shipyard.id) + " spawns a ship to defend the position.")
                        self.spawn_ship(shipyard)
                    else:
                        logging.debug("Shipyard " + str(shipyard.id) + " cannot be protected.")

    def should_convert(self, ship: Ship):
        if self.halite + ship.halite < self.config.convert_cost:
            return False
        if self.shipyard_count == 0:
            return True  # TODO: choose best ship
        if self.shipyard_count >= self.parameters['max_shipyards']:
            return False
        if self.average_halite_per_cell / self.shipyard_count < self.parameters[
            'shipyard_conversion_threshold'] or self.shipyard_count / self.ship_count >= self.parameters[
            'ships_shipyards_threshold']:
            return False
        ship_pos = TO_INDEX[ship.position]
        if self.medium_dominance_map[ship_pos] < self.parameters['shipyard_min_dominance']:
            return False
        distance_to_nearest_shipyard = self.shipyard_distances[ship_pos]
        if self.parameters['min_shipyard_distance'] <= distance_to_nearest_shipyard <= self.parameters[
            'max_shipyard_distance']:
            ship_point = ship.position
            good_distance = []
            for shipyard_position in self.shipyard_positions:
                if self.parameters['min_shipyard_distance'] <= get_distance(ship_pos, shipyard_position) <= \
                        self.parameters['max_shipyard_distance']:
                    good_distance.append(Point.from_index(shipyard_position, SIZE))
            if len(good_distance) == 0:
                return False
            midpoints = []
            if self.shipyard_count == 1:
                half = 0.5 * get_vector(ship_point, good_distance[0])
                half = Point(round(half.x), round(half.y))
                midpoints.append((ship_point + half) % SIZE)
            else:
                for i in range(len(good_distance)):
                    for j in range(i + 1, len(good_distance)):
                        pos1, pos2 = good_distance[i], good_distance[j]
                        if (pos1.x == pos2.x == ship_point.x) or (
                                pos1.y == pos2.y == ship_point.y):  # rays don't intersect
                            midpoints.append(ship_point)
                        else:
                            midpoints.append(get_excircle_midpoint(pos1, pos2, ship_point))
            threshold = self.parameters[
                            'shipyard_min_population'] * self.average_halite_population * self.nb_cells_in_farming_radius
            if any([self.get_populated_cells_in_radius_count(TO_INDEX[midpoint]) >= threshold for midpoint in
                    set(midpoints)]):
                return True
        return False

    def get_populated_cells_in_radius_count(self, position):
        return sum([1 if self.observation['halite'][cell] > 0 else 0 for cell in self.farming_radius_list[position]])

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
        dominance = 1 + self.parameters['mining_score_dominance_norm'] * clip(
            self.small_dominance_map[cell_position] + self.parameters['mining_score_dominance_clip'], 0,
            self.parameters['mining_score_dominance_clip'] * 2) / (
                                self.parameters['mining_score_dominance_clip'] * 2) if self.step_count > 60 else 1
        score = self.parameters['mining_score_gamma'] ** (distance_from_ship + mining_steps) * (
                self.mining_score_beta * ship_halite + (1 - 0.75 ** mining_steps) * min(
            1.02 ** distance_from_ship * halite_val, 500)) * dominance / (
                        distance_from_ship + mining_steps + self.parameters[
                    'mining_score_alpha'] * distance_from_shipyard)
        if distance_from_shipyard == 0 and self.step_count <= 11:
            score *= 0.1  # We don't want to block the shipyard.
        if self.step_count < self.parameters[
            'farming_end'] and halite_val < 499 and cell_position in self.farming_positions:
            score *= self.parameters['mining_score_farming_penalty']
        return score

    def calculate_hunting_score(self, ship: Ship, enemy: Ship) -> float:
        d_halite = enemy.halite - ship.halite
        ship_pos = TO_INDEX[ship.position]
        enemy_pos = TO_INDEX[enemy.position]
        distance = get_distance(ship_pos, enemy_pos)
        if d_halite < 0:
            halite_score = -1
        elif d_halite == 0:
            halite_score = 0.25 * self.parameters['hunting_score_ship_bonus'] * (1 - self.step_count / 398) / \
                           self.parameters['hunting_score_halite_norm']
        else:
            ship_bonus = self.parameters['hunting_score_ship_bonus'] * (1 - self.step_count / 398)
            halite_score = (ship_bonus + d_halite) / self.parameters['hunting_score_halite_norm']
        player_score = 1 + self.parameters['hunting_score_kappa'] * (
            3 - self.player_ranking[ship.player_id] if self.rank <= 1 else self.player_ranking[ship.player_id])
        return self.parameters['hunting_score_gamma'] ** distance * halite_score * (
                self.parameters['hunting_score_delta'] + self.parameters['hunting_score_beta'] * clip(
            self.medium_dominance_map[enemy_pos] + 20, 0, 40) / 40) * player_score * (
                       1 + (self.parameters['hunting_score_iota'] * clip(self.blurred_halite_map[enemy_pos], 0,
                                                                         500) / 500)) * (
                       1 + (self.parameters['hunting_score_zeta'] * clip(self.cargo_map[enemy_pos], 0,
                                                                         self.parameters['hunting_score_cargo_clip']) /
                            self.parameters['hunting_score_cargo_clip'])
               )

    def calculate_cell_score(self, ship: Ship, cell: Cell) -> float:
        score = 0
        if cell.position in self.planned_moves:
            score -= 1500
            return score
        if cell.shipyard is not None:
            shipyard = cell.shipyard
            if shipyard.player_id != self.player_id:
                if ship.halite > self.parameters['max_halite_attack_shipyard']:
                    score -= (300 + ship.halite)
                elif ship.halite == 0 and self.rank == 0:  # only crash into enemy shipyards if we're in a good position
                    score += 400  # Attack the enemy shipyard
            elif self.halite >= self.config.convert_cost and self.shipyard_count == 1 and not self.spawn_limit_reached:
                if self.step_count <= 75 or self.medium_dominance_map[TO_INDEX[shipyard.position]] >= self.parameters[
                    'spawn_min_dominance']:
                    score += self.parameters['move_preference_block_shipyard']
        if cell.ship is not None and cell.ship.player_id != self.player_id:
            if cell.ship.halite < ship.halite:
                score -= (500 + ship.halite - 0.5 * cell.ship.halite)
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
        score += self.parameters['cell_score_dominance'] * self.small_dominance_map[TO_INDEX[cell.position]]
        return score * (1 + self.parameters['cell_score_ship_halite'] * ship.halite)

    def calculate_player_score(self, player):
        return player.halite + len(player.ships) * 500 * (1 - self.step_count / 398) + sum(
            [ship.halite / 4 for ship in player.ships] if len(player.ships) > 0 else [0])

    def calculate_player_map_presence(self, player):
        return len(player.ships) + len(player.shipyards)

    def prefer_moves(self, ship, directions, longest_axis, weight):
        for dir in directions:
            position = (ship.position + dir.to_point()) % self.size
            w = weight
            if dir in longest_axis:
                w += self.parameters['move_preference_longest_axis']
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
        len(self.position_to_index):] = 9999999  # TODO: fix the amount of available shipyard conversions
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
