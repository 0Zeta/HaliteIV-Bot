import logging
from enum import Enum

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Shipyard, Ship

from haliteivbot.utils import *

logging.basicConfig(level=logging.WARNING)

env = make("halite", debug=True)

PARAMETERS = {
    'spawn_till': 360,
    'spawn_step_multiplier': 0,
    'min_ships': 8,
    'ship_spawn_threshold': 2.17,
    'shipyard_conversion_threshold': 1.53,
    'ships_shipyards_threshold': 0.33,
    'shipyard_stop': 313,
    'min_shipyard_distance': 11,
    'mining_threshold': 8,
    'mining_decay': -0.005,
    'min_mining_halite': 6,
    'return_halite': 3.0,
    'return_halite_decay': 0.0,
    'min_return_halite': 0.1,
    'convert_when_attacked_threshold': 326,
    'max_halite_attack_shipyard': 103,
    'mining_score_alpha': 0.5,
    'mining_score_gamma': 0.9511898071037446,
    'hunting_threshold': 0.9,
    'hunting_halite_threshold': 15,
    'hunting_score_gamma': 0.9,
}

BOT = None


class ShipType(Enum):
    MINING = 1
    RETURNING = 2
    HUNTING = 3


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
        self.average_halite_per_cell = 0

        self.planned_moves = list()  # a list of positions where our ships will be in the next step
        self.planned_shipyards = list()
        self.ship_types = dict()
        self.mining_targets = dict()
        self.friendly_neighbour_count = dict()

        self.enemies = list()

        self.returning_ships = list()
        self.mining_ships = list()
        self.hunting_ships = list()
        self.endangered_ships = list()

        self.optimal_mining_steps = create_optimal_mining_steps_matrix(self.parameters['mining_score_alpha'],
                                                                       self.parameters['mining_score_gamma'])
        create_distance_list(self.size)

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

        # Compute distances to the next shipyard:
        self.shipyard_distances = []
        for position in range(self.size ** 2):
            min_distance = float('inf')
            for shipyard in self.me.shipyards:
                distance = get_distance(position, shipyard.position.to_index(self.size))
                if distance < min_distance:
                    min_distance = distance
            self.shipyard_distances.append(min_distance)

        self.planned_moves.clear()
        self.planned_shipyards.clear()
        self.ship_types.clear()
        self.mining_targets.clear()
        self.enemies = [ship for player in board.players.values() for ship in player.ships if
                        player.id != self.player_id]

        if self.handle_special_steps(board):
            return  # don't execute the functions below
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
        step = board.step
        # Spawn a ship if there are none left
        if len(self.me.ships) == 0:
            if len(self.me.shipyards) > 0:
                self.spawn_ship(self.me.shipyards[0])

        spawn_limit_reached = step > self.parameters['spawn_till']
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
            if self.halite < 2 * self.config.spawn_cost + board.step * self.parameters['spawn_step_multiplier']:
                continue
            if spawn_limit_reached:
                continue
            if self.ship_count > self.parameters['min_ships'] and self.average_halite_per_cell / self.ship_count < \
                    self.parameters['ship_spawn_threshold']:
                continue
            if any(filter(lambda cell: cell.ship is None and cell.position not in self.planned_moves,
                          get_neighbours(shipyard.cell))):
                # Only spawn a ship if there are not too many own ships around the shipyard
                self.spawn_ship(shipyard)

    def move_ships(self, board: Board):
        # TODO: remove these lists
        self.returning_ships.clear()
        self.mining_ships.clear()
        self.hunting_ships.clear()
        self.endangered_ships.clear()

        for ship in self.me.ships:
            ship_type = self.get_ship_type(ship, board)
            self.ship_types[ship.id] = ship_type
            if ship_type == ShipType.MINING and not self.guards_shipyard(ship):
                self.mining_ships.append(ship)
            elif ship_type == ShipType.RETURNING:
                self.returning_ships.append(ship)

        ships = list(
            sorted(self.returning_ships + self.mining_ships, key=lambda s: self.get_move_priority(s), reverse=True))
        self.assign_ship_targets(self.mining_ships, board)  # also converts some ships tp hunting/returning ships
        while len(ships) > 0:
            ship = ships[0]
            ship_type = self.ship_types[ship.id]
            if ship_type == ShipType.MINING:
                self.handle_mining_ship(ship, board)
            elif ship_type == ShipType.RETURNING:
                self.handle_returning_ship(ship, board)
            elif ship_type == ShipType.HUNTING:
                self.handle_hunting_ship(ship, board)

            ships.remove(ship)
            # This could be made more efficient by only computing the new scores of ships affected by the changes of this ship.
            ships.sort(key=lambda s: self.get_move_priority(s), reverse=True)

        map(lambda ship: self.handle_endangered_ship(ship, board),
            sorted(self.endangered_ships, key=lambda es: es.halite, reverse=True))

    def get_move_priority(self, ship: Ship) -> int:
        # Prioritize ships with much halite and ships on cells with much halite
        score = ship.halite + int(ship.cell.halite / 2)

        # Prioritize ships on  shipyards
        if ship.cell.shipyard is not None:
            score += 1200

        # Prioritize ships with many friendly neighbours
        score += self.friendly_neighbour_count[ship.cell.position.to_index(self.size)] * 150

        # Prioritize ships with less safe squares
        score -= len(self.get_safe_cells(ship)[0]) * 200

        return score

    def assign_ship_targets(self, ships, board: Board):
        ship_targets = []
        halite_map = board.observation['halite']
        for ship in self.mining_ships:
            ship_position = ship.position.to_index(self.size)
            targets = sorted([(self.calculate_mining_score(ship_position, cell_position, halite), cell_position) for
                              cell_position, halite in enumerate(halite_map)], key=lambda t: t[0], reverse=True)
            best = targets[0]
            del targets[0]
            ship_targets.append((ship, best[0], best[1], targets))

        ship_targets.sort(key=lambda mt: mt[1], reverse=True)
        mine = True
        while len(ship_targets) > 0:
            ship, best_score, best_target, _ = ship_targets[0]
            if not mine and best_score >= self.parameters['hunting_threshold']:
                logging.debug(
                    "Assigning target " + str(Point.from_index(best_target, self.size)) + " with score " + str(
                        best_score) + " to ship " + str(ship.id))
                self.mining_targets[ship.id] = best_target
                del ship_targets[0]
                conflicting_ship_targets = [ship_target for ship_target in ship_targets if
                                            ship_target[2] == best_target]
                ship_targets = [ship_target for ship_target in ship_targets if
                                ship_target not in conflicting_ship_targets]
                assigned_targets = self.mining_targets.values()
                for conflicting_ship_target in conflicting_ship_targets:
                    targets = conflicting_ship_target[3]
                    target_score, target_cell = targets[0]
                    del targets[0]
                    while target_cell in assigned_targets:
                        target_score, target_cell = conflicting_ship_target[3][0]
                        del conflicting_ship_target[3][0]
                    ship_targets.append((conflicting_ship_target[0], target_score, target_cell, targets))
                ship_targets.sort(key=lambda mt: mt[1], reverse=True)
            else:
                mine = False
                del ship_targets[0]
                if ship.halite < self.parameters['hunting_halite_threshold']:
                    self.hunting_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.HUNTING
                else:
                    self.returning_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.RETURNING

        # Convert indexed positions to points
        for ship_id, target_pos in self.mining_targets.items():
            self.mining_targets[ship_id] = Point.from_index(target_pos, self.size)

    def get_ship_type(self, ship: Ship, board: Board) -> ShipType:
        if ship.halite > self.average_halite_per_cell * (
                max(self.parameters['return_halite'] + self.parameters['return_halite_decay'] * board.step,
                    self.parameters['min_return_halite'])):
            if ship.cell.halite > max(self.average_halite_per_cell * (
                    self.parameters['mining_threshold'] + self.parameters['mining_decay'] * board.step),
                                      self.parameters['min_mining_halite']) and self.is_safe(ship, ship.cell):
                return ShipType.MINING
            return ShipType.RETURNING
        else:
            return ShipType.MINING

    def handle_returning_ship(self, ship: Ship, board: Board):
        destination = self.get_nearest_shipyard(ship.position)
        if destination is None:
            if self.halite >= self.config.convert_cost:
                if self.shipyard_count == 0:
                    ship.next_action = ShipAction.CONVERT
                    self.halite -= self.config.convert_cost
                    logging.debug("Returning ship " + str(
                        ship.id) + " has no shipyard and converts to one at position " + str(ship.position) + ".")
                    return
                else:
                    destination = board.cells[self.planned_shipyards[0]]
            else:
                self.planned_moves.append(ship.position)  # TODO: check cell safety
                logging.debug(
                    "Returning ship " + str(ship.id) + " has no shipyard to go to and stays at position " + str(
                        ship.position) + ".")
                return
        destination = destination.position
        if board.step <= self.parameters['shipyard_stop'] and calculate_distance(ship.position, destination) >= \
                self.parameters[
                    'min_shipyard_distance'] and self.halite >= self.config.convert_cost + self.config.spawn_cost:
            if self.average_halite_per_cell / self.shipyard_count >= self.parameters[
                'shipyard_conversion_threshold'] \
                    and self.shipyard_count / self.ship_count < self.parameters['ships_shipyards_threshold']:
                self.convert_to_shipyard(ship)
                logging.debug("Returning ship " + str(ship.id) + " converts to a shipyard at position " + str(
                    ship.position) + ".")
                return
        action = None
        next_pos = None
        safe_cells, alternative_cells = self.get_safe_cells(ship)
        preferred_moves = navigate(ship.position, destination, self.size)
        for direction in preferred_moves:
            next_pos = (ship.position + direction.to_point()) % self.size
            if next_pos not in self.planned_moves and board.cells[next_pos] in safe_cells:
                action = direction
                break
        if action is None:
            if ship.cell in safe_cells:
                next_pos = ship.position  # do nothing
            elif len(safe_cells) > 0:
                action = navigate(ship.position, safe_cells[0].position, self.size)[0]
                next_pos = safe_cells[0].position
            else:
                self.endangered_ships.append(ship)
                return

        ship.next_action = action
        self.planned_moves.append(next_pos)
        logging.debug(
            "Returning ship " + str(ship.id) + " plans to acquire position " + str(next_pos) + " regularly.")

    def handle_mining_ship(self, ship: Ship, board: Board):
        safe_positions, not_so_safe_positions = self.get_safe_positions(ship)
        positions_to_check = safe_positions if len(safe_positions) > 0 else not_so_safe_positions

        target = self.mining_targets[ship.id]
        if target != ship.position:
            directions = filter(lambda dir: (ship.position + dir.to_point()) % self.size in positions_to_check,
                                navigate(ship.position, target, self.size))
            action = next(directions, False)
            if action:
                ship.next_action = action
                next_pos = (ship.position + action.to_point()) % self.size
                self.planned_moves.append(next_pos)
                logging.debug(
                    "Mining ship " + str(ship.id) + " moves to position " + str(next_pos) + " to reach position " + str(
                        target) + ".")
                return
        else:
            if target in positions_to_check:
                self.planned_moves.append(target)
                logging.debug("Mining ship " + str(ship.id) + " stays at position " + str(target) + ".")
                return
        if len(positions_to_check) == 0:
            # TODO: add to endangered ships?
            unsafe_positions = [cell.position for cell in get_neighbours(ship.cell) if
                                cell.position not in self.planned_moves]
            if len(unsafe_positions) > 0:
                next_pos = choice(unsafe_positions)
                self.planned_moves.append(next_pos)
                ship.next_action = navigate(ship.position, next_pos, self.size)[0]
                logging.debug(
                    "Mining ship " + str(ship.id) + " has nowhere to go and acquires unsafe position " + str(
                        next_pos) + ".")
            else:
                logging.warning("Mining ship " + str(
                    ship.id) + " has nowhere to go and causes a collision at position " + str(
                    ship.position) + ".")
            return
        next_pos = choice(positions_to_check)
        self.planned_moves.append(next_pos)
        if next_pos != ship.position:
            ship.next_action = navigate(ship.position, next_pos, self.size)[0]
        logging.debug(
            "Mining ship " + str(ship.id) + " has no target and acquires position " + str(next_pos) + ".")

    def handle_hunting_ship(self, ship: Ship, board: Board):
        safe_positions, not_so_safe_positions = self.get_safe_positions(ship)
        positions_to_check = safe_positions if len(safe_positions) > 0 else not_so_safe_positions
        if len(self.enemies) > 0:
            target = max(self.enemies, key=lambda enemy: self.calculate_hunting_score(ship, enemy))
            if target.halite > ship.halite:
                directions = filter(lambda dir: (ship.position + dir.to_point()) % self.size in positions_to_check,
                                    navigate(ship.position, target.position, self.size))
                action = next(directions, False)
                if action:
                    ship.next_action = action
                    next_pos = (ship.position + action.to_point()) % self.size
                    self.planned_moves.append(next_pos)
                    logging.debug(
                        "Hunting ship " + str(ship.id) + " moves to position " + str(
                            next_pos) + " to reach position " + str(
                            target.position) + ".")
                    return
        if len(positions_to_check) == 0:
            # TODO: add to endangered ships?
            unsafe_positions = [cell.position for cell in get_neighbours(ship.cell) if
                                cell.position not in self.planned_moves]
            if len(unsafe_positions) > 0:
                next_pos = choice(unsafe_positions)
                self.planned_moves.append(next_pos)
                ship.next_action = navigate(ship.position, next_pos, self.size)[0]
                logging.debug(
                    "Hunting ship " + str(ship.id) + " has nowhere to go and acquires unsafe position " + str(
                        next_pos) + ".")
            else:
                logging.warning("Hunting ship " + str(
                    ship.id) + " has nowhere to go and causes a collision at position " + str(
                    ship.position) + ".")
            return
        next_pos = choice(positions_to_check)
        self.planned_moves.append(next_pos)
        if next_pos != ship.position:
            ship.next_action = navigate(ship.position, next_pos, self.size)[0]
        logging.debug(
            "Hunting ship " + str(ship.id) + " has no target and acquires position " + str(next_pos) + ".")

    def guards_shipyard(self, ship: Ship) -> bool:
        if ship.cell.shipyard is not None and ship.position not in self.planned_moves:
            # Guard the shipyard the ship is currently on
            enemies = set(filter(lambda cell: cell.ship is not None and cell.ship.player_id != self.player_id,
                                 get_neighbours(ship.cell)))
            if len(enemies) > 0:
                # An enemy ship is nearby. We stay on the shipyard to protect it or attack it.
                unattacked_enemies = [enemy for enemy in enemies if enemy.position not in self.planned_moves]
                if self.halite < self.config.spawn_cost or len(unattacked_enemies) == 0:
                    self.planned_moves.append(ship.position)
                    logging.debug("Exploring ship " + str(ship.id) + " stays at position " + str(
                        ship.position) + " to guard a shipyard.")
                else:
                    # attack the ship and immediately spawn a new ship on the shipyard to protect it
                    target_to_attack = choice(unattacked_enemies).position
                    self.planned_moves.append(target_to_attack)
                    logging.debug("Exploring ship " + str(ship.id) + " attacks an enemy ship at position " + str(
                        target_to_attack) + " endangering a shipyard.")
                    ship.next_action = navigate(ship.position, target_to_attack, self.size)[0]
                    self.spawn_ship(ship.cell.shipyard)
                return True
        return False

    def handle_endangered_ship(self, ship: Ship, board: Board):
        # Convert endangered ships to shipyards
        if ship.halite >= self.parameters[
            'convert_when_attacked_threshold'] and self.halite >= self.config.convert_cost:
            self.convert_to_shipyard(ship)
            logging.debug(
                "Returning ship " + str(ship.id) + " can't escape and converts to a shipyard at position " + str(
                    ship.position) + ".")
            return

        _, not_so_safe_cells = self.get_safe_cells(ship)
        solved = False

        if self.ship_types[ship.id] == ShipType.RETURNING:
            destination = self.get_nearest_shipyard(ship.position).position
            for action in navigate(ship.position, destination, self.size):
                new_pos = (ship.position + action.to_point()) % self.size
                if board.cells[new_pos] in not_so_safe_cells:
                    ship.next_action = action
                    self.planned_moves.append(new_pos)
                    solved = True
                    logging.debug("Endangered ship " + str(ship.id) + " moves to unsafe position " + str(
                        new_pos) + " towards destination.")
                    break
            if solved:
                return

        # TODO: maybe implement attacking other ships
        if len(not_so_safe_cells) > 0:
            if ship.cell in not_so_safe_cells:
                # do nothing
                self.planned_moves.append(ship.position)
                logging.debug(
                    "Endangered ship " + str(ship.id) + " stays at unsafe position " + str(ship.position) + ".")
            else:
                target = choice(not_so_safe_cells).position
                ship.next_action = navigate(ship.position, target, self.size)[0]
                self.planned_moves.append(target)
                logging.debug("Endangered ship " + str(ship.id) + " moves to unsafe position " + str(target) + ".")
            return

        for neighbour in get_neighbours(ship.cell):
            if neighbour.position not in self.planned_moves:
                self.planned_moves.append(neighbour.position)
                ship.next_action = navigate(ship.position, neighbour.position, self.size)[0]
                solved = True
                logging.debug("Endangered ship " + str(ship.id) + " is forced to move to position " + str(
                    neighbour.position) + ".")
                break
        if not solved:
            logging.warning("Collision unavoidable:", ship.position)

    def calculate_mining_score(self, ship_position: int, cell_position: int, halite):
        # TODO: account for enemies
        distance_from_ship = get_distance(ship_position, cell_position)
        distance_from_shipyard = self.shipyard_distances[cell_position]
        if distance_from_shipyard > 20:
            # There is no shipyard.
            distance_from_shipyard = 20
        mining_steps = self.optimal_mining_steps[distance_from_ship - 1][distance_from_shipyard - 1]
        return self.parameters['mining_score_gamma'] ** (distance_from_ship + mining_steps) * (
                1 - 0.75 ** mining_steps) * min(1.02 ** (distance_from_ship) * halite, 500) * 1.02 ** mining_steps / (
                       distance_from_ship + mining_steps + self.parameters[
                   'mining_score_alpha'] * distance_from_shipyard)

    def calculate_hunting_score(self, ship: Ship, enemy: Ship) -> float:
        d_halite = enemy.halite - ship.halite
        distance = calculate_distance(ship.position, enemy.position)
        return self.parameters['hunting_score_gamma'] ** distance * d_halite / distance

    def get_nearest_shipyard(self, pos: Point):
        min_distance = float('inf')
        nearest_shipyard = None
        for shipyard in self.me.shipyards:
            distance = calculate_distance(pos, shipyard.position)
            if distance < min_distance:
                min_distance = distance
                nearest_shipyard = shipyard
        return nearest_shipyard

    def get_safe_positions(self, ship: Ship):
        return list(map(lambda cell: cell.position, self.get_safe_cells(ship)[0])), list(
            map(lambda cell: cell.position, self.get_safe_cells(ship)[1]))

    def get_safe_cells(self, ship: Ship):
        neighbours = get_neighbours(ship.cell)
        # cells to which no dangerous enemy ship can get in the next move, but enemies could still convert their ships
        true_safe_cells = []
        # cells on which no dangerous enemy ships are
        not_so_safe_cells = []
        if self.is_safe(ship, ship.cell):
            true_safe_cells.append(ship.cell)
        elif ship.cell.position not in self.planned_moves:
            not_so_safe_cells.append(ship.cell)

        for neighbour in neighbours:
            if self.is_safe(ship, neighbour):
                true_safe_cells.append(neighbour)
            elif (neighbour.ship is None or (
                    neighbour.ship.player_id != self.player_id and neighbour.ship.halite > ship.halite)
                  or neighbour.ship.player_id == self.player_id) and neighbour.position not in self.planned_moves:
                not_so_safe_cells.append(neighbour)
        return true_safe_cells, not_so_safe_cells

    def is_safe(self, ship: Ship, cell: Cell):
        if cell.position in self.planned_moves:
            return False

        if cell.ship is not None and cell.ship.player_id != self.player_id and cell.ship.halite < ship.halite:
            return False

        if cell.shipyard is not None and cell.shipyard.player_id != self.player_id and ship.halite > self.parameters[
            'max_halite_attack_shipyard']:
            return False

        for neighbour in get_neighbours(cell):
            if neighbour.ship is not None and neighbour.ship.player_id != self.player_id and neighbour.ship.halite < ship.halite:
                return False

        return True

    def get_friendly_neighbour_count(self, cell: Cell):
        return sum(1 for _ in
                   filter(lambda n: n.ship is not None and n.ship.player_id == self.player_id, get_neighbours(cell)))

    def convert_to_shipyard(self, ship: Ship):
        assert self.halite >= self.config.convert_cost
        ship.next_action = ShipAction.CONVERT
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
