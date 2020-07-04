import logging

from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Shipyard, Ship

from haliteivbot.utils import *

logging.basicConfig(level=logging.WARNING)
env = make("halite", debug=True)

PARAMETERS = {
    'spawn_till': 290,
    'spawn_step_multiplier': 0,
    'min_ships': 13,
    'ship_spawn_threshold': 0.5029560770386772,
    'shipyard_conversion_threshold': 3.1031254707686524,
    'ships_shipyards_threshold': 1,
    'shipyard_stop': 175,
    'min_shipyard_distance': 7,
    'mining_threshold': 1.0388008062420975,
    'mining_decay': -0.006900507899132262,
    'min_mining_halite': 5,
    'return_halite': 3.1,
    'return_halite_decay': -0.005099217813007813,
    'min_return_halite': 0.12837363338533517,
    'exploring_window_size': 5,
    'convert_when_attacked_threshold': 321,
    'max_halite_attack_shipyard': 50,
    'distance_penalty': 1.1
}

BOT = None


class HaliteBot(object):

    def __init__(self, parameters):
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

        self.parameters = parameters

        # Create exploring window
        window_size = self.parameters['exploring_window_size']
        self.exploring_window = [Point(x, y) for x in range(-window_size, window_size + 1) for y in
                                 range(-window_size, window_size + 1)]
        self.exploring_window.remove(Point(0, 0))

    def step(self, board: Board):
        if self.me is None:
            self.me = board.current_player
            self.player_id = board.current_player_id
            self.config = board.configuration
            self.size = self.config.size
        self.planned_moves.clear()
        self.me = board.current_player
        self.halite = self.me.halite
        self.ship_count = len(self.me.ships)
        self.shipyard_count = len(self.me.shipyards)
        self.planned_shipyards = list()

        self.average_halite_per_cell = sum([halite for halite in board.observation['halite']]) / self.size ** 2

        if self.handle_special_steps(board):
            return  # don't execute the functions below
        self.move_ships(board)
        self.spawn_ships(board)
        return self.me.next_actions

    def handle_special_steps(self, board: Board):
        step = board.step
        if step == 0:
            # Immediately spawn a shipyard
            self.convert_to_shipyard(self.me.ships[0])
            return True
        return False

    def spawn_ships(self, board: Board):
        step = board.step
        # Spawn a ship if there are none left
        if len(self.me.ships) == 0:
            if len(self.me.shipyards) > 0:
                self.spawn_ship(self.me.shipyards[0])
        natural_spawn_limit = step < self.parameters['spawn_till']
        for shipyard in self.me.shipyards:
            if self.halite < 2 * self.config.spawn_cost + board.step * self.parameters['spawn_step_multiplier']:
                return
            if shipyard.position in self.planned_moves:
                continue
            if any(filter(lambda cell: cell.ship is not None and cell.ship.player_id != self.player_id,
                          get_neighbours(shipyard.cell))):
                self.spawn_ship(shipyard)
                continue
            if natural_spawn_limit:
                continue
            if self.ship_count > self.parameters['min_ships'] and self.average_halite_per_cell / self.ship_count < \
                    self.parameters['ship_spawn_threshold']:
                continue
            if any(filter(lambda cell: cell.ship is None and cell.position not in self.planned_moves,
                          get_neighbours(shipyard.cell))):
                self.spawn_ship(shipyard)

    def move_ships(self, board: Board):
        returning_ships = list()
        mining_ships = list()
        exploring_ships = list()
        endangered_ships = list()

        for ship in sorted(self.me.ships, key=lambda s: self.get_friendly_neighbour_count(s.cell), reverse=True):
            if ship.halite > self.average_halite_per_cell * (
                    max(self.parameters['return_halite'] + self.parameters['return_halite_decay'] * board.step,
                        self.parameters['min_return_halite'])):
                returning_ships.append(ship)
            elif ship.cell.halite >= max(self.average_halite_per_cell * (
                    self.parameters['mining_threshold'] + self.parameters['mining_decay'] * board.step),
                                         self.parameters['min_mining_halite']):
                mining_ships.append(ship)
            else:
                exploring_ships.append(ship)

        for ship in mining_ships:
            safe_cells, alternative_cells = self.get_safe_cells(ship)
            if ship.cell in safe_cells:
                self.planned_moves.append(ship.position)
                continue

            if len(safe_cells) > 0:
                target = sorted(safe_cells, key=lambda cell: cell.halite, reverse=True)[0].position
                self.planned_moves.append(target)
                ship.next_action = navigate(ship.position, target, self.size)[0]
                continue

            endangered_ships.append(ship)

        for ship in returning_ships:
            destination = self.get_nearest_shipyard(ship.position)
            if destination is None:
                if self.halite >= self.config.convert_cost:
                    if self.shipyard_count == 0:
                        ship.next_action = ShipAction.CONVERT
                        self.halite -= self.config.convert_cost
                        continue
                    else:
                        destination = board.cells[self.planned_shipyards[0]]
                else:
                    self.planned_moves.append(ship.position)  # TODO: check cell safety
                    continue
            destination = destination.position
            if board.step <= self.parameters['shipyard_stop'] and calculate_distance(ship.position, destination,
                                                                                     self.size) >= self.parameters[
                'min_shipyard_distance'] and self.halite >= self.config.convert_cost + self.config.spawn_cost:
                if self.average_halite_per_cell / self.shipyard_count >= self.parameters[
                    'shipyard_conversion_threshold'] \
                        and self.shipyard_count / self.ship_count < self.parameters['ships_shipyards_threshold']:
                    self.convert_to_shipyard(ship)
                    continue
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
                    endangered_ships.append(ship)
                    continue

            ship.next_action = action
            self.planned_moves.append(next_pos)

        for ship in sorted(endangered_ships, key=lambda es: es.halite, reverse=True):
            # Convert endangered ships to shipyards
            if ship.halite >= self.parameters[
                'convert_when_attacked_threshold'] and self.halite >= self.config.convert_cost:
                self.convert_to_shipyard(ship)
                continue

            _, not_so_safe_cells = self.get_safe_cells(ship)
            solved = False

            if ship in returning_ships:
                destination = self.get_nearest_shipyard(ship.position).position
                for action in navigate(ship.position, destination, self.size):
                    new_pos = (ship.position + action.to_point()) % self.size
                    if board.cells[new_pos] in not_so_safe_cells:
                        ship.next_action = action
                        self.planned_moves.append(new_pos)
                        solved = True
                        break
                if solved:
                    continue

            # TODO: maybe implement attacking other ships
            if len(not_so_safe_cells) > 0:
                if ship.cell in not_so_safe_cells:
                    # do nothing
                    self.planned_moves.append(ship.position)
                else:
                    target = choice(not_so_safe_cells).position
                    ship.next_action = navigate(ship.position, target, self.size)[0]
                    self.planned_moves.append(target)
                continue

            for neighbour in get_neighbours(ship.cell):
                if neighbour.position not in self.planned_moves:
                    self.planned_moves.append(neighbour.position)
                    ship.next_action = navigate(ship.position, neighbour.position, self.size)[0]
                    solved = True
                    break
            if not solved:
                logging.warning("Collision unavoidable:", ship.position)

        for ship in exploring_ships:
            # Guard the shipyard currently on
            if ship.cell.shipyard is not None and ship.position not in self.planned_moves:
                enemies = set(filter(lambda cell: cell.ship is not None and cell.ship.player_id != self.player_id,
                                     get_neighbours(ship.cell)))
                if len(enemies) > 0:
                    # An enemy ship is nearby. We stay on the shipyard to protect it or attack it.
                    unattacked_enemies = [enemy for enemy in enemies if enemy.position not in self.planned_moves]
                    if self.halite < self.config.spawn_cost or len(unattacked_enemies) == 0:
                        self.planned_moves.append(ship.position)
                    else:
                        # attack the ship and immediately spawn a new ship on the shipyard to protect it
                        target_to_attack = choice(unattacked_enemies).position
                        self.planned_moves.append(target_to_attack)
                        ship.next_action = navigate(ship.position, target_to_attack, self.size)[0]
                        self.spawn_ship(ship.cell.shipyard)
                    continue

            possible_targets = sorted(filter(lambda cell: cell.position not in self.planned_moves,
                                             [board.cells[(ship.position + w) % self.size] for w in
                                              self.exploring_window]),
                                      key=lambda cell: cell.halite / (
                                              self.parameters['distance_penalty'] ** calculate_distance(
                                          ship.position, cell.position, self.size)),
                                      reverse=True)  # optimize with target selection

            for target in possible_targets:
                directions = filter(lambda dir: (ship.position + dir.to_point()) % self.size not in self.planned_moves,
                                    navigate(ship.position, target.position, self.size))
                action = next(directions, False)
                if not action:
                    continue
                ship.next_action = action
                self.planned_moves.append((ship.position + action.to_point()) % self.size)
                break

    def get_nearest_shipyard(self, pos: Point):
        min_distance = float('inf')
        nearest_shipyard = None
        for shipyard in self.me.shipyards:
            distance = calculate_distance(pos, shipyard.position, self.size)
            if distance < min_distance:
                min_distance = distance
                nearest_shipyard = shipyard
        return nearest_shipyard

    def get_safe_cells(self, ship: Ship):
        neighbours = get_neighbours(ship.cell)
        # cells to which no dangerous enemy ship can get in the next move, but enemies could still convert their ships
        true_safe_cells = []
        # cells on which no dangerous enemy ships are
        not_so_safe_cells = []
        if self.is_safe(ship, ship.cell):
            true_safe_cells.append(ship.cell)
        else:
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


@board_agent
def agent(board: Board):
    global BOT
    if BOT is None:
        BOT = HaliteBot(PARAMETERS)

    logging.debug("Begin step " + str(board.step))
    BOT.step(board)
