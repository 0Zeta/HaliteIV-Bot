import logging

from kaggle_environments import make

from haliteivbot.utils import *

logging.basicConfig(level=logging.WARNING)
env = make("halite", debug=True)
DIRECTIONS = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]

HYPERPARAMETERS = {
    'spawn_till': 230,  # spawn ships till move 250
    'spawn_step_multiplier': 5,
    'shipyard_stop': 250,
    'min_shipyard_distance': 10,
    'mining_threshold': 0.8,
    'min_mining_halite': 3,
    'return_halite': 11,
    'exploring_window_size': 4,
    'distance_penalty': 0.5
}

BOT = None


@board_agent
def agent(board: Board):
    global BOT
    if BOT is None:
        BOT = HaliteBot()

    logging.debug("Begin step " + str(board.step))
    BOT.step(board)


class HaliteBot(object):

    def __init__(self, hyperparameters=HYPERPARAMETERS):
        self.config = None
        self.size = 21
        self.me = None
        self.halite = 5000

        self.planned_moves = list()  # a list of positions where our ships will be in the next step

        self.hyperparameters = hyperparameters

        # Create exploring window
        window_size = self.hyperparameters['exploring_window_size']
        self.exploring_window = [Point(x, y) for x in range(-window_size, window_size + 1) for y in
                                 range(-window_size, window_size + 1)]
        self.exploring_window.remove(Point(0, 0))

    def step(self, board: Board):
        if self.me is None:
            self.me = board.current_player
            self.config = board.configuration
            self.size = self.config.size
        self.planned_moves.clear()
        self.me = board.current_player
        self.halite = self.me.halite
        if self.handle_special_steps(board):
            return  # don't execute the functions below
        self.move_ships(board)
        self.spawn_ships(board)
        return self.me.next_actions

    def handle_special_steps(self, board: Board):
        step = board.step
        if step == 0:
            # Immediately spawn a shipyard
            self.me.ships[0].next_action = ShipAction.CONVERT
            self.halite -= self.config.convert_cost
            return True
        return False

    def handle_halite_deposits(self, board: Board):
        pass

    def spawn_ships(self, board: Board):
        step = board.step
        # TODO: spawn a ship if there are none
        if step < self.hyperparameters['spawn_till']:
            for shipyard in self.me.shipyards:
                if self.halite < 2 * self.config.spawn_cost + board.step * self.hyperparameters[
                    'spawn_step_multiplier']:
                    return
                if shipyard.position not in self.planned_moves:
                    shipyard.next_action = ShipyardAction.SPAWN
                    self.halite -= self.config.spawn_cost
                    self.planned_moves.append(shipyard.position)

    def move_ships(self, board: Board):
        average_halite_per_cell = sum([halite for halite in board.observation['halite']]) / self.size ** 2
        returning_ships = list()
        mining_ships = list()
        exploring_ships = list()
        building_shipyard = False

        for ship in self.me.ships:
            if ship.halite > average_halite_per_cell * self.hyperparameters['return_halite']:
                returning_ships.append(ship)
            elif ship.cell.halite >= max(average_halite_per_cell * self.hyperparameters['mining_threshold'],
                                         self.hyperparameters['min_mining_halite']):
                mining_ships.append(ship)
            else:
                exploring_ships.append(ship)

        for ship in mining_ships:
            self.planned_moves.append(ship.position)

        for ship in returning_ships:
            destination = self.get_nearest_shipyard(ship.position)
            if destination is None:
                if self.halite >= self.config.convert_cost:
                    ship.next_action = ShipAction.CONVERT
                    self.halite -= self.config.convert_cost
                else:
                    self.planned_moves.append(ship.position)
                continue
            destination = destination.position
            if board.step <= self.hyperparameters['shipyard_stop'] and calculate_distance(ship.position, destination,
                                                                                          self.size) >= \
                    self.hyperparameters['min_shipyard_distance'] \
                    and self.halite >= self.config.convert_cost + self.config.spawn_cost and not building_shipyard:
                ship.next_action = ShipAction.CONVERT
                self.halite -= self.config.convert_cost
                self.planned_moves.append(ship.position)
                building_shipyard = True
            action = None
            next_pos = None
            preferred_moves = navigate(ship.position, destination, self.size)
            for direction in preferred_moves:
                next_pos = (ship.position + direction.to_point()) % self.size
                if next_pos not in self.planned_moves:
                    action = direction
                    break
            if action is None:
                if ship.position not in self.planned_moves:
                    next_pos = ship.position  # do nothing
                else:
                    possible_moves = [x for x in DIRECTIONS if x not in preferred_moves and (
                            ship.position + x.to_point()) % self.size not in self.planned_moves]
                    if len(possible_moves) > 0:
                        action = possible_moves[0]
                        next_pos = (ship.position + possible_moves[0].to_point()) % self.size
                    else:
                        logging.warning("Collision unavoidable: ", ship.id)

            ship.next_action = action
            self.planned_moves.append(next_pos)

        for ship in exploring_ships:
            possible_targets = sorted(filter(lambda cell: cell.position not in self.planned_moves,
                                             [board.cells[(ship.position + w) % self.size] for w in
                                              self.exploring_window]),
                                      key=lambda cell: cell.halite / (
                                              self.hyperparameters['distance_penalty'] * calculate_distance(
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
