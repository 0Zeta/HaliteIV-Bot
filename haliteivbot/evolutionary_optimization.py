import pickle
from datetime import datetime
from random import random, choice

import numpy as np
from kaggle_environments import make

from haliteivbot.utils import imdict

MUTATION_PROBABILITY = 0.15
POOL_SIZE = 12
SELECTION_CAP = 5  # take the fittest five genomes of a generation
IGNORE_SELECTION_PROBABILITY = 0.1  # the probability to let another genome survive
NB_PARENTS = 3

POOL_NAME = ""

hyperparameters = {
    'cell_score_enemy_halite': ('float', (0.15, 0.5)),
    'cell_score_neighbour_discount': ('float', (0.45, 0.8)),
    'cell_score_ship_halite': ('float', (0.0005, 0.001)),
    'conflict_map_alpha': ('float', (1.1, 1.8)),
    'conflict_map_sigma': ('float', (0.2, 0.8)),
    'conflict_map_zeta': ('float', (0.2, 0.9)),
    'convert_when_attacked_threshold': ('int', (100, 600)),
    'disable_hunting_till': ('int', (7, 100)),
    'dominance_map_medium_radius': ('int', (5, 5)),
    'dominance_map_medium_sigma': ('float', (0.01, 0.9)),
    'dominance_map_small_radius': ('int', (3, 3)),
    'dominance_map_small_sigma': ('float', (0.01, 0.8)),
    'end_return_extra_moves': ('int', (6, 15)),
    'end_start': ('int', (380, 390)),
    'ending_halite_threshold': ('int', (5, 30)),
    'hunting_halite_threshold': ('int', (0, 30)),
    'hunting_score_alpha': ('float', (0.5, 1.2)),
    'hunting_score_beta': ('float', (1.2, 4)),
    'hunting_score_delta': ('float', (0.5, 1.5)),
    'hunting_score_gamma': ('float', (0.75, 0.99)),
    'hunting_threshold': ('float', (0.3, 8)),
    'map_blur_gamma': ('float', (0.4, 0.95)),
    'map_blur_sigma': ('float', (0.15, 0.8)),
    'max_halite_attack_shipyard': ('int', (10, 250)),
    'max_hunting_ships_per_direction': ('int', (1, 3)),
    'max_ship_advantage': ('int', (-5, 15)),
    'max_shipyard_distance': ('int', (11, 20)),
    'min_mining_halite': ('int', (1, 50)),
    'min_ships': ('int', (8, 40)),
    'min_shipyard_distance': ('int', (0, 10)),
    'mining_score_alpha': ('float', (0.65, 0.99)),
    'mining_score_beta': ('float', (0.65, 0.99)),
    'mining_score_gamma': ('float', (0.97, 0.9999)),
    'mining_score_delta': ('float', (1, 25)),
    'move_preference_base': ('int', (85, 110)),
    'move_preference_hunting': ('int', (85, 115)),
    'move_preference_mining': ('int', (110, 135)),
    'move_preference_return': ('int', (105, 125)),
    'return_halite': ('int', (250, 3000)),
    'ship_spawn_threshold': ('float', (0.1, 4.0)),
    'ships_shipyards_threshold': ('float', (0.05, 1.2)),
    'shipyard_abandon_dominance': ('float', (-7, 0)),
    'shipyard_conversion_threshold': ('float', (0.3, 10)),
    'shipyard_guarding_attack_probability': ('float', (0.1, 1)),
    'shipyard_guarding_min_dominance': ('float', (2, 7)),
    'shipyard_min_dominance': ('float', (4, 7)),
    'shipyard_stop': ('int', (200, 390)),
    'spawn_min_dominance': ('float', (3.5, 8)),
    'spawn_step_multiplier': ('int', (0, 30)),
    'spawn_till': ('int', (200, 390))
}

first_genome = {
    'cell_score_enemy_halite': 0.4613959378409517,
    'cell_score_neighbour_discount': 0.6113947160691992,
    'cell_score_ship_halite': 0.0005851883655675024,
    'conflict_map_alpha': 1.6917672920044187,
    'conflict_map_sigma': 0.7273409399199504,
    'conflict_map_zeta': 0.8229121771183198,
    'convert_when_attacked_threshold': 457,
    'disable_hunting_till': 57,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.286263775244114,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.16162648714299707,
    'end_return_extra_moves': 6,
    'end_start': 381,
    'ending_halite_threshold': 26,
    'hunting_halite_threshold': 1,
    'hunting_score_alpha': 0.8358236303543974,
    'hunting_score_beta': 2.3428123008540775,
    'hunting_score_delta': 0.6769838756696485,
    'hunting_score_gamma': 0.8940225811830314,
    'hunting_threshold': 3.3593034816163847,
    'map_blur_gamma': 0.536029448522942,
    'map_blur_sigma': 0.6046014225393499,
    'max_halite_attack_shipyard': 181,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 8,
    'max_shipyard_distance': 11,
    'min_mining_halite': 47,
    'min_ships': 13,
    'min_shipyard_distance': 1,
    'mining_score_alpha': 0.99,
    'mining_score_beta': 0.99,
    'mining_score_delta': 5.909304395959387,
    'mining_score_gamma': 0.9958724155601597,
    'move_preference_base': 109,
    'move_preference_hunting': 107,
    'move_preference_mining': 127,
    'move_preference_return': 117,
    'return_halite': 1866,
    'ship_spawn_threshold': 0.2517015781310412,
    'ships_shipyards_threshold': 0.867178528312429,
    'shipyard_abandon_dominance': -0.640362150288657,
    'shipyard_conversion_threshold': 9.02440555815,
    'shipyard_guarding_attack_probability': 1.0,
    'shipyard_guarding_min_dominance': 4.961712988280522,
    'shipyard_min_dominance': 6.2981094083031,
    'shipyard_stop': 297,
    'spawn_min_dominance': 3.5,
    'spawn_step_multiplier': 0,
    'spawn_till': 390
}

second_genome = {
    'cell_score_enemy_halite': 0.48185602292539215,
    'cell_score_neighbour_discount': 0.6314176680808354,
    'cell_score_ship_halite': 0.0005851883655675024,
    'conflict_map_alpha': 1.6408682243831636,
    'conflict_map_sigma': 0.8,
    'conflict_map_zeta': 0.8826298980363294,
    'convert_when_attacked_threshold': 424,
    'disable_hunting_till': 57,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.286263775244114,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.16162648714299707,
    'end_return_extra_moves': 6,
    'end_start': 381,
    'ending_halite_threshold': 26,
    'hunting_halite_threshold': 1,
    'hunting_score_alpha': 0.7630723992535422,
    'hunting_score_beta': 2.3428123008540775,
    'hunting_score_delta': 0.8057812106367592,
    'hunting_score_gamma': 0.9100930938916377,
    'hunting_threshold': 5.127083533456105,
    'map_blur_gamma': 0.536029448522942,
    'map_blur_sigma': 0.5434126106796429,
    'max_halite_attack_shipyard': 116,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 6,
    'max_shipyard_distance': 11,
    'min_mining_halite': 46,
    'min_ships': 18,
    'min_shipyard_distance': 1,
    'mining_score_alpha': 0.99,
    'mining_score_beta': 0.9772613084891808,
    'mining_score_delta': 6.640630394518772,
    'mining_score_gamma': 0.9942253129737255,
    'move_preference_base': 109,
    'move_preference_hunting': 108,
    'move_preference_mining': 127,
    'move_preference_return': 117,
    'return_halite': 2059,
    'ship_spawn_threshold': 0.2517015781310412,
    'ships_shipyards_threshold': 0.7940620595317368,
    'shipyard_abandon_dominance': -1.2076664226521994,
    'shipyard_conversion_threshold': 9.02440555815,
    'shipyard_guarding_attack_probability': 0.9731528985405345,
    'shipyard_guarding_min_dominance': 4.990430590375083,
    'shipyard_min_dominance': 6.980137883872445,
    'shipyard_stop': 283,
    'spawn_min_dominance': 4.705573749336174,
    'spawn_step_multiplier': 1,
    'spawn_till': 311
}

if __name__ == "__main__":
    from haliteivbot.bot_tournament import Tournament

    env = make("halite", configuration={"size": 21, "startingHalite": 5000}, debug=True)


def create_new_genome(parents):
    genome = dict()
    for characteristic in hyperparameters.keys():
        if random() <= MUTATION_PROBABILITY:
            mutation = choice(parents)[characteristic]
            if hyperparameters[characteristic][0] == "float":
                genome[characteristic] = np.clip(np.random.normal(mutation, (
                        hyperparameters[characteristic][1][1] - hyperparameters[characteristic][1][0]) / 10),
                                                 *hyperparameters[characteristic][1])
            elif hyperparameters[characteristic][0] == "int":
                genome[characteristic] = np.clip(int(np.random.normal(mutation, (
                        hyperparameters[characteristic][1][1] - hyperparameters[characteristic][1][0]) / 10)),
                                                 *hyperparameters[characteristic][1])
        else:
            genome[characteristic] = choice(parents)[characteristic]
    return genome


def optimize():
    if POOL_NAME != "":
        pool = load_pool(POOL_NAME)
        print("Best genome: " + str(pool[0]))
    else:
        pool = [first_genome, second_genome]
    for epoch in range(100000):
        print("Creating generation %i" % epoch)
        new_pool = pool[:SELECTION_CAP] if len(pool) >= SELECTION_CAP else pool
        old_pool = pool[SELECTION_CAP:] if len(pool) > SELECTION_CAP else []
        for genome in old_pool:
            if random() <= IGNORE_SELECTION_PROBABILITY:
                new_pool.append(genome)
        pool = new_pool.copy()
        print("Filling pool with current size of %i" % len(pool))
        for i in range(POOL_SIZE - len(new_pool)):
            pool.append(create_new_genome([choice(new_pool) for n in range(NB_PARENTS)]))
        pool = [imdict(genome) for genome in pool]
        pool.append("evolutionary/bots/optimusmine.py")
        pool.append("evolutionary/bots/uninstalllol4.py")
        pool.append("evolutionary/bots/threesigma.py")
        pool.append("evolutionary/bots/piratehaven.py")
        # pool.append("evolutionary/bots/pytorchstarter.py")
        print("Testing new genomes")
        tournament = Tournament(pool)
        results = tournament.play_tournament(
            games=int(len(pool) * 6 / 4 + 1))  # TODO: reduce the number of games played per generation
        pool = [genome for genome in results if isinstance(genome, dict)]
        best_genome = pool[0]
        print("Best genome so far: " + str(best_genome))
        print("Saving new genomes")
        save_current_pool(pool)
        print("Best genome so far:")
        print(str(pool[0]))


def save_current_pool(pool):
    pickle.dump([dict(genome) for genome in pool],
                open('evolutionary/genomes/' + str(datetime.now().strftime("%Y-%m-%d %H-%M")), 'wb'))


def load_pool(name):
    return pickle.load(open('evolutionary/genomes/' + name, 'rb'))


if __name__ == '__main__':
    optimize()
