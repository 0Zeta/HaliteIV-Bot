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

POOL_NAME = "2020-07-31 23-30"

hyperparameters = {
    'cell_score_enemy_halite': ('float', (0.15, 0.5)),
    'cell_score_neighbour_discount': ('float', (0.45, 0.8)),
    'cell_score_ship_halite': ('float', (0.0005, 0.001)),
    'conflict_map_alpha': ('float', (1.1, 1.8)),
    'conflict_map_sigma': ('float', (0.2, 0.8)),
    'conflict_map_zeta': ('float', (0.2, 0.9)),
    'convert_when_attacked_threshold': ('int', (100, 600)),
    'disable_hunting_till': ('int', (7, 100)),
    'dominance_map_medium_radius': ('int', (5, 6)),
    'dominance_map_medium_sigma': ('float', (0.01, 0.9)),
    'dominance_map_small_radius': ('int', (3, 4)),
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
    'max_deposits_per_shipyard': ('int', (2, 8)),
    'max_halite_attack_shipyard': ('int', (10, 250)),
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
    'cell_score_enemy_halite': 0.39330696233048124,
    'cell_score_neighbour_discount': 0.7041223180439514,
    'cell_score_ship_halite': 0.0005,
    'conflict_map_alpha': 1.8,
    'conflict_map_sigma': 0.7023804839341244,
    'conflict_map_zeta': 0.834025988173528,
    'convert_when_attacked_threshold': 309,
    'disable_hunting_till': 89,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.2024065620465002,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.11450683183979835,
    'end_return_extra_moves': 8,
    'end_start': 380,
    'ending_halite_threshold': 23,
    'hunting_halite_threshold': 3,
    'hunting_score_alpha': 0.7826592451483466,
    'hunting_score_beta': 2.516498214879097,
    'hunting_score_delta': 0.8616587667982931,
    'hunting_score_gamma': 0.9278759881495462,
    'hunting_threshold': 3.298297463816602,
    'map_blur_gamma': 0.5231760829513671,
    'map_blur_sigma': 0.5849146910861537,
    'max_deposits_per_shipyard': 3,
    'max_halite_attack_shipyard': 71,
    'max_ship_advantage': 4,
    'max_shipyard_distance': 11,
    'min_mining_halite': 50,
    'min_ships': 20,
    'min_shipyard_distance': 2,
    'mining_score_alpha': 0.9605653191081336,
    'mining_score_beta': 0.9263567999512893,
    'mining_score_gamma': 0.9816820598537683,
    'mining_score_delta': 5,
    'move_preference_base': 109,
    'move_preference_hunting': 109,
    'move_preference_mining': 129,
    'move_preference_return': 118,
    'return_halite': 782,
    'ship_spawn_threshold': 2.2302036265028176,
    'ships_shipyards_threshold': 0.6344446668571576,
    'shipyard_conversion_threshold': 1.897782298822837,
    'shipyard_guarding_min_dominance': 4.269949526616712,
    'shipyard_guarding_attack_probability': 0.5,
    'shipyard_min_dominance': 5.902370354332303,
    'shipyard_stop': 258,
    'spawn_min_dominance': 4.949055398942274,
    'spawn_step_multiplier': 8,
    'spawn_till': 258,
    'shipyard_abandon_dominance': -3
}

second_genome = {
    'cell_score_enemy_halite': 0.39596872452014853,
    'cell_score_neighbour_discount': 0.8,
    'cell_score_ship_halite': 0.000538604773950009,
    'conflict_map_alpha': 1.63893767221509,
    'conflict_map_sigma': 0.7279859793894261,
    'conflict_map_zeta': 0.834025988173528,
    'convert_when_attacked_threshold': 384,
    'disable_hunting_till': 85,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.3577964335027798,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.17561175012921554,
    'end_return_extra_moves': 9,
    'end_start': 380,
    'ending_halite_threshold': 23,
    'hunting_halite_threshold': 0,
    'hunting_score_alpha': 0.7467288152303709,
    'hunting_score_beta': 2.1981782077557686,
    'hunting_score_delta': 0.7702657707618337,
    'hunting_score_gamma': 0.8927649798379433,
    'hunting_threshold': 3.1024310355516356,
    'map_blur_gamma': 0.4773755216689396,
    'map_blur_sigma': 0.5196314170328221,
    'max_deposits_per_shipyard': 3,
    'max_halite_attack_shipyard': 83,
    'max_ship_advantage': 3,
    'max_shipyard_distance': 11,
    'min_mining_halite': 50,
    'min_ships': 19,
    'min_shipyard_distance': 2,
    'mining_score_alpha': 0.9463703062657575,
    'mining_score_beta': 0.9263567999512893,
    'mining_score_gamma': 0.9816820598537683,
    'mining_score_delta': 4,
    'move_preference_base': 110,
    'move_preference_hunting': 109,
    'move_preference_mining': 129,
    'move_preference_return': 115,
    'return_halite': 782,
    'ship_spawn_threshold': 2.2302036265028176,
    'ships_shipyards_threshold': 0.6344446668571576,
    'shipyard_conversion_threshold': 1.4670769137016086,
    'shipyard_guarding_min_dominance': 5.21544416167314,
    'shipyard_min_dominance': 5.519871676136477,
    'shipyard_stop': 217,
    'spawn_min_dominance': 5.159993027190642,
    'spawn_step_multiplier': 9,
    'spawn_till': 249,
    'shipyard_abandon_dominance': -2.5,
    'shipyard_guarding_attack_probability': 0.8
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
        print("Testing new genomes")
        tournament = Tournament(pool)
        results = tournament.play_tournament(
            games=int(len(pool) * 6 / 4 + 2))  # TODO: reduce the number of games played per generation
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
