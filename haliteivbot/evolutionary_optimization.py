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

POOL_NAME = "2020-08-03 23-54"

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
    'min_shipyard_distance': ('int', (1, 10)),
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
    'ships_shipyards_threshold': ('float', (0.05, 0.8)),
    'shipyard_abandon_dominance': ('float', (-7, 0)),
    'shipyard_conversion_threshold': ('float', (0.3, 17)),
    'shipyard_guarding_attack_probability': ('float', (0.1, 1)),
    'shipyard_guarding_min_dominance': ('float', (2, 7)),
    'shipyard_min_dominance': ('float', (4, 7)),
    'shipyard_stop': ('int', (200, 350)),
    'spawn_min_dominance': ('float', (3.5, 8)),
    'spawn_step_multiplier': ('int', (0, 30)),
    'spawn_till': ('int', (200, 350))
}

first_genome = {
    'cell_score_enemy_halite': 0.5,
    'cell_score_neighbour_discount': 0.6314176680808354,
    'cell_score_ship_halite': 0.0006475776727929211,
    'conflict_map_alpha': 1.6204313907158685,
    'conflict_map_sigma': 0.7758427455727701,
    'conflict_map_zeta': 0.8638736196759889,
    'convert_when_attacked_threshold': 566,
    'disable_hunting_till': 74,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.07032581227654462,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.16162648714299707,
    'end_return_extra_moves': 6,
    'end_start': 380,
    'ending_halite_threshold': 26,
    'hunting_halite_threshold': 1,
    'hunting_score_alpha': 0.8341107700885344,
    'hunting_score_beta': 1.9068001523156506,
    'hunting_score_delta': 0.5142337849582957,
    'hunting_score_gamma': 0.9124755750840987,
    'hunting_threshold': 5.9551109040067285,
    'map_blur_gamma': 0.46063343500277076,
    'map_blur_sigma': 0.5434126106796429,
    'max_halite_attack_shipyard': 204,
    'max_hunting_ships_per_direction': 1,
    'max_ship_advantage': 0,
    'max_shipyard_distance': 10,
    'min_mining_halite': 44,
    'min_ships': 14,
    'min_shipyard_distance': 3,
    'mining_score_alpha': 0.99,
    'mining_score_beta': 0.9151352865019396,
    'mining_score_delta': 8.222842191153497,
    'mining_score_gamma': 0.9999,
    'move_preference_base': 106,
    'move_preference_hunting': 108,
    'move_preference_mining': 126,
    'move_preference_return': 115,
    'return_halite': 1777,
    'ship_spawn_threshold': 0.45803692967349235,
    'ships_shipyards_threshold': 0.15,
    'shipyard_abandon_dominance': -0.814848142470677,
    'shipyard_conversion_threshold': 9,
    'shipyard_guarding_attack_probability': 1.0,
    'shipyard_guarding_min_dominance': 5.337764506648078,
    'shipyard_min_dominance': 7.0,
    'shipyard_stop': 283,
    'spawn_min_dominance': 3.8657244812902714,
    'spawn_step_multiplier': 1,
    'spawn_till': 315
}

second_genome = {
    'cell_score_enemy_halite': 0.4903782000403733,
    'cell_score_neighbour_discount': 0.5666033969939952,
    'cell_score_ship_halite': 0.0006413833657954024,
    'conflict_map_alpha': 1.545985760884417,
    'conflict_map_sigma': 0.7682710492890434,
    'conflict_map_zeta': 0.8638736196759889,
    'convert_when_attacked_threshold': 566,
    'disable_hunting_till': 78,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.07032581227654462,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.159123792819245,
    'end_return_extra_moves': 6,
    'end_start': 382,
    'ending_halite_threshold': 26,
    'hunting_halite_threshold': 5,
    'hunting_score_alpha': 0.7874234604325203,
    'hunting_score_beta': 2.123061010819677,
    'hunting_score_delta': 0.6769838756696485,
    'hunting_score_gamma': 0.9281365296670487,
    'hunting_threshold': 5.51820945733682,
    'map_blur_gamma': 0.5499922931549402,
    'map_blur_sigma': 0.5434126106796429,
    'max_halite_attack_shipyard': 204,
    'max_hunting_ships_per_direction': 1,
    'max_ship_advantage': 3,
    'max_shipyard_distance': 12,
    'min_mining_halite': 43,
    'min_ships': 14,
    'min_shipyard_distance': 5,
    'mining_score_alpha': 0.99,
    'mining_score_beta': 0.9705326969662169,
    'mining_score_delta': 8.222842191153497,
    'mining_score_gamma': 0.9909440276439139,
    'move_preference_base': 106,
    'move_preference_hunting': 108,
    'move_preference_mining': 127,
    'move_preference_return': 115,
    'return_halite': 1926,
    'ship_spawn_threshold': 0.5300589513288174,
    'ships_shipyards_threshold': 0.2,
    'shipyard_abandon_dominance': -2.170914375668406,
    'shipyard_conversion_threshold': 11,
    'shipyard_guarding_attack_probability': 0.6,
    'shipyard_guarding_min_dominance': 6.355055354872542,
    'shipyard_min_dominance': 6.980137883872445,
    'shipyard_stop': 277,
    'spawn_min_dominance': 3.8657244812902714,
    'spawn_step_multiplier': 1,
    'spawn_till': 290
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
    baseline_bots = ["optimusmine", "uninstalllol4", "threesigma", "piratehaven"]
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
        for i in range(POOL_SIZE - len(new_pool) - ((POOL_SIZE + len(baseline_bots)) % 4)):
            pool.append(create_new_genome([choice(new_pool) for n in range(NB_PARENTS)]))
        pool = [imdict(genome) for genome in pool]
        for baseline_bot in baseline_bots:
            pool.append("evolutionary/bots/" + baseline_bot + ".py")
        print("Testing new genomes")
        tournament = Tournament(pool)
        results = tournament.play_tournament(rounds=6)
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
