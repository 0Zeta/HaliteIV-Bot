import pickle
from datetime import datetime
from random import random, choice, sample

import numpy as np

from haliteivbot.rule_based.utils import imdict

MUTATION_PROBABILITY = 0.08
CROSSOVER_PROBABILITY = 0.1
POOL_SIZE = 12
SELECTION_CAP = 5  # take the fittest five genomes of a generation
IGNORE_SELECTION_PROBABILITY = 0.1  # the probability to let another genome survive
NB_PARENTS = 3

POOL_NAME = ""

hyperparameters = {
    'cell_score_enemy_halite': ('float', (0.15, 0.5)),
    'cell_score_neighbour_discount': ('float', (0.6, 0.8)),
    'cell_score_ship_halite': ('float', (0.0005, 0.001)),
    'convert_when_attacked_threshold': ('int', (100, 600)),
    'disable_hunting_till': ('int', (7, 100)),
    'dominance_map_halite_clip': ('int', (200, 400)),
    'dominance_map_medium_radius': ('int', (5, 5)),
    'dominance_map_medium_sigma': ('float', (0.01, 0.9)),
    'dominance_map_small_radius': ('int', (3, 3)),
    'dominance_map_small_sigma': ('float', (0.01, 0.8)),
    'end_return_extra_moves': ('int', (6, 15)),
    'end_start': ('int', (370, 390)),
    'ending_halite_threshold': ('int', (1, 30)),
    'hunting_min_ships': ('int', (15, 25)),
    'hunting_halite_threshold': ('float', (0.01, 0.5)),
    'hunting_score_alpha': ('float', (-1, 1.2)),
    'hunting_score_beta': ('float', (1.2, 4)),
    'hunting_score_delta': ('float', (0.5, 1.5)),
    'hunting_score_gamma': ('float', (0.75, 0.99)),
    'hunting_score_iota': ('float', (0.2, 0.8)),
    'hunting_score_kappa': ('float', (0.15, 0.4)),
    'hunting_threshold': ('float', (4, 30)),
    'map_blur_gamma': ('float', (0.4, 0.95)),
    'map_blur_sigma': ('float', (0.15, 0.8)),
    'max_halite_attack_shipyard': ('int', (0, 0)),
    'max_hunting_ships_per_direction': ('int', (1, 3)),
    'max_ship_advantage': ('int', (-5, 30)),
    'max_shipyard_distance': ('int', (7, 20)),
    'max_shipyards': ('int', (2, 5)),
    'min_mining_halite': ('int', (1, 50)),
    'min_ships': ('int', (8, 40)),
    'min_shipyard_distance': ('int', (1, 10)),
    'mining_score_alpha': ('float', (0.65, 0.99)),
    'mining_score_beta': ('float', (0.55, 0.99)),
    'mining_score_gamma': ('float', (0.97, 0.9999)),
    'mining_score_dominance_clip': ('float', (2, 7)),
    'mining_score_dominance_norm': ('float', (0.45, 0.95)),
    'move_preference_base': ('int', (85, 110)),
    'move_preference_hunting': ('int', (85, 115)),
    'move_preference_mining': ('int', (110, 135)),
    'move_preference_return': ('int', (115, 145)),
    'move_preference_longest_axis': ('int', (10, 30)),
    'move_preference_stay_on_shipyard': ('int', (-150, -90)),
    'move_preference_block_shipyard': ('int', (-200, -50)),
    'return_halite': ('int', (250, 3000)),
    'ship_spawn_threshold': ('float', (0.1, 4.0)),
    'ships_shipyards_threshold': ('float', (0.01, 0.8)),
    'shipyard_abandon_dominance': ('float', (-7, 0)),
    'shipyard_conversion_threshold': ('float', (0.3, 17)),
    'shipyard_guarding_attack_probability': ('float', (0.1, 1)),
    'shipyard_guarding_min_dominance': ('float', (2, 7)),
    'shipyard_min_dominance': ('float', (4, 7)),
    'shipyard_start': ('int', (50, 100)),
    'shipyard_stop': ('int', (200, 350)),
    'spawn_min_dominance': ('float', (3.5, 8)),
    'spawn_till': ('int', (200, 350))
}

first_genome = {
    'cell_score_enemy_halite': 0.35,
    'cell_score_neighbour_discount': 0.7,
    'cell_score_ship_halite': 0.0006600467572978282,
    'convert_when_attacked_threshold': 500,
    'disable_hunting_till': 65,
    'dominance_map_halite_clip': 350,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.05,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.05,
    'end_return_extra_moves': 8,
    'end_start': 375,
    'ending_halite_threshold': 1,
    'hunting_halite_threshold': 0.15,
    'hunting_min_ships': 18,
    'hunting_score_alpha': 1,
    'hunting_score_beta': 2.5942517199955524,
    'hunting_score_delta': 0.5142337849582957,
    'hunting_score_gamma': 0.9647931896975708,
    'hunting_score_iota': 0.5,
    'hunting_score_kappa': 0.3114198925625326,
    'hunting_threshold': 8,
    'map_blur_gamma': 0.75,
    'map_blur_sigma': 0.6,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 30,
    'max_shipyard_distance': 7,
    'max_shipyards': 2,
    'min_mining_halite': 37,
    'min_ships': 25,
    'min_shipyard_distance': 2,
    'mining_score_alpha': 0.96,
    'mining_score_beta': 0.7,
    'mining_score_dominance_clip': 4,
    'mining_score_dominance_norm': 0.7,
    'mining_score_gamma': 0.98,
    'move_preference_base': 106,
    'move_preference_hunting': 113,
    'move_preference_longest_axis': 10,
    'move_preference_mining': 130,
    'move_preference_return': 116,
    'move_preference_stay_on_shipyard': -112,
    'move_preference_block_shipyard': -130,
    'return_halite': 1970,
    'ship_spawn_threshold': 0.55,
    'ships_shipyards_threshold': 0.09,
    'shipyard_abandon_dominance': -3.5,
    'shipyard_conversion_threshold': 10,
    'shipyard_guarding_attack_probability': 0.35,
    'shipyard_guarding_min_dominance': 6,
    'shipyard_min_dominance': 7,
    'shipyard_start': 55,
    'shipyard_stop': 260,
    'spawn_min_dominance': 4.5,
    'spawn_till': 240
}

second_genome = {
    'cell_score_enemy_halite': 0.4,
    'cell_score_neighbour_discount': 0.65,
    'cell_score_ship_halite': 0.0005,
    'convert_when_attacked_threshold': 520,
    'disable_hunting_till': 75,
    'dominance_map_halite_clip': 340,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.05,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.05,
    'end_return_extra_moves': 9,
    'end_start': 376,
    'ending_halite_threshold': 2,
    'hunting_halite_threshold': 0.2,
    'hunting_min_ships': 19,
    'hunting_score_alpha': 0.5,
    'hunting_score_beta': 2.3,
    'hunting_score_delta': 0.52,
    'hunting_score_gamma': 0.98,
    'hunting_score_iota': 0.55,
    'hunting_score_kappa': 0.35,
    'hunting_threshold': 11,
    'map_blur_gamma': 0.7,
    'map_blur_sigma': 0.55,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 35,
    'max_shipyard_distance': 8,
    'max_shipyards': 4,
    'min_mining_halite': 30,
    'min_ships': 30,
    'min_shipyard_distance': 3,
    'mining_score_alpha': 0.95,
    'mining_score_beta': 0.65,
    'mining_score_dominance_clip': 5,
    'mining_score_dominance_norm': 0.65,
    'mining_score_gamma': 0.985,
    'move_preference_base': 106,
    'move_preference_hunting': 113,
    'move_preference_longest_axis': 10,
    'move_preference_mining': 130,
    'move_preference_return': 116,
    'move_preference_stay_on_shipyard': -112,
    'move_preference_block_shipyard': -130,
    'return_halite': 1970,
    'ship_spawn_threshold': 0.6,
    'ships_shipyards_threshold': 0.09,
    'shipyard_abandon_dominance': -3.5,
    'shipyard_conversion_threshold': 10,
    'shipyard_guarding_attack_probability': 0.35,
    'shipyard_guarding_min_dominance': 6,
    'shipyard_min_dominance': 7,
    'shipyard_start': 55,
    'shipyard_stop': 210,
    'spawn_min_dominance': 4.5,
    'spawn_till': 220
}

if __name__ == "__main__":
    from haliteivbot.rule_based.bot_tournament import Tournament


def create_new_genome(parents):
    genome = dict()
    current_parent = choice(parents)
    for characteristic in hyperparameters.keys():
        if random() <= CROSSOVER_PROBABILITY:
            current_parent = choice([parent for parent in parents if parent != current_parent])
        if random() <= MUTATION_PROBABILITY:
            mutation = current_parent[characteristic]
            if hyperparameters[characteristic][0] == "float":
                genome[characteristic] = np.clip(np.random.normal(mutation, (
                        hyperparameters[characteristic][1][1] - hyperparameters[characteristic][1][0]) / 10),
                                                 *hyperparameters[characteristic][1])
            elif hyperparameters[characteristic][0] == "int":
                genome[characteristic] = np.clip(int(np.random.normal(mutation, (
                        hyperparameters[characteristic][1][1] - hyperparameters[characteristic][1][0]) / 10)),
                                                 *hyperparameters[characteristic][1])
        else:
            genome[characteristic] = current_parent[characteristic]
    return genome


def optimize():
    baseline_bots = ["optimusmine", "uninstalllol4", "threesigma", "piratehaven", "swarm_intelligence"]
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
            if len(new_pool) > NB_PARENTS:
                pool.append(create_new_genome(sample(new_pool, k=NB_PARENTS)))
            else:
                pool.append(create_new_genome(new_pool))
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
