import os
import pickle
from datetime import datetime
from random import random, choice, sample

import numpy as np

MUTATION_PROBABILITY = 0.15
CROSSOVER_PROBABILITY = 0.2
POOL_SIZE = 16
SELECTION_CAP = 4  # take the fittest four genomes of a generation
NB_OLD_GENOMES = 2
NB_BASELINE_BOTS = 4
IGNORE_SELECTION_PROBABILITY = 0.03  # the probability to let another genome survive
NB_PARENTS = 2

POOL_NAME = ""

hyperparameters = {
    'cargo_map_halite_norm': ('int', (100, 400)),
    'cell_score_enemy_halite': ('float', (0.05, 0.5)),
    'cell_score_neighbour_discount': ('float', (0.6, 0.8)),
    'cell_score_ship_halite': ('float', (0.0002, 0.001)),
    'cell_score_dominance': ('float', (1, 8)),
    'convert_when_attacked_threshold': ('int', (100, 520)),
    'disable_hunting_till': ('int', (7, 100)),
    'dominance_map_halite_clip': ('int', (200, 400)),
    'dominance_map_medium_radius': ('int', (5, 5)),
    'dominance_map_medium_sigma': ('float', (0.01, 0.9)),
    'dominance_map_small_radius': ('int', (3, 3)),
    'dominance_map_small_sigma': ('float', (0.01, 0.8)),
    'end_return_extra_moves': ('int', (4, 15)),
    'end_start': ('int', (370, 390)),
    'ending_halite_threshold': ('int', (1, 30)),
    'hunting_min_ships': ('int', (10, 20)),
    'hunting_halite_threshold': ('float', (0.01, 0.3)),
    'hunting_score_alpha': ('float', (-1, 2.5)),
    'hunting_score_beta': ('float', (1.2, 4)),
    'hunting_score_delta': ('float', (0.5, 1.5)),
    'hunting_score_gamma': ('float', (0.85, 0.999)),
    'hunting_score_zeta': ('float', (0.2, 3)),
    'hunting_score_iota': ('float', (0.1, 0.8)),
    'hunting_score_kappa': ('float', (0.1, 0.4)),
    'hunting_score_cargo_clip': ('float', (1.5, 4.5)),
    'hunting_score_ship_bonus': ('int', (100, 350)),
    'hunting_score_halite_norm': ('int', (50, 300)),
    'hunting_threshold': ('float', (1, 15)),
    'hunting_proportion': ('float', (0.05, 0.9)),
    'map_blur_gamma': ('float', (0.7, 0.99)),
    'map_blur_sigma': ('float', (0.1, 0.8)),
    'max_halite_attack_shipyard': ('int', (0, 0)),
    'max_hunting_ships_per_direction': ('int', (1, 3)),
    'max_ship_advantage': ('int', (-1, 25)),
    'max_shipyard_distance': ('int', (7, 8)),
    'max_shipyards': ('int', (3, 15)),
    'min_mining_halite': ('int', (5, 40)),
    'min_ships': ('int', (15, 40)),
    'min_shipyard_distance': ('int', (5, 6)),
    'mining_score_alpha': ('float', (0.85, 1.15)),
    'mining_score_beta': ('float', (0.75, 1)),
    'mining_score_gamma': ('float', (0.97, 1)),
    'mining_score_dominance_clip': ('float', (3, 7)),
    'mining_score_dominance_norm': ('float', (0.2, 2)),
    'mining_score_farming_penalty': ('float', (0.001, 0.15)),
    'move_preference_base': ('int', (85, 110)),
    'move_preference_hunting': ('int', (85, 115)),
    'move_preference_mining': ('int', (110, 135)),
    'move_preference_return': ('int', (115, 145)),
    'move_preference_longest_axis': ('int', (10, 30)),
    'move_preference_stay_on_shipyard': ('int', (-150, -20)),
    'move_preference_block_shipyard': ('int', (-200, -50)),
    'move_preference_constructing': ('int', (120, 250)),
    'move_preference_construction_guarding': ('int', (120, 200)),
    'farming_end': ('int', (345, 365)),
    'return_halite': ('int', (250, 2000)),
    'ship_spawn_threshold': ('float', (0.05, 2)),
    'ships_shipyards_threshold': ('float', (0.05, 0.25)),
    'shipyard_abandon_dominance': ('float', (-50, -15)),
    'shipyard_min_population': ('float', (0.7, 1.5)),
    'shipyard_conversion_threshold': ('float', (0.3, 7)),
    'shipyard_guarding_attack_probability': ('float', (0.1, 1)),
    'shipyard_guarding_min_dominance': ('float', (-40, -10)),
    'shipyard_min_dominance': ('float', (-15, 7)),
    'shipyard_start': ('int', (100, 230)),
    'shipyard_stop': ('int', (200, 350)),
    'spawn_min_dominance': ('float', (-2, 5)),
    'spawn_till': ('int', (220, 325)),
    'guarding_stop': ('int', (270, 370)),
    'guarding_norm': ('float', (0.25, 1.5)),
    'guarding_radius': ('int', (2, 3)),
    'guarding_aggression_radius': ('int', (4, 12)),
    'guarding_min_distance_to_shipyard': ('int', (1, 4)),
    'guarding_max_distance_to_shipyard': ('int', (3, 6)),
    'move_preference_guarding_stay': ('int', (-200, -50)),
    'move_preference_guarding': ('int', (60, 100)),
    'guarding_max_ships_per_shipyard': ('int', (1, 4)),
    'farming_start': ('int', (1, 50)),
    'harvest_threshold_alpha': ('float', (0.01, 0.3)),
    'harvest_threshold_hunting_norm': ('float', (0.3, 0.9)),
    'harvest_threshold_base': ('int', (180, 255)),
    'hunting_score_ypsilon': ('float', (1.1, 3)),
    'mining_score_juicy': ('float', (0.1, 0.6)),
    'mining_score_start_returning': ('int', (30, 65)),
    'hunting_proportion_after_farming': ('float', (0.1, 0.5)),
    'guarding_ship_advantage_norm': ('int', (10, 30)),
    'guarding_end': ('int', (350, 390)),
    'hunting_score_intercept': ('float', (1.1, 2.5)),
    'hunting_score_hunt': ('float', (1.2, 3.2)),
    'hunting_max_group_size': ('int', (1, 4)),
    'hunting_max_group_distance': ('int', (3, 9)),
    'cell_score_farming': ('int', (-500, -100)),
    'hunting_score_farming_position_penalty': ('float', (0.1, 0.95)),
    'third_shipyard_step': ('int', (40, 100)),
    'min_enemy_shipyard_distance': ('int', (1, 9)),
    'shipyard_min_ship_advantage': ('int', (-30, -8)),
    'third_shipyard_min_ships': ('int', (15, 22)),
    'mining_score_juicy_end': ('float', (0.01, 0.4)),
    'second_shipyard_step': ('int', (15, 60)),
    'second_shipyard_min_ships': ('int', (8, 18)),
    'farming_start_shipyards': ('int', (2, 3)),
    'map_ultra_blur': ('float', (1, 2)),
    'early_second_shipyard': ('int', (25, 45))
}

frozen_parameters = ['dominance_map_medium_radius', 'dominance_map_small_radius', 'guarding_min_distance_to_shipyard',
                     'guarding_max_distance_to_shipyard', 'farming_end', 'guarding_norm', 'guarding_radius',
                     'guarding_end', 'map_blur_gamma', 'max_halite_attack_shipyard', 'max_hunting_ships_per_direction',
                     'max_shipyard_distance', 'min_shipyard_distance', 'mining_score_farming_penalty',
                     'shipyard_abandon_dominance', 'shipyard_guarding_min_dominance', 'shipyard_min_population',
                     'shipyard_start']

first_genome = {
    'cargo_map_halite_norm': 276,
    'cell_score_dominance': 2.5481736258265433,
    'cell_score_enemy_halite': 0.4530629543797203,
    'cell_score_neighbour_discount': 0.676200480431318,
    'cell_score_ship_halite': 0.0006229108666303259,
    'cell_score_farming': -130,
    'convert_when_attacked_threshold': 469,
    'disable_hunting_till': 75,
    'dominance_map_halite_clip': 340,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.10724586649242973,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.06854925842441753,
    'end_return_extra_moves': 5,
    'end_start': 382,
    'ending_halite_threshold': 10,
    'farming_end': 355,
    'farming_start': 40,
    'guarding_aggression_radius': 6,
    'guarding_min_distance_to_shipyard': 2,
    'guarding_max_distance_to_shipyard': 4,
    'guarding_max_ships_per_shipyard': 2,
    'guarding_ship_advantage_norm': 20,
    'guarding_norm': 0.65,
    'guarding_radius': 3,
    'guarding_end': 370,
    'guarding_stop': 342,
    'harvest_threshold_alpha': 0.15,
    'harvest_threshold_hunting_norm': 0.5,
    'harvest_threshold_base': 185,
    'hunting_halite_threshold': 0.04077647561190107,
    'hunting_min_ships': 10,
    'hunting_proportion': 0.4,
    'hunting_proportion_after_farming': 0.28,
    'hunting_score_alpha': 0.8,
    'hunting_score_beta': 2.391546761028965,
    'hunting_score_cargo_clip': 1.5,
    'hunting_score_delta': 0.7181206477863321,
    'hunting_score_gamma': 0.95,
    'hunting_score_halite_norm': 203,
    'hunting_score_iota': 0.6344102425255267,
    'hunting_score_kappa': 0.39089297661963435,
    'hunting_score_ship_bonus': 174,
    'hunting_score_ypsilon': 2,
    'hunting_score_zeta': 2,
    'hunting_score_farming_position_penalty': 0.8,
    'hunting_threshold': 6,
    'map_blur_gamma': 0.95,
    'map_blur_sigma': 0.32460420355548203,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 27,
    'max_shipyard_distance': 7,
    'max_shipyards': 10,
    'min_mining_halite': 5,
    'min_ships': 20,
    'min_shipyard_distance': 6,
    'mining_score_alpha': 1,
    'mining_score_beta': 0.96,
    'mining_score_dominance_clip': 2.7914078388504846,
    'mining_score_dominance_norm': 0.35,
    'mining_score_farming_penalty': 0.01,
    'mining_score_gamma': 0.98,
    'mining_score_juicy': 0.35,
    'mining_score_juicy_end': 0.1,
    'mining_score_start_returning': 56,
    'move_preference_base': 95,
    'move_preference_block_shipyard': -200,
    'move_preference_guarding': 98,
    'move_preference_guarding_stay': -99,
    'move_preference_hunting': 109,
    'move_preference_longest_axis': 10,
    'move_preference_mining': 125,
    'move_preference_return': 119,
    'move_preference_constructing': 150,
    'move_preference_construction_guarding': 130,
    'move_preference_stay_on_shipyard': -95,
    'return_halite': 989,
    'ship_spawn_threshold': 0.12,
    'ships_shipyards_threshold': 0.15,
    'shipyard_abandon_dominance': -36.82080985520312,
    'shipyard_conversion_threshold': 3,
    'shipyard_guarding_attack_probability': 0.35,
    'shipyard_guarding_min_dominance': -15.702344974762006,
    'shipyard_min_dominance': -3,
    'shipyard_min_population': 1.5,
    'shipyard_start': 180,
    'shipyard_stop': 250,
    'spawn_min_dominance': -10,
    'spawn_till': 270,
    'hunting_max_group_size': 1,
    'hunting_max_group_distance': 5,
    'hunting_score_intercept': 1.25,
    'hunting_score_hunt': 2,
    'second_shipyard_step': 30,
    'third_shipyard_step': 50,
    'min_enemy_shipyard_distance': 6,
    'shipyard_min_ship_advantage': -4,
    'second_shipyard_min_ships': 15,
    'third_shipyard_min_ships': 18,
    'farming_start_shipyards': 2,
    'map_ultra_blur': 1.5,
    'early_second_shipyard': 30
}

second_genome = {
    'cargo_map_halite_norm': 27,
    'cell_score_dominance': 2.4,
    'cell_score_enemy_halite': 0.4,
    'cell_score_neighbour_discount': 0.6,
    'cell_score_ship_halite': 0.0006229108666303259,
    'cell_score_farming': -130,
    'convert_when_attacked_threshold': 420,
    'disable_hunting_till': 65,
    'dominance_map_halite_clip': 340,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.10724586649242973,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.06854925842441753,
    'end_return_extra_moves': 6,
    'end_start': 382,
    'ending_halite_threshold': 10,
    'farming_end': 355,
    'farming_start': 40,
    'guarding_aggression_radius': 6,
    'guarding_min_distance_to_shipyard': 2,
    'guarding_max_distance_to_shipyard': 4,
    'guarding_max_ships_per_shipyard': 2,
    'guarding_ship_advantage_norm': 20,
    'guarding_norm': 0.65,
    'guarding_radius': 3,
    'guarding_end': 375,
    'guarding_stop': 341,
    'harvest_threshold_alpha': 0.1,
    'harvest_threshold_hunting_norm': 0.4,
    'harvest_threshold_base': 190,
    'hunting_halite_threshold': 0.04077647561190107,
    'hunting_min_ships': 10,
    'hunting_proportion': 0.45,
    'hunting_proportion_after_farming': 0.28,
    'hunting_score_alpha': 0.8,
    'hunting_score_beta': 2.391546761028965,
    'hunting_score_cargo_clip': 1.5,
    'hunting_score_delta': 0.65,
    'hunting_score_gamma': 0.95,
    'hunting_score_halite_norm': 203,
    'hunting_score_iota': 0.6,
    'hunting_score_kappa': 0.35,
    'hunting_score_ship_bonus': 174,
    'hunting_score_ypsilon': 2,
    'hunting_score_zeta': 1.1452680492519223,
    'hunting_score_farming_position_penalty': 0.8,
    'hunting_threshold': 7,
    'map_blur_gamma': 0.94,
    'map_blur_sigma': 0.32460420355548203,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 27,
    'max_shipyard_distance': 7,
    'max_shipyards': 10,
    'min_mining_halite': 5,
    'min_ships': 20,
    'min_shipyard_distance': 6,
    'mining_score_alpha': 1,
    'mining_score_beta': 0.96,
    'mining_score_dominance_clip': 2.7914078388504846,
    'mining_score_dominance_norm': 0.35,
    'mining_score_farming_penalty': 0.01,
    'mining_score_gamma': 0.98,
    'mining_score_juicy': 0.35,
    'mining_score_juicy_end': 0.1,
    'mining_score_start_returning': 56,
    'move_preference_base': 95,
    'move_preference_block_shipyard': -200,
    'move_preference_guarding': 98,
    'move_preference_guarding_stay': -99,
    'move_preference_hunting': 109,
    'move_preference_longest_axis': 10,
    'move_preference_mining': 125,
    'move_preference_return': 119,
    'move_preference_constructing': 150,
    'move_preference_construction_guarding': 130,
    'move_preference_stay_on_shipyard': -95,
    'return_halite': 989,
    'ship_spawn_threshold': 0.1,
    'ships_shipyards_threshold': 0.16,
    'shipyard_abandon_dominance': -36.82080985520312,
    'shipyard_conversion_threshold': 3,
    'shipyard_guarding_attack_probability': 0.35,
    'shipyard_guarding_min_dominance': -15.702344974762006,
    'shipyard_min_dominance': -3,
    'shipyard_min_population': 1.5,
    'shipyard_start': 180,
    'shipyard_stop': 260,
    'spawn_min_dominance': -10,
    'spawn_till': 280,
    'hunting_max_group_size': 1,
    'hunting_max_group_distance': 5,
    'hunting_score_intercept': 1.25,
    'hunting_score_hunt': 2,
    'second_shipyard_step': 25,
    'third_shipyard_step': 55,
    'min_enemy_shipyard_distance': 6,
    'shipyard_min_ship_advantage': -4,
    'second_shipyard_min_ships': 14,
    'third_shipyard_min_ships': 19,
    'farming_start_shipyards': 2,
    'map_ultra_blur': 1.25,
    'early_second_shipyard': 35
}

if __name__ == "__main__":
    from haliteivbot.rule_based.bot_tournament import Tournament


def create_new_genome(parents):
    genome = dict()
    current_parent = choice(parents)
    for characteristic in sample(hyperparameters.keys(), k=len(hyperparameters)):
        if random() <= CROSSOVER_PROBABILITY:
            possible_parents = [parent for parent in parents if parent != current_parent]
            if len(possible_parents) == 0:
                possible_parents = parents
            current_parent = choice(possible_parents)
        if random() <= MUTATION_PROBABILITY and characteristic not in frozen_parameters:
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
    baseline_bots_list = list(os.listdir('evolutionary/bots'))
    if len(baseline_bots_list) < NB_BASELINE_BOTS:
        baseline_bots = baseline_bots_list
    else:
        baseline_bots = sample(baseline_bots_list, k=NB_BASELINE_BOTS)
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
        old_genomes = list(os.listdir('evolutionary/genomes'))
        for _ in range(min(NB_OLD_GENOMES, len(old_genomes))):  # Put some random old genomes into the pool
            random_pool = load_pool(choice(old_genomes))[:2]  # consider only the best two genomes of the pool
            pool.append(choice(random_pool))

        print("Filling pool with current size of %i" % len(pool))
        for i in range(POOL_SIZE - len(pool) - ((POOL_SIZE + len(baseline_bots)) % 4)):
            if len(pool) > NB_PARENTS:
                pool.append(create_new_genome(sample(pool, k=NB_PARENTS)))
            else:
                pool.append(create_new_genome(pool))
        for index, genome in enumerate(pool):
            genome['evo_id'] = index
        for baseline_bot in baseline_bots:
            pool.append("evolutionary/bots/" + baseline_bot)
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
