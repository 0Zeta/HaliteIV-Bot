import os
import pickle
from datetime import datetime
from random import random, choice, sample

import numpy as np

MUTATION_PROBABILITY = 0.15
CROSSOVER_PROBABILITY = 0.2
POOL_SIZE = 12
SELECTION_CAP = 4  # take the fittest four genomes of a generation
NB_OLD_GENOMES = 2
NB_BASELINE_BOTS = 8
IGNORE_SELECTION_PROBABILITY = 0.03  # the probability to let another genome survive
NB_PARENTS = 2

POOL_NAME = "2020-09-11 06-12"

hyperparameters = {
    'cargo_map_halite_norm': ('int', (100, 400)),
    'cell_score_dominance': ('float', (0.1, 8)),
    'cell_score_enemy_halite': ('float', (0.05, 0.5)),
    'cell_score_farming': ('int', (-150, -10)),
    'cell_score_danger': ('int', (40, 80)),
    'cell_score_neighbour_discount': ('float', (0.6, 0.8)),
    'cell_score_ship_halite': ('float', (0.0002, 0.001)),
    'convert_when_attacked_threshold': ('int', (100, 520)),
    'disable_hunting_till': ('int', (7, 100)),
    'dominance_map_halite_clip': ('int', (50, 250)),
    'dominance_map_medium_sigma': ('float', (2.65, 3.1)),
    'dominance_map_small_sigma': ('float', (1.3, 1.8)),
    'dominance_map_small_radius': ('int', (3, 3)),
    'dominance_map_medium_radius': ('int', (5, 5)),
    'early_second_shipyard': ('int', (35, 65)),
    'end_return_extra_moves': ('int', (4, 15)),
    'end_start': ('int', (370, 390)),
    'ending_halite_threshold': ('int', (1, 30)),
    'farming_end': ('int', (345, 365)),
    'farming_start': ('int', (1, 50)),
    'farming_start_shipyards': ('int', (2, 3)),
    'greed_min_map_diff': ('int', (8, 15)),
    'greed_stop': ('int', (30, 50)),
    'guarding_aggression_radius': ('int', (2, 6)),
    'guarding_end': ('int', (350, 390)),
    'guarding_max_distance_to_shipyard': ('int', (3, 6)),
    'guarding_max_ships_per_shipyard': ('int', (1, 4)),
    'guarding_min_distance_to_shipyard': ('int', (1, 4)),
    'guarding_norm': ('float', (0.25, 1.5)),
    'guarding_radius': ('int', (2, 4)),
    'guarding_radius2': ('int', (0, 1)),
    'guarding_ship_advantage_norm': ('int', (10, 30)),
    'guarding_stop': ('int', (270, 370)),
    'harvest_threshold_alpha': ('float', (0.01, 0.3)),
    'harvest_threshold_beta': ('float', (0.32, 0.38)),
    'harvest_threshold_ship_advantage_norm': ('int', (10, 15)),
    'harvest_threshold_hunting_norm': ('float', (0.3, 0.9)),
    'hunting_halite_threshold': ('float', (0.01, 0.3)),
    'hunting_max_group_distance': ('int', (3, 9)),
    'hunting_max_group_size': ('int', (1, 4)),
    'hunting_min_ships': ('int', (10, 20)),
    'hunting_proportion': ('float', (0.05, 0.9)),
    'hunting_proportion_after_farming': ('float', (0.1, 0.5)),
    'hunting_score_alpha': ('float', (-1, 2.5)),
    'hunting_score_beta': ('float', (0.1, 1)),
    'hunting_score_cargo_clip': ('float', (1.5, 4.5)),
    'hunting_score_delta': ('float', (0.8, 1.2)),
    'hunting_score_farming_position_penalty': ('float', (0.1, 0.95)),
    'hunting_score_gamma': ('float', (0.85, 0.999)),
    'hunting_score_halite_norm': ('int', (50, 300)),
    'hunting_score_hunt': ('float', (1.2, 8)),
    'hunting_score_intercept': ('float', (1.1, 5)),
    'hunting_score_iota': ('float', (0.1, 0.8)),
    'hunting_score_kappa': ('float', (0.1, 0.4)),
    'hunting_score_region': ('float', (1.8, 3.8)),
    'hunting_score_ship_bonus': ('int', (100, 350)),
    'hunting_score_ypsilon': ('float', (1.1, 3)),
    'hunting_score_zeta': ('float', (0.2, 3)),
    'hunting_threshold': ('float', (1, 15)),
    'map_blur_gamma': ('float', (0.7, 0.99)),
    'map_blur_sigma': ('float', (0.1, 0.8)),
    'map_ultra_blur': ('float', (1, 2)),
    'max_guarding_ships_per_target': ('int', (2, 4)),
    'max_halite_attack_shipyard': ('int', (0, 0)),
    'max_hunting_ships_per_direction': ('int', (1, 2)),
    'max_ship_advantage': ('int', (-1, 25)),
    'max_shipyard_distance': ('int', (7, 8)),
    'max_shipyards': ('int', (6, 15)),
    'min_enemy_shipyard_distance': ('int', (1, 9)),
    'min_mining_halite': ('int', (5, 40)),
    'min_ships': ('int', (15, 40)),
    'min_shipyard_distance': ('int', (5, 6)),
    'mining_score_alpha': ('float', (0.85, 1.15)),
    'mining_score_alpha_min': ('float', (0.6, 0.99)),
    'mining_score_alpha_step': ('float', (0.007, 0.015)),
    'mining_score_beta': ('float', (0.85, 1)),
    'mining_score_beta_min': ('float', (0.85, 0.99)),
    'mining_score_cargo_norm': ('float', (1.5, 3.5)),
    'mining_score_dominance_clip': ('float', (3, 7)),
    'mining_score_dominance_norm': ('float', (0.2, 2)),
    'mining_score_farming_penalty': ('float', (0.001, 0.15)),
    'mining_score_gamma': ('float', (0.97, 1)),
    'mining_score_juicy': ('float', (0.1, 0.6)),
    'mining_score_juicy_end': ('float', (0.01, 0.4)),
    'mining_score_start_returning': ('int', (30, 65)),
    'move_preference_base': ('int', (85, 110)),
    'move_preference_block_shipyard': ('int', (-200, -50)),
    'move_preference_constructing': ('int', (120, 180)),
    'move_preference_construction_guarding': ('int', (120, 200)),
    'move_preference_guarding': ('int', (60, 100)),
    'move_preference_guarding_stay': ('int', (-200, -50)),
    'move_preference_hunting': ('int', (85, 115)),
    'move_preference_longest_axis': ('int', (10, 30)),
    'move_preference_mining': ('int', (110, 135)),
    'move_preference_return': ('int', (115, 145)),
    'move_preference_stay_on_shipyard': ('int', (-150, -20)),
    'return_halite': ('int', (250, 2000)),
    'second_shipyard_min_ships': ('int', (8, 18)),
    'second_shipyard_step': ('int', (15, 60)),
    'ship_spawn_threshold': ('float', (0.05, 2)),
    'ships_shipyards_threshold': ('float', (0.05, 0.25)),
    'shipyard_abandon_dominance': ('float', (-50, -15)),
    'shipyard_conversion_threshold': ('float', (0.3, 7)),
    'shipyard_guarding_attack_probability': ('float', (0.1, 1)),
    'shipyard_guarding_min_dominance': ('float', (-40, -10)),
    'shipyard_min_dominance': ('float', (-15, 7)),
    'shipyard_min_population': ('float', (0.7, 1.5)),
    'shipyard_min_ship_advantage': ('int', (-30, -4)),
    'shipyard_start': ('int', (100, 230)),
    'shipyard_stop': ('int', (200, 350)),
    'spawn_min_dominance': ('float', (-2, 5)),
    'spawn_till': ('int', (220, 325)),
    'third_shipyard_min_ships': ('int', (15, 22)),
    'third_shipyard_step': ('int', (40, 100)),
    'trading_start': ('int', (150, 350)),
    'max_intrusion_count': ('int', (3, 5)),
    'minor_harvest_threshold': ('float', (0.2, 0.7)),
    'mining_score_minor_farming_penalty': ('float', (0.01, 0.17))
}

frozen_parameters = ['dominance_map_medium_radius', 'dominance_map_small_radius', 'guarding_min_distance_to_shipyard',
                     'guarding_max_distance_to_shipyard', 'farming_end', 'guarding_norm', 'guarding_radius',
                     'guarding_end', 'map_blur_gamma', 'max_halite_attack_shipyard', 'max_hunting_ships_per_direction',
                     'max_shipyard_distance', 'min_shipyard_distance', 'mining_score_farming_penalty',
                     'shipyard_abandon_dominance', 'shipyard_guarding_min_dominance', 'shipyard_min_population',
                     'shipyard_start']
frozen_early_parameters = frozen_parameters + [
    'cell_score_farming',
    'end_return_extra_moves',
    'end_start',
    'ending_halite_threshold',
    'farming_start',
    'farming_start_shipyards',
    'guarding_end',
    'guarding_ship_advantage_norm',
    'guarding_stop',
    'harvest_threshold_alpha',
    'harvest_threshold_hunting_norm',
    'harvest_threshold_slope',
    'hunting_halite_threshold',
    'hunting_max_group_distance',
    'hunting_max_group_size',
    'hunting_proportion_after_farming',
    'map_ultra_blur',
    'max_guarding_ships_per_target',
    'min_enemy_shipyard_distance',
    'mining_score_juicy_end',
    'shipyard_min_population',
    'shipyard_min_ship_advantage',
    'shipyard_stop',
    'spawn_till']

first_genome = {
    'cargo_map_halite_norm': 200,
    'cell_score_dominance': 0.5,
    'cell_score_enemy_halite': 0.25,
    'cell_score_farming': -50,
    'cell_score_neighbour_discount': 0.65,
    'cell_score_ship_halite': 0.0005,
    'convert_when_attacked_threshold': 500,
    'disable_hunting_till': 65,
    'dominance_map_halite_clip': 100,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 2.8,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 1.5,
    'early_second_shipyard': 15,
    'end_return_extra_moves': 5,
    'end_start': 382,
    'ending_halite_threshold': 10,
    'farming_end': 355,
    'farming_start': 40,
    'farming_start_shipyards': 2,
    'guarding_aggression_radius': 3,
    'guarding_end': 375,
    'guarding_max_distance_to_shipyard': 3,
    'guarding_max_ships_per_shipyard': 4,
    'guarding_min_distance_to_shipyard': 1,
    'guarding_norm': 0.6,
    'guarding_radius': 3,
    'guarding_radius2': 0,
    'guarding_ship_advantage_norm': 17,
    'guarding_stop': 342,
    'max_guarding_ships_per_target': 2,
    'harvest_threshold_alpha': 0.25,
    'harvest_threshold_hunting_norm': 0.65,
    'harvest_threshold_beta': 0.35,
    'harvest_threshold_ship_advantage_norm': 15,
    'hunting_halite_threshold': 0.05,
    'hunting_max_group_distance': 5,
    'hunting_max_group_size': 1,
    'hunting_min_ships': 8,
    'hunting_proportion': 0.45,
    'hunting_proportion_after_farming': 0.35,
    'hunting_score_alpha': 0.8,
    'hunting_score_beta': 0.25,
    'hunting_score_cargo_clip': 1.5,
    'hunting_score_delta': 0.9,
    'hunting_score_farming_position_penalty': 0.8,
    'hunting_score_gamma': 0.98,
    'hunting_score_halite_norm': 200,
    'hunting_score_hunt': 2,
    'hunting_score_intercept': 1.5,
    'hunting_score_iota': 0.3,
    'hunting_score_kappa': 0.1,
    'hunting_score_ship_bonus': 200,
    'hunting_score_ypsilon': 2,
    'hunting_score_zeta': 0.2,
    'hunting_threshold': 12,
    'hunting_score_region': 2.8,
    'map_blur_gamma': 0.9,
    'map_blur_sigma': 0.3579575706817798,
    'map_ultra_blur': 1.75,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 1,
    'max_ship_advantage': 25,
    'max_shipyard_distance': 8,
    'max_shipyards': 10,
    'min_enemy_shipyard_distance': 4,
    'min_mining_halite': 25,
    'min_ships': 20,
    'min_shipyard_distance': 7,
    'mining_score_alpha': 1,
    'mining_score_cargo_norm': 4,
    'mining_score_alpha_step': 0.008,
    'mining_score_alpha_min': 0.45,
    'mining_score_beta': 0.95,
    'mining_score_beta_min': 0.6,
    'mining_score_dominance_clip': 3,
    'mining_score_dominance_norm': 0.35,
    'mining_score_farming_penalty': 0.01,
    'mining_score_gamma': 0.99,
    'mining_score_juicy': 0.25,
    'mining_score_juicy_end': 0.1,
    'mining_score_start_returning': 50,
    'move_preference_base': 94,
    'move_preference_block_shipyard': -170,
    'move_preference_constructing': 145,
    'move_preference_construction_guarding': 130,
    'move_preference_guarding': 98,
    'move_preference_guarding_stay': -110,
    'move_preference_hunting': 105,
    'move_preference_longest_axis': 15,
    'move_preference_mining': 125,
    'move_preference_return': 120,
    'move_preference_stay_on_shipyard': -80,
    'return_halite': 1000,
    'second_shipyard_min_ships': 8,
    'second_shipyard_step': 7,
    'ship_spawn_threshold': 0.18,
    'ships_shipyards_threshold': 0.18,
    'shipyard_abandon_dominance': -20,
    'shipyard_conversion_threshold': 2.5,
    'shipyard_guarding_attack_probability': 0.35,
    'shipyard_guarding_min_dominance': -15,
    'shipyard_min_dominance': -1,
    'shipyard_min_population': 5,  # deactivate normal shipyard placement
    'shipyard_min_ship_advantage': -12,
    'shipyard_start': 30,
    'shipyard_stop': 285,
    'spawn_min_dominance': -8.0,
    'spawn_till': 285,
    'third_shipyard_min_ships': 17,
    'third_shipyard_step': 35,
    'trading_start': 100,
    'max_intrusion_count': 3,
    'minor_harvest_threshold': 0.55,
    'mining_score_minor_farming_penalty': 0.15,
    'greed_stop': 35,
    'greed_min_map_diff': 12
}

second_genome = {
    'cargo_map_halite_norm': 210,
    'cell_score_dominance': 0.6,
    'cell_score_enemy_halite': 0.15,
    'cell_score_farming': -60,
    'cell_score_neighbour_discount': 0.7,
    'cell_score_ship_halite': 0.0007,
    'convert_when_attacked_threshold': 490,
    'disable_hunting_till': 55,
    'dominance_map_halite_clip': 90,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 2.8,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 1.5,
    'early_second_shipyard': 25,
    'end_return_extra_moves': 5,
    'end_start': 382,
    'ending_halite_threshold': 10,
    'farming_end': 355,
    'farming_start': 40,
    'farming_start_shipyards': 2,
    'guarding_aggression_radius': 4,
    'guarding_end': 375,
    'guarding_max_distance_to_shipyard': 3,
    'guarding_max_ships_per_shipyard': 4,
    'guarding_min_distance_to_shipyard': 1,
    'guarding_norm': 0.5,
    'guarding_radius': 3,
    'guarding_radius2': 1,
    'guarding_ship_advantage_norm': 17,
    'guarding_stop': 342,
    'max_guarding_ships_per_target': 2,
    'harvest_threshold_alpha': 0.25,
    'harvest_threshold_hunting_norm': 0.65,
    'harvest_threshold_beta': 0.35,
    'harvest_threshold_ship_advantage_norm': 15,
    'hunting_halite_threshold': 0.05,
    'hunting_max_group_distance': 5,
    'hunting_max_group_size': 1,
    'hunting_min_ships': 7,
    'hunting_proportion': 0.4,
    'hunting_proportion_after_farming': 0.35,
    'hunting_score_alpha': 0.82,
    'hunting_score_beta': 0.27,
    'hunting_score_cargo_clip': 1.6,
    'hunting_score_delta': 0.85,
    'hunting_score_farming_position_penalty': 0.9,
    'hunting_score_gamma': 0.99,
    'hunting_score_halite_norm': 220,
    'hunting_score_hunt': 1.8,
    'hunting_score_intercept': 1.2,
    'hunting_score_iota': 0.25,
    'hunting_score_kappa': 0.2,
    'hunting_score_ship_bonus': 250,
    'hunting_score_ypsilon': 1.8,
    'hunting_score_zeta': 0.25,
    'hunting_threshold': 10,
    'hunting_score_region': 3,
    'map_blur_gamma': 0.9,
    'map_blur_sigma': 0.3579575706817798,
    'map_ultra_blur': 1.75,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 1,
    'max_ship_advantage': 35,
    'max_shipyard_distance': 8,
    'max_shipyards': 10,
    'min_enemy_shipyard_distance': 4,
    'min_mining_halite': 25,
    'min_ships': 20,
    'min_shipyard_distance': 7,
    'mining_score_alpha': 0.98,
    'mining_score_cargo_norm': 3.5,
    'mining_score_alpha_step': 0.007,
    'mining_score_alpha_min': 0.6,
    'mining_score_beta': 0.98,
    'mining_score_beta_min': 0.7,
    'mining_score_dominance_clip': 3.3,
    'mining_score_dominance_norm': 0.45,
    'mining_score_farming_penalty': 0.01,
    'mining_score_gamma': 0.985,
    'mining_score_juicy': 0.35,
    'mining_score_juicy_end': 0.1,
    'mining_score_start_returning': 45,
    'move_preference_base': 100,
    'move_preference_block_shipyard': -140,
    'move_preference_constructing': 135,
    'move_preference_construction_guarding': 125,
    'move_preference_guarding': 98,
    'move_preference_guarding_stay': -110,
    'move_preference_hunting': 107,
    'move_preference_longest_axis': 17,
    'move_preference_mining': 125,
    'move_preference_return': 130,
    'move_preference_stay_on_shipyard': -80,
    'return_halite': 1000,
    'second_shipyard_min_ships': 10,
    'second_shipyard_step': 15,
    'ship_spawn_threshold': 0.18,
    'ships_shipyards_threshold': 0.18,
    'shipyard_abandon_dominance': -20,
    'shipyard_conversion_threshold': 2.5,
    'shipyard_guarding_attack_probability': 0.35,
    'shipyard_guarding_min_dominance': -15,
    'shipyard_min_dominance': -1,
    'shipyard_min_population': 5,  # deactivate normal shipyard placement
    'shipyard_min_ship_advantage': -12,
    'shipyard_start': 30,
    'shipyard_stop': 285,
    'spawn_min_dominance': -8.0,
    'spawn_till': 285,
    'third_shipyard_min_ships': 18,
    'third_shipyard_step': 45,
    'trading_start': 100,
    'max_intrusion_count': 4,
    'minor_harvest_threshold': 0.65,
    'mining_score_minor_farming_penalty': 0.15,
    'greed_stop': 32,
    'greed_min_map_diff': 15
}

if __name__ == "__main__":
    from haliteivbot.rule_based.bot_tournament import Tournament, EarlyTournament


def create_new_genome(parents, early=True):
    genome = dict()
    current_parent = choice(parents)
    for characteristic in sample(hyperparameters.keys(), k=len(hyperparameters)):
        if random() <= CROSSOVER_PROBABILITY:
            possible_parents = [parent for parent in parents if parent != current_parent]
            if len(possible_parents) == 0:
                possible_parents = parents
            current_parent = choice(possible_parents)
        if random() <= MUTATION_PROBABILITY and characteristic not in (
        frozen_early_parameters if early else frozen_parameters):
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


def optimize(early=True):
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
                pool.append(create_new_genome(sample(pool, k=NB_PARENTS), early=early))
            else:
                pool.append(create_new_genome(pool, early=early))
        for index, genome in enumerate(pool):
            genome['evo_id'] = index
        for baseline_bot in baseline_bots:
            pool.append("evolutionary/bots/" + baseline_bot)
        print("Testing new genomes")
        if early:
            tournament = EarlyTournament(pool, 100)
        else:
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
