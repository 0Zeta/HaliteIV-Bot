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
NB_BASELINE_BOTS = 8
IGNORE_SELECTION_PROBABILITY = 0.03  # the probability to let another genome survive
NB_PARENTS = 2

POOL_NAME = ""

hyperparameters = {
    'cargo_map_halite_norm': ('int', (50, 500)),
    'cell_score_enemy_halite': ('float', (0.05, 0.5)),
    'cell_score_neighbour_discount': ('float', (0.6, 0.8)),
    'cell_score_ship_halite': ('float', (0.0005, 0.001)),
    'cell_score_dominance': ('float', (1, 8)),
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
    'hunting_score_alpha': ('float', (-1, 2.5)),
    'hunting_score_beta': ('float', (1.2, 4)),
    'hunting_score_delta': ('float', (0.5, 1.5)),
    'hunting_score_gamma': ('float', (0.75, 0.99)),
    'hunting_score_zeta': ('float', (0.3, 5.)),
    'hunting_score_iota': ('float', (0.2, 0.8)),
    'hunting_score_kappa': ('float', (0.15, 0.4)),
    'hunting_score_cargo_clip': ('float', (1.5, 4.5)),
    'hunting_score_ship_bonus': ('int', (100, 350)),
    'hunting_score_halite_norm': ('int', (50, 300)),
    'hunting_threshold': ('float', (1, 17)),
    'hunting_proportion': ('float', (0., 1.)),
    'map_blur_gamma': ('float', (0.4, 0.95)),
    'map_blur_sigma': ('float', (0.15, 0.8)),
    'max_halite_attack_shipyard': ('int', (0, 0)),
    'max_hunting_ships_per_direction': ('int', (1, 3)),
    'max_ship_advantage': ('int', (-5, 30)),
    'max_shipyard_distance': ('int', (7, 8)),
    'max_shipyards': ('int', (2, 5)),
    'min_mining_halite': ('int', (1, 50)),
    'min_ships': ('int', (8, 40)),
    'min_shipyard_distance': ('int', (5, 6)),
    'mining_score_alpha': ('float', (0.65, 1.5)),
    'mining_score_beta': ('float', (0.55, 0.99)),
    'mining_score_gamma': ('float', (0.97, 0.9999)),
    'mining_score_dominance_clip': ('float', (2, 7)),
    'mining_score_dominance_norm': ('float', (0.2, 2)),
    'mining_score_farming_penalty': ('float', (0.01, 0.15)),
    'move_preference_base': ('int', (85, 110)),
    'move_preference_hunting': ('int', (85, 115)),
    'move_preference_mining': ('int', (110, 135)),
    'move_preference_return': ('int', (115, 145)),
    'move_preference_longest_axis': ('int', (10, 30)),
    'move_preference_stay_on_shipyard': ('int', (-150, -20)),
    'move_preference_block_shipyard': ('int', (-200, -50)),
    'move_preference_constructing': ('int', (120, 250)),
    'move_preference_construction_guarding': ('int', (120, 200)),
    'farming_end': ('int', (340, 370)),
    'return_halite': ('int', (250, 3000)),
    'ship_spawn_threshold': ('float', (0.1, 4.0)),
    'ships_shipyards_threshold': ('float', (0.01, 0.8)),
    'shipyard_abandon_dominance': ('float', (-50, -15)),
    'shipyard_min_population': ('float', (0.5, 1.5)),
    'shipyard_conversion_threshold': ('float', (0.3, 17)),
    'shipyard_guarding_attack_probability': ('float', (0.1, 1)),
    'shipyard_guarding_min_dominance': ('float', (-40, -10)),
    'shipyard_min_dominance': ('float', (-15, 7)),
    'shipyard_start': ('int', (50, 100)),
    'shipyard_stop': ('int', (200, 350)),
    'spawn_min_dominance': ('float', (3.5, 8)),
    'spawn_till': ('int', (200, 350)),
    'guarding_stop': ('int', (250, 370)),
    'guarding_norm': ('float', (0.25, 1.5)),
    'guarding_radius': ('int', (2, 5)),
    'guarding_aggression_radius': ('int', (4, 12)),
    'guarding_min_distance_to_shipyard': ('int', (0, 4)),
    'guarding_max_distance_to_shipyard': ('int', (3, 6)),
    'move_preference_guarding_stay': ('int', (-200, -50)),
    'move_preference_guarding': ('int', (60, 100)),
    'guarding_max_ships_per_shipyard': ('int', (1, 4)),
    'farming_start': ('int', (1, 50)),
    'harvest_threshold': ('int', (440, 499)),
    'hunting_score_ypsilon': ('float', (1.1, 3)),
    'mining_score_juicy': ('float', (0.1, 0.6)),
    'mining_score_start_returning': ('int', (30, 70)),
    'hunting_proportion_after_farming': ('float', (0.01, 0.5)),
    'guarding_ship_advantage_norm': ('int', (10, 30)),
    'mining_score_cargo_norm': ('int', (40, 200)),
    'guarding_end': ('int', (340, 390)),
    'hunting_score_intercept': ('float', (1.5, 10)),
    'hunting_score_hunt': ('float', (3, 15)),
    'hunting_max_group_size': ('int', (2, 5)),
    'hunting_max_group_distance': ('int', (3, 9)),
    'cell_score_farming': ('int', (-500, -100)),
    'hunting_score_farming_position_penalty': ('float', (0.1, 0.9)),
    'third_shipyard_step': ('int', (40, 100)),
    'min_enemy_shipyard_distance': ('int', (1, 9)),
    'shipyard_min_ship_advantage': ('int', (-10, 0)),
    'third_shipyard_min_ships': ('int', (15, 22)),
    'mining_score_juicy_end': ('float', (0.01, 0.4)),
    'second_shipyard_step': ('int', (20, 60)),
}

first_genome = {
    'cargo_map_halite_norm': 309,
    'cell_score_dominance': 2.5481736258265433,
    'cell_score_enemy_halite': 0.45741776220092856,
    'cell_score_neighbour_discount': 0.6726454003260056,
    'cell_score_ship_halite': 0.0006206102891185107,
    'convert_when_attacked_threshold': 550,
    'disable_hunting_till': 75,
    'dominance_map_halite_clip': 340,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.10724586649242973,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.01,
    'end_return_extra_moves': 8,
    'end_start': 378,
    'ending_halite_threshold': 10,
    'farming_end': 350,
    'farming_start': 1,
    'guarding_aggression_radius': 5,
    'guarding_distance_to_shipyard': 3,
    'guarding_max_ships_per_shipyard': 2,
    'guarding_norm': 0.45,
    'guarding_radius': 4,
    'guarding_stop': 343,
    'harvest_threshold': 340,
    'hunting_halite_threshold': 0.01,
    'hunting_min_ships': 15,
    'hunting_proportion': 0.5,
    'hunting_proportion_after_farming': 0.2,
    'hunting_score_alpha': 0.8,
    'hunting_score_beta': 2.3942735021301558,
    'hunting_score_cargo_clip': 1.5,
    'hunting_score_delta': 0.7181206477863321,
    'hunting_score_gamma': 0.9787375189834934,
    'hunting_score_halite_norm': 180,
    'hunting_score_iota': 0.5895761569886987,
    'hunting_score_kappa': 0.37494760608426264,
    'hunting_score_ship_bonus': 150,
    'hunting_score_ypsilon': 1.5,
    'hunting_score_zeta': 2.092552662351012,
    'hunting_threshold': 6,
    'map_blur_gamma': 0.6771645212467934,
    'map_blur_sigma': 0.65022683952122,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 1,
    'max_ship_advantage': 27,
    'max_shipyard_distance': 7,
    'max_shipyards': 7,
    'min_mining_halite': 15,
    'min_ships': 28,
    'min_shipyard_distance': 6,
    'mining_score_alpha': 1,
    'mining_score_beta': 1,
    'mining_score_dominance_clip': 4,
    'mining_score_dominance_norm': 0.35,
    'mining_score_farming_penalty': 0.06542164541355099,
    'mining_score_gamma': 0.99,
    'move_preference_base': 95,
    'move_preference_block_shipyard': -200,
    'move_preference_guarding': 100,
    'move_preference_guarding_stay': -99,
    'move_preference_hunting': 113,
    'move_preference_longest_axis': 13,
    'move_preference_mining': 126,
    'move_preference_return': 125,
    'move_preference_stay_on_shipyard': -75,
    'return_halite': 989,
    'ship_spawn_threshold': 0.14,
    'ships_shipyards_threshold': 0.14,
    'shipyard_abandon_dominance': -23.30616133925598,
    'shipyard_conversion_threshold': 4,
    'shipyard_guarding_attack_probability': 0.35,
    'shipyard_guarding_min_dominance': -15.702344974762006,
    'shipyard_min_dominance': 1.9341508153543074,
    'shipyard_min_population': 0.85,
    'shipyard_start': 35,
    'shipyard_stop': 244,
    'spawn_min_dominance': -10,
    'spawn_till': 280,
    'mining_score_juicy': 0.35,
    'mining_score_start_returning': 50
}

second_genome = {
    'cargo_map_halite_norm': 300,
    'cell_score_dominance': 2.4,
    'cell_score_enemy_halite': 0.35,
    'cell_score_neighbour_discount': 0.6,
    'cell_score_ship_halite': 0.0005,
    'convert_when_attacked_threshold': 500,
    'disable_hunting_till': 35,
    'dominance_map_halite_clip': 350,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.10724586649242973,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.01,
    'end_return_extra_moves': 8,
    'end_start': 370,
    'ending_halite_threshold': 8,
    'farming_end': 345,
    'farming_start': 50,
    'guarding_aggression_radius': 5,
    'guarding_distance_to_shipyard': 3,
    'guarding_max_ships_per_shipyard': 2,
    'guarding_norm': 0.55,
    'guarding_radius': 4,
    'guarding_stop': 350,
    'harvest_threshold': 350,
    'hunting_halite_threshold': 0.01,
    'hunting_min_ships': 12,
    'hunting_proportion': 0.7,
    'hunting_proportion_after_farming': 0.25,
    'hunting_score_alpha': 0.7,
    'hunting_score_beta': 2.5,
    'hunting_score_cargo_clip': 1.7,
    'hunting_score_delta': 0.6,
    'hunting_score_gamma': 0.98,
    'hunting_score_halite_norm': 170,
    'hunting_score_iota': 0.5,
    'hunting_score_kappa': 0.45,
    'hunting_score_ship_bonus': 170,
    'hunting_score_ypsilon': 1.6,
    'hunting_score_zeta': 0.6,
    'hunting_threshold': 7,
    'map_blur_gamma': 0.98,
    'map_blur_sigma': 0.4,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 50,
    'max_shipyard_distance': 7,
    'max_shipyards': 15,
    'min_mining_halite': 12,
    'min_ships': 32,
    'min_shipyard_distance': 6,
    'mining_score_alpha': 0.55,
    'mining_score_beta': 0.7,
    'mining_score_dominance_clip': 4.2,
    'mining_score_dominance_norm': 0.25,
    'mining_score_farming_penalty': 0.01,
    'mining_score_gamma': 0.995,
    'mining_score_delta': 0.6,
    'move_preference_base': 95,
    'move_preference_block_shipyard': -200,
    'move_preference_guarding': 100,
    'move_preference_guarding_stay': -99,
    'move_preference_hunting': 113,
    'move_preference_longest_axis': 13,
    'move_preference_mining': 126,
    'move_preference_return': 125,
    'move_preference_stay_on_shipyard': -75,
    'return_halite': 989,
    'ship_spawn_threshold': 0.01,
    'ships_shipyards_threshold': 0.12,
    'shipyard_abandon_dominance': -23.30616133925598,
    'shipyard_conversion_threshold': 1,
    'shipyard_guarding_attack_probability': 0.35,
    'shipyard_guarding_min_dominance': -15.702344974762006,
    'shipyard_min_dominance': 2,
    'shipyard_min_population': 0.8,
    'shipyard_start': 35,
    'shipyard_stop': 244,
    'spawn_min_dominance': -10,
    'spawn_till': 285,
    'mining_score_juicy': 0.25,
    'mining_score_start_returning': 50
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
