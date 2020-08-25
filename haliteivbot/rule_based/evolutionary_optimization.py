import os
import pickle
from datetime import datetime
from random import random, choice, sample

import numpy as np

MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.15
POOL_SIZE = 12
SELECTION_CAP = 4  # take the fittest four genomes of a generation
NB_OLD_GENOMES = 2
NB_BASELINE_BOTS = 10
IGNORE_SELECTION_PROBABILITY = 0.03  # the probability to let another genome survive
NB_PARENTS = 3

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
    'guarding_distance_to_shipyard': ('int', (0, 4)),
    'move_preference_guarding_stay': ('int', (-200, -50)),
    'move_preference_guarding': ('int', (60, 100)),
    'guarding_max_ships_per_shipyard': ('int', (1, 4)),
    'farming_start': ('int', (1, 50)),
    'harvest_threshold': ('int', (440, 499)),
    'hunting_score_ypsilon': ('float', (1.1, 3))
}

first_genome = {
    'cargo_map_halite_norm': 197,
    'cell_score_dominance': 2.1600286890260088,
    'cell_score_enemy_halite': 0.35,
    'cell_score_neighbour_discount': 0.7,
    'cell_score_ship_halite': 0.000600485620060368,
    'convert_when_attacked_threshold': 500,
    'disable_hunting_till': 73,
    'dominance_map_halite_clip': 340,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.05,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.14318026743743137,
    'end_return_extra_moves': 7,
    'end_start': 371,
    'ending_halite_threshold': 9,
    'farming_end': 360,
    'guarding_aggression_radius': 8,
    'guarding_distance_to_shipyard': 3,
    'guarding_norm': 0.6535341601623262,
    'guarding_radius': 4,
    'guarding_stop': 340,
    'guarding_max_ships_per_shipyard': 3,
    'hunting_halite_threshold': 0.393243121980592,
    'hunting_min_ships': 16,
    'hunting_score_alpha': 0.5,
    'hunting_score_beta': 2,
    'hunting_score_cargo_clip': 2.434932143755778,
    'hunting_score_delta': 0.73,
    'hunting_score_gamma': 0.9509334468781269,
    'hunting_score_halite_norm': 171,
    'hunting_score_iota': 0.4412732205800474,
    'hunting_score_kappa': 0.39357038462375626,
    'hunting_score_ship_bonus': 200,
    'hunting_score_zeta': 1.2,
    'hunting_threshold': 4,
    'hunting_proportion': 0.3,
    'map_blur_gamma': 0.6534115332552308,
    'map_blur_sigma': 0.8,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 30,
    'max_shipyard_distance': 7,
    'max_shipyards': 5,
    'min_mining_halite': 27,
    'min_ships': 32,
    'min_shipyard_distance': 6,
    'mining_score_alpha': 0.9807755015639334,
    'mining_score_beta': 0.7980915650104368,
    'mining_score_dominance_clip': 3.769020996875946,
    'mining_score_dominance_norm': 0.6513300515784521,
    'mining_score_farming_penalty': 0.06543369486010683,
    'mining_score_gamma': 0.9871204935101099,
    'move_preference_base': 102,
    'move_preference_block_shipyard': -100,
    'move_preference_guarding': 69,
    'move_preference_guarding_stay': -120,
    'move_preference_hunting': 107,
    'move_preference_longest_axis': 12,
    'move_preference_mining': 130,
    'move_preference_return': 115,
    'move_preference_stay_on_shipyard': -120,
    'return_halite': 1000,
    'ship_spawn_threshold': 1.4001702394113038,
    'ships_shipyards_threshold': 0.1,
    'shipyard_abandon_dominance': -15,
    'shipyard_conversion_threshold': 4,
    'shipyard_guarding_attack_probability': 0.1,
    'shipyard_guarding_min_dominance': -15,
    'shipyard_min_dominance': 6.5,
    'shipyard_min_population': 0.6,
    'shipyard_start': 50,
    'shipyard_stop': 260,
    'spawn_min_dominance': 3.528656727561098,
    'spawn_till': 275
}

second_genome = {
    'cargo_map_halite_norm': 197,
    'cell_score_dominance': 2.1600286890260088,
    'cell_score_enemy_halite': 0.35,
    'cell_score_neighbour_discount': 0.7,
    'cell_score_ship_halite': 0.000600485620060368,
    'convert_when_attacked_threshold': 500,
    'disable_hunting_till': 73,
    'dominance_map_halite_clip': 340,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.05,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.14318026743743137,
    'end_return_extra_moves': 7,
    'end_start': 371,
    'ending_halite_threshold': 9,
    'farming_end': 340,
    'guarding_aggression_radius': 8,
    'guarding_distance_to_shipyard': 3,
    'guarding_norm': 0.6535341601623262,
    'guarding_radius': 4,
    'guarding_stop': 340,
    'hunting_halite_threshold': 0.393243121980592,
    'hunting_min_ships': 16,
    'hunting_score_alpha': 0.6,
    'hunting_score_beta': 2,
    'hunting_score_cargo_clip': 2.434932143755778,
    'hunting_score_delta': 0.73,
    'hunting_score_gamma': 0.9509334468781269,
    'hunting_score_halite_norm': 171,
    'hunting_score_iota': 0.4412732205800474,
    'hunting_score_kappa': 0.39357038462375626,
    'hunting_score_ship_bonus': 200,
    'hunting_score_zeta': 1.2,
    'hunting_threshold': 8,
    'hunting_proportion': 0.35,
    'map_blur_gamma': 0.6534115332552308,
    'map_blur_sigma': 0.8,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 30,
    'max_shipyard_distance': 7,
    'max_shipyards': 5,
    'min_mining_halite': 27,
    'min_ships': 32,
    'min_shipyard_distance': 6,
    'mining_score_alpha': 0.9807755015639334,
    'mining_score_beta': 0.7980915650104368,
    'mining_score_dominance_clip': 3.769020996875946,
    'mining_score_dominance_norm': 0.6513300515784521,
    'mining_score_farming_penalty': 0.06543369486010683,
    'mining_score_gamma': 0.9871204935101099,
    'move_preference_base': 102,
    'move_preference_block_shipyard': -100,
    'move_preference_guarding': 69,
    'move_preference_guarding_stay': -120,
    'move_preference_hunting': 107,
    'move_preference_longest_axis': 12,
    'move_preference_mining': 130,
    'move_preference_return': 115,
    'move_preference_stay_on_shipyard': -120,
    'return_halite': 1000,
    'ship_spawn_threshold': 1.4001702394113038,
    'ships_shipyards_threshold': 0.08,
    'shipyard_abandon_dominance': 0.0,
    'shipyard_conversion_threshold': 5.813630085043901,
    'shipyard_guarding_attack_probability': 0.1,
    'shipyard_guarding_min_dominance': -15,
    'shipyard_min_dominance': 5.725656901908077,
    'shipyard_min_population': 0.5,
    'shipyard_start': 50,
    'shipyard_stop': 260,
    'spawn_min_dominance': 3.528656727561098,
    'spawn_till': 275,
    'guarding_max_ships_per_shipyard': 2
}

if __name__ == "__main__":
    from haliteivbot.rule_based.bot_tournament import Tournament


def create_new_genome(parents):
    genome = dict()
    current_parent = choice(parents)
    for characteristic in sample(hyperparameters.keys(), k=len(hyperparameters)):
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
