import os
import pickle
from datetime import datetime
from random import random, choice, sample

import numpy as np

MUTATION_PROBABILITY = 0.05
CROSSOVER_PROBABILITY = 0.1
POOL_SIZE = 16
SELECTION_CAP = 4  # take the fittest four genomes of a generation
NB_OLD_GENOMES = 3
IGNORE_SELECTION_PROBABILITY = 0.03  # the probability to let another genome survive
NB_PARENTS = 3

POOL_NAME = "2020-08-15 21-42"

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
    'hunting_avg_halite_threshold': ('float', (20, 45)),
    'hunting_min_ships': ('int', (15, 25)),
    'hunting_halite_threshold': ('float', (0.01, 0.5)),
    'hunting_score_alpha': ('float', (-1, 2.5)),
    'hunting_score_beta': ('float', (1.2, 4)),
    'hunting_score_delta': ('float', (0.5, 1.5)),
    'hunting_score_gamma': ('float', (0.75, 0.99)),
    'hunting_score_zeta': ('float', (0.3, 3.)),
    'hunting_score_iota': ('float', (0.2, 0.8)),
    'hunting_score_kappa': ('float', (0.15, 0.4)),
    'hunting_score_cargo_clip': ('float', (1.5, 4.5)),
    'hunting_threshold': ('float', (1, 17)),
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
    'mining_score_alpha': ('float', (0.65, 0.99)),
    'mining_score_beta': ('float', (0.55, 0.99)),
    'mining_score_gamma': ('float', (0.97, 0.9999)),
    'mining_score_dominance_clip': ('float', (2, 7)),
    'mining_score_dominance_norm': ('float', (0.45, 0.95)),
    'mining_score_farming_penalty': ('float', (0.01, 0.25)),
    'move_preference_base': ('int', (85, 110)),
    'move_preference_hunting': ('int', (85, 115)),
    'move_preference_mining': ('int', (110, 135)),
    'move_preference_return': ('int', (115, 145)),
    'move_preference_longest_axis': ('int', (10, 30)),
    'move_preference_stay_on_shipyard': ('int', (-150, -90)),
    'move_preference_block_shipyard': ('int', (-200, -50)),
    'farming_end': ('int', (340, 370)),
    'return_halite': ('int', (250, 3000)),
    'ship_spawn_threshold': ('float', (0.1, 4.0)),
    'ships_shipyards_threshold': ('float', (0.01, 0.8)),
    'shipyard_abandon_dominance': ('float', (-15, 0)),
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
    'cargo_map_halite_norm': 259,
    'cell_score_dominance': 2.0690023066592538,
    'cell_score_enemy_halite': 0.40986189593968725,
    'cell_score_neighbour_discount': 0.7,
    'cell_score_ship_halite': 0.0006600467572978282,
    'convert_when_attacked_threshold': 520,
    'disable_hunting_till': 81,
    'dominance_map_halite_clip': 340,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.15718059530479717,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.05,
    'end_return_extra_moves': 7,
    'end_start': 377,
    'ending_halite_threshold': 3,
    'hunting_halite_threshold': 0.258104350651479,
    'hunting_min_ships': 19,
    'hunting_avg_halite_threshold': 25,
    'hunting_score_alpha': 0.6185031612833689,
    'hunting_score_beta': 2.6052114574581884,
    'hunting_score_delta': 0.8709006820260277,
    'hunting_score_gamma': 0.9647931896975708,
    'hunting_score_iota': 0.48901814674196414,
    'hunting_score_kappa': 0.34774561811044974,
    'hunting_score_zeta': 1,
    'hunting_score_cargo_clip': 2.5,
    'hunting_threshold': 13.789006382752898,
    'map_blur_gamma': 0.6534115332552308,
    'map_blur_sigma': 0.6556642121639878,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 30,
    'max_shipyard_distance': 7,
    'max_shipyards': 4,
    'min_mining_halite': 38,
    'min_ships': 30,
    'min_shipyard_distance': 1,
    'mining_score_alpha': 0.9081426212090371,
    'mining_score_beta': 0.7277067758648436,
    'mining_score_dominance_clip': 4,
    'mining_score_dominance_norm': 0.6161301729692376,
    'mining_score_gamma': 0.98,
    'move_preference_base': 104,
    'move_preference_block_shipyard': -97,
    'move_preference_hunting': 115,
    'move_preference_longest_axis': 10,
    'move_preference_mining': 130,
    'move_preference_return': 116,
    'move_preference_stay_on_shipyard': -125,
    'return_halite': 1970,
    'ship_spawn_threshold': 1.4001702394113038,
    'ships_shipyards_threshold': 0.07791666764994667,
    'shipyard_abandon_dominance': -4.115371900722006,
    'shipyard_conversion_threshold': 10.122728414396494,
    'shipyard_guarding_attack_probability': 0.35,
    'shipyard_guarding_min_dominance': 6.643278855975787,
    'shipyard_min_dominance': 7,
    'shipyard_start': 54,
    'shipyard_stop': 225,
    'spawn_min_dominance': 4.305778383923281,
    'spawn_till': 221
}

second_genome = {
    'cargo_map_halite_norm': 197,
    'cell_score_dominance': 3.0917487434890623,
    'cell_score_enemy_halite': 0.39566629220330396,
    'cell_score_neighbour_discount': 0.6893811619791591,
    'cell_score_ship_halite': 0.0005848591112498942,
    'convert_when_attacked_threshold': 599,
    'disable_hunting_till': 69,
    'dominance_map_halite_clip': 270,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.01,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.134837546384747,
    'end_return_extra_moves': 7,
    'end_start': 376,
    'ending_halite_threshold': 6,
    'hunting_avg_halite_threshold': 40,
    'hunting_halite_threshold': 0.3127343558704865,
    'hunting_min_ships': 19,
    'hunting_score_alpha': 0.2592189407009563,
    'hunting_score_beta': 1.9444542517064307,
    'hunting_score_cargo_clip': 1.7829118845455392,
    'hunting_score_delta': 0.5,
    'hunting_score_gamma': 0.9719276816410414,
    'hunting_score_iota': 0.5574007942685942,
    'hunting_score_kappa': 0.39357038462375626,
    'hunting_score_zeta': 0.40822224414005664,
    'hunting_threshold': 12.12833619658105,
    'map_blur_gamma': 0.7055972481015564,
    'map_blur_sigma': 0.7138008701011678,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 30,
    'max_shipyard_distance': 8,
    'max_shipyards': 3,
    'min_mining_halite': 40,
    'min_ships': 30,
    'min_shipyard_distance': 1,
    'mining_score_alpha': 0.9596977379532147,
    'mining_score_beta': 0.8090153388632068,
    'mining_score_dominance_clip': 4.479191446578104,
    'mining_score_dominance_norm': 0.6933890071390262,
    'mining_score_gamma': 0.9813222444017456,
    'move_preference_base': 105,
    'move_preference_block_shipyard': -150,
    'move_preference_hunting': 107,
    'move_preference_longest_axis': 14,
    'move_preference_mining': 130,
    'move_preference_return': 117,
    'move_preference_stay_on_shipyard': -130,
    'return_halite': 1886,
    'ship_spawn_threshold': 0.7736499100565323,
    'ships_shipyards_threshold': 0.06519798296737947,
    'shipyard_abandon_dominance': -3.9223773173783023,
    'shipyard_conversion_threshold': 13.016513149297758,
    'shipyard_guarding_attack_probability': 0.1,
    'shipyard_guarding_min_dominance': 6.643278855975787,
    'shipyard_min_dominance': 6.40772788905446,
    'shipyard_start': 50,
    'shipyard_stop': 215,
    'spawn_min_dominance': 4.474750310121838,
    'spawn_till': 210
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
    baseline_bots = ["optimusmine", "uninstalllol6", "threesigma", "piratehaven", "swarm_intelligence"]
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
        for i in range(POOL_SIZE - len(new_pool) - ((POOL_SIZE + len(baseline_bots)) % 4)):
            if len(new_pool) > NB_PARENTS:
                pool.append(create_new_genome(sample(new_pool, k=NB_PARENTS)))
            else:
                pool.append(create_new_genome(new_pool))
        for index, genome in enumerate(pool):
            genome['evo_id'] = index
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
