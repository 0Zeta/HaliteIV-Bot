import pickle
from datetime import datetime
from random import random, choice

import numpy as np
from kaggle_environments import make

from haliteivbot.rule_based.utils import imdict

MUTATION_PROBABILITY = 0.05
CROSSOVER_PROBABILITY = 0.07
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
    'min_shipyard_distance': ('int', (1, 10)),
    'mining_score_alpha': ('float', (0.65, 0.99)),
    'mining_score_beta': ('float', (0.65, 0.99)),
    'mining_score_gamma': ('float', (0.97, 0.9999)),
    'mining_score_delta': ('float', (1, 25)),
    'move_preference_base': ('int', (85, 110)),
    'move_preference_hunting': ('int', (85, 115)),
    'move_preference_mining': ('int', (110, 135)),
    'move_preference_return': ('int', (115, 145)),
    'move_preference_longest_axis': ('int', (10, 30)),
    'move_preference_stay_on_shipyard': ('int', (-150, -90)),
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
    'spawn_till': ('int', (200, 350))
}

first_genome = {
    'cell_score_enemy_halite': 0.4715114974967425,
    'cell_score_neighbour_discount': 0.6060934491222125,
    'cell_score_ship_halite': 0.000622167405266674,
    'conflict_map_alpha': 1.5516007490707622,
    'conflict_map_sigma': 0.8,
    'conflict_map_zeta': 0.8722136499708464,
    'convert_when_attacked_threshold': 540,
    'disable_hunting_till': 78,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.33970481838807126,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.4145512722151439,
    'end_return_extra_moves': 6,
    'end_start': 382,
    'ending_halite_threshold': 27,
    'hunting_halite_threshold': 5,
    'hunting_score_alpha': 1.0122074915618837,
    'hunting_score_beta': 2.6022792696008183,
    'hunting_score_delta': 0.5,
    'hunting_score_gamma': 0.9747362728981035,
    'hunting_threshold': 4.339439292405498,
    'map_blur_gamma': 0.7569741178073052,
    'map_blur_sigma': 0.6042835862403136,
    'max_halite_attack_shipyard': 250,
    'max_hunting_ships_per_direction': 1,
    'max_ship_advantage': 5,
    'max_shipyard_distance': 11,
    'min_mining_halite': 38,
    'min_ships': 8,
    'min_shipyard_distance': 1,
    'mining_score_alpha': 0.99,
    'mining_score_beta': 0.9151352865019396,
    'mining_score_delta': 6.468192402783087,
    'mining_score_gamma': 0.9875563728525527,
    'move_preference_base': 108,
    'move_preference_hunting': 115,
    'move_preference_mining': 130,
    'move_preference_return': 116,
    'move_preference_longest_axis': 20,
    'return_halite': 1586,
    'ship_spawn_threshold': 0.35385497733106647,
    'ships_shipyards_threshold': 0.22400570922219704,
    'shipyard_abandon_dominance': -2.569244604957094,
    'shipyard_conversion_threshold': 9.761480093798687,
    'shipyard_guarding_attack_probability': 1.0,
    'shipyard_guarding_min_dominance': 5.899371757054431,
    'shipyard_min_dominance': 6.980137883872445,
    'shipyard_stop': 277,
    'spawn_min_dominance': 5.253645801315921,
    'spawn_step_multiplier': 2,
    'spawn_till': 327
}

second_genome = {
    'cell_score_enemy_halite': 0.4715114974967425,
    'cell_score_neighbour_discount': 0.6060934491222125,
    'cell_score_ship_halite': 0.0006413833657954024,
    'conflict_map_alpha': 1.5953629094011479,
    'conflict_map_sigma': 0.7572100660853711,
    'conflict_map_zeta': 0.9,
    'convert_when_attacked_threshold': 540,
    'disable_hunting_till': 78,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 0.33970481838807126,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 0.4145512722151439,
    'end_return_extra_moves': 6,
    'end_start': 381,
    'ending_halite_threshold': 27,
    'hunting_halite_threshold': 5,
    'hunting_score_alpha': 1.0122074915618837,
    'hunting_score_beta': 2.6022792696008183,
    'hunting_score_delta': 0.5142337849582957,
    'hunting_score_gamma': 0.948335898856567,
    'hunting_threshold': 5.116440638204147,
    'map_blur_gamma': 0.7569741178073052,
    'map_blur_sigma': 0.6042835862403136,
    'max_halite_attack_shipyard': 250,
    'max_hunting_ships_per_direction': 1,
    'max_ship_advantage': 6,
    'max_shipyard_distance': 9,
    'min_mining_halite': 38,
    'min_ships': 8,
    'min_shipyard_distance': 3,
    'mining_score_alpha': 0.99,
    'mining_score_beta': 0.9151352865019396,
    'mining_score_delta': 2.9044182229094444,
    'mining_score_gamma': 0.99,
    'move_preference_base': 109,
    'move_preference_hunting': 113,
    'move_preference_mining': 130,
    'move_preference_return': 116,
    'move_preference_longest_axis': 15,
    'return_halite': 1586,
    'ship_spawn_threshold': 0.35385497733106647,
    'ships_shipyards_threshold': 0.15,
    'shipyard_abandon_dominance': -3.299048786606774,
    'shipyard_conversion_threshold': 7.731358616954416,
    'shipyard_guarding_attack_probability': 0.8,
    'shipyard_guarding_min_dominance': 5.899371757054431,
    'shipyard_min_dominance': 6.980137883872445,
    'shipyard_stop': 277,
    'spawn_min_dominance': 5.290290410672599,
    'spawn_till': 320
}

if __name__ == "__main__":
    from haliteivbot.rule_based.bot_tournament import Tournament

    env = make("halite", configuration={"size": 21, "startingHalite": 5000}, debug=True)


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
    baseline_bots = ["optimusmine", "optimusmine2", "uninstalllol4", "threesigma", "threesigma2", "piratehaven",
                     "swarm_intelligence", "swarm_intelligence2"]
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
