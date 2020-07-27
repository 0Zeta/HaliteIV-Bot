import os
import pickle
import traceback
from datetime import datetime
from random import random, choice

import numpy as np
from kaggle_environments import evaluate
from kaggle_environments import make

from haliteivbot.bot import HaliteBot, Board

MUTATION_PROBABILITY = 0.15
POOL_SIZE = 12
SELECTION_CAP = 5  # take the fittest five genomes of a generation
IGNORE_SELECTION_PROBABILITY = 0.1  # the probability to let another genome survive
NB_PARENTS = 3

POOL_NAME = ""

hyperparameters = {
    'spawn_till': ('int', (200, 390)),
    'spawn_step_multiplier': ('int', (0, 30)),
    'min_ships': ('int', (8, 40)),
    'ship_spawn_threshold': ('float', (0.1, 4.0)),
    'shipyard_conversion_threshold': ('float', (0.3, 10)),
    'ships_shipyards_threshold': ('float', (0.05, 1.2)),
    'shipyard_stop': ('int', (200, 390)),
    'min_shipyard_distance': ('int', (0, 35)),
    'min_mining_halite': ('int', (1, 50)),
    'convert_when_attacked_threshold': ('int', (100, 600)),
    'max_halite_attack_shipyard': ('int', (10, 250)),
    'mining_score_alpha': ('float', (0.65, 0.99)),
    'mining_score_beta': ('float', (0.65, 0.99)),
    'mining_score_gamma': ('float', (0.97, 0.9999)),
    'hunting_threshold': ('float', (0.3, 8)),
    'hunting_halite_threshold': ('int', (0, 30)),
    'disable_hunting_till': ('int', (7, 100)),
    'hunting_score_alpha': ('float', (0.5, 1.2)),
    'hunting_score_gamma': ('float', (0.75, 0.99)),
    'return_halite': ('int', (250, 3000)),
    'max_ship_advantage': ('int', (-5, 15)),
    'map_blur_sigma': ('float', (0.15, 0.8)),
    'map_blur_gamma': ('float', (0.4, 0.95)),
    'max_deposits_per_shipyard': ('int', (2, 8)),
    'end_return_extra_moves': ('int', (6, 15)),
    'ending_halite_threshold': ('int', (5, 30)),
    'end_start': ('int', (380, 390)),
    'cell_score_enemy_halite': ('float', (0.15, 0.5)),
    'cell_score_neighbour_discount': ('float', (0.45, 0.8)),
    'move_preference_base': ('int', (65, 90)),
    'move_preference_return': ('int', (75, 100)),
    'move_preference_mining': ('int', (90, 115)),
    'move_preference_hunting': ('int', (65, 95)),
    'cell_score_ship_halite': ('float', (0.0005, 0.001)),
    'fight_map_alpha': ('float', (1.1, 2.5)),
    'fight_map_sigma': ('float', (0.2, 0.8)),
    'fight_map_zeta': ('float', (0.2, 0.9))
}

first_genome = {'spawn_till': 287, 'spawn_step_multiplier': 0, 'min_ships': 29,
                'ship_spawn_threshold': 0.6514033017687603, 'shipyard_conversion_threshold': 0.9811205645720819,
                'ships_shipyards_threshold': 0.34972049028642327, 'shipyard_stop': 281, 'min_shipyard_distance': 4,
                'min_mining_halite': 37, 'convert_when_attacked_threshold': 309, 'max_halite_attack_shipyard': 75,
                'mining_score_alpha': 0.9688444416956035, 'mining_score_beta': 0.8567920727563001,
                'mining_score_gamma': 0.9852345353878476, 'hunting_threshold': 2.7496466266517787,
                'hunting_halite_threshold': 5, 'disable_hunting_till': 50, 'hunting_score_alpha': 0.7807246594018692,
                'hunting_score_gamma': 0.8730637123696053, 'return_halite': 422, 'max_ship_advantage': 7,
                'map_blur_sigma': 0.5024307902655017, 'map_blur_gamma': 0.4643344385358665,
                'max_deposits_per_shipyard': 3, 'end_return_extra_moves': 6, 'ending_halite_threshold': 17,
                'end_start': 380, 'cell_score_enemy_halite': 0.35063481489408443,
                'cell_score_neighbour_discount': 0.6036005046079449, 'move_preference_base': 87,
                'move_preference_return': 100, 'move_preference_mining': 110, 'move_preference_hunting': 85,
                'cell_score_ship_halite': 0.0006394592279623872, 'fight_map_alpha': 1.466352483965488,
                'fight_map_sigma': 0.4340374548084376, 'fight_map_zeta': 0.628238061509821}

second_genome = {'spawn_till': 309, 'spawn_step_multiplier': 0, 'min_ships': 29, 'ship_spawn_threshold': 0.1,
                 'shipyard_conversion_threshold': 1.9207293700980244, 'ships_shipyards_threshold': 0.34972049028642327,
                 'shipyard_stop': 281, 'min_shipyard_distance': 9, 'min_mining_halite': 20,
                 'convert_when_attacked_threshold': 307, 'max_halite_attack_shipyard': 93,
                 'mining_score_alpha': 0.9688444416956035, 'mining_score_beta': 0.8567920727563001,
                 'mining_score_gamma': 0.9867576882916219, 'hunting_threshold': 1.4591207568215503,
                 'hunting_halite_threshold': 0, 'disable_hunting_till': 50, 'hunting_score_alpha': 0.9085023751884967,
                 'hunting_score_gamma': 0.8730637123696053, 'return_halite': 422, 'max_ship_advantage': 7,
                 'map_blur_sigma': 0.5024307902655017, 'map_blur_gamma': 0.4643344385358665,
                 'max_deposits_per_shipyard': 2, 'end_return_extra_moves': 9, 'ending_halite_threshold': 15,
                 'end_start': 380, 'cell_score_enemy_halite': 0.35063481489408443,
                 'cell_score_neighbour_discount': 0.6421279345969618, 'move_preference_base': 92,
                 'move_preference_return': 99, 'move_preference_mining': 115, 'move_preference_hunting': 86,
                 'cell_score_ship_halite': 0.0006380361487177068, 'fight_map_alpha': 1.4474479033694228,
                 'fight_map_sigma': 0.45, 'fight_map_zeta': 0.4}

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
    else:
        pool = [first_genome, second_genome]
    best_genome = pool[0]
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
        print("Testing new genomes")
        pool.sort(key=lambda genome: determine_fitness(genome, best_genome), reverse=True)
        best_genome = pool[0]
        print("Saving new genomes")
        save_current_pool(pool)
        print("Best genome so far:")
        print(str(pool[0]))


def final_optimization():
    pool = []
    for saved_pool in os.listdir('evolutionary/finalpool/'):
        s_pool = pickle.load(open('evolutionary/finalpool/' + saved_pool, 'rb'))
        pool.append(s_pool[0])
    best_genome = pool[-1]
    POOL_SIZE = 25
    SELECTION_CAP = 10
    for epoch in range(100000):
        print("Creating generation %i" % epoch)
        print("Testing new genomes")
        pool.sort(key=lambda genome: determine_fitness(genome, best_genome), reverse=True)
        best_genome = pool[0]
        print("Saving new genomes")
        save_current_pool(pool)
        print("Best genome so far:")
        print(str(pool[0]))
        new_pool = pool[:SELECTION_CAP] if len(pool) >= SELECTION_CAP else pool
        old_pool = pool[SELECTION_CAP:] if len(pool) > SELECTION_CAP else []
        for genome in old_pool:
            if random() <= IGNORE_SELECTION_PROBABILITY:
                new_pool.append(genome)
        pool = new_pool.copy()
        print("Filling pool with current size of %i" % len(pool))
        for i in range(POOL_SIZE - len(new_pool)):
            pool.append(create_new_genome([choice(new_pool) for n in range(NB_PARENTS)]))


def play_game(genome1, genome2, genome3, genome4):
    env.reset(4)

    results = \
        evaluate("halite", [get_bot(genome1), get_bot(genome2), get_bot(genome3), get_bot(genome4)], env.configuration)[
            0]
    return results


def play_game_against_bot(genome1, bot):
    env.reset(4)

    results = \
        evaluate("halite", [get_bot(genome1), "evolutionary/bots/" + bot + ".py", get_bot(genome1),
                            "evolutionary/bots/" + bot + ".py"], env.configuration)[
            0]
    return results


def play_game_against_bots(genome1, bot1, bot2, bot3):
    env.reset(4)
    shuffled_indices = np.random.permutation(4)
    bots = [get_bot(genome1), "evolutionary/bots/" + bot1 + ".py", "evolutionary/bots/" + bot2 + ".py",
            "evolutionary/bots/" + bot3 + ".py"]
    bots[:] = [bots[i] for i in shuffled_indices]

    results = evaluate("halite", bots, env.configuration)[0]
    results[:] = [results[i] for i in shuffled_indices]
    return results


def get_bot(genome):
    bot = HaliteBot(genome)
    return lambda obs, config: bot.step(Board(obs, config))


def determine_fitness(genome, best_genome=first_genome):
    print("Determining fitness for a genome")
    print(genome)
    score = 0
    for i in range(4):
        if i > 0:
            print("Current score: %i" % score)
        # not optimal
        try:
            result = play_game_against_bots(genome, "uninstalllol1", "uninstalllol2", "optimusmine")
            standings = np.argsort(result)
            for place, agent in enumerate(standings):
                if agent == 0:
                    score += (place - 1.5) * 100000
            score += 2 * result[0] - result[1] - result[2] - result[3]

        except Exception as e:
            print("An error has occurred.")
            print(e)
            traceback.print_exc()
    print("Final score: %i" % score)
    return score


def save_current_pool(pool):
    pickle.dump(pool, open('evolutionary/genomes/' + str(datetime.now().strftime("%Y-%m-%d %H-%M")), 'wb'))


def load_pool(name):
    return pickle.load(open('evolutionary/genomes/' + name, 'rb'))


optimize()
