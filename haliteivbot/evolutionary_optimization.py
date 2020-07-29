import os
import pickle
import traceback
from datetime import datetime
from random import random, randrange, choice

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
    'min_shipyard_distance': ('int', (0, 10)),
    'max_shipyard_distance': ('int', (11, 20)),
    'shipyard_min_dominance': ('float', (4, 7)),
    'shipyard_guarding_min_dominance': ('float', (2, 7)),
    'spawn_min_dominance': ('float', (3.5, 8)),
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
    'hunting_score_beta': ('float', (1.2, 4)),
    'hunting_score_gamma': ('float', (0.75, 0.99)),
    'hunting_score_delta': ('float', (0.5, 1.5)),
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
    'move_preference_base': ('int', (85, 110)),
    'move_preference_return': ('int', (105, 125)),
    'move_preference_mining': ('int', (110, 135)),
    'move_preference_hunting': ('int', (85, 115)),
    'cell_score_ship_halite': ('float', (0.0005, 0.001)),
    'conflict_map_alpha': ('float', (1.1, 1.8)),
    'conflict_map_sigma': ('float', (0.2, 0.8)),
    'conflict_map_zeta': ('float', (0.2, 0.9)),
    'dominance_map_small_sigma': ('float', (0.1, 0.8)),
    'dominance_map_medium_sigma': ('float', (0.2, 0.9)),
    'dominance_map_small_radius': ('int', (3, 4)),
    'dominance_map_medium_radius': ('int', (5, 6))
}

first_genome = {'spawn_till': 287, 'spawn_step_multiplier': 0, 'min_ships': 29,
                'ship_spawn_threshold': 0.6514033017687603, 'shipyard_conversion_threshold': 0.3,
                'ships_shipyards_threshold': 0.38453266848611917, 'shipyard_stop': 281, 'min_shipyard_distance': 4,
                'min_mining_halite': 37, 'convert_when_attacked_threshold': 309, 'max_halite_attack_shipyard': 90,
                'mining_score_alpha': 0.9271257605704805, 'mining_score_beta': 0.8680308293965289,
                'mining_score_gamma': 0.9852345353878476, 'hunting_threshold': 2.340539636989916,
                'hunting_halite_threshold': 2, 'disable_hunting_till': 50, 'hunting_score_alpha': 0.7129313058301051,
                'hunting_score_gamma': 0.8730637123696053, 'return_halite': 422, 'max_ship_advantage': 8,
                'map_blur_sigma': 0.5040818209754316, 'map_blur_gamma': 0.45651384176160353,
                'max_deposits_per_shipyard': 3, 'end_return_extra_moves': 6, 'ending_halite_threshold': 17,
                'end_start': 380, 'cell_score_enemy_halite': 0.35063481489408443,
                'cell_score_neighbour_discount': 0.616051369788526, 'move_preference_base': 110,
                'move_preference_return': 120, 'move_preference_mining': 131, 'move_preference_hunting': 106,
                'cell_score_ship_halite': 0.0006394592279623872, 'conflict_map_alpha': 1.550464810350371,
                'conflict_map_sigma': 0.4754435936208387, 'conflict_map_zeta': 0.628238061509821,
                'dominance_map_small_sigma': 0.3,
                'dominance_map_medium_sigma': 0.4,
                'dominance_map_small_radius': 3,
                'dominance_map_medium_radius': 5,
                'max_shipyard_distance': 10,
                'shipyard_min_dominance': 5,
                'shipyard_guarding_min_dominance': 4.5,
                'spawn_min_dominance': 4.5,
                'hunting_score_beta': 2.5,
                'hunting_score_delta': 0.8
                }

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
                 'cell_score_neighbour_discount': 0.6421279345969618, 'move_preference_base': 112,
                 'move_preference_return': 119, 'move_preference_mining': 135, 'move_preference_hunting': 106,
                 'cell_score_ship_halite': 0.0006380361487177068, 'conflict_map_alpha': 1.4474479033694228,
                 'conflict_map_sigma': 0.45, 'conflict_map_zeta': 0.4,
                 'dominance_map_small_sigma': 0.2,
                 'dominance_map_medium_sigma': 0.3,
                 'dominance_map_small_radius': 3,
                 'dominance_map_medium_radius': 5,
                 'max_shipyard_distance': 9,
                 'shipyard_min_dominance': 4,
                 'shipyard_guarding_min_dominance': 5.5,
                 'spawn_min_dominance': 5,
                 'hunting_score_beta': 2.7,
                 'hunting_score_delta': 0.5
                 }

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
    bot1 = HaliteBot(genome1)
    bot2 = HaliteBot(genome2)
    bot3 = HaliteBot(genome3)
    bot4 = HaliteBot(genome4)
    results = \
        evaluate("halite", [wrap_bot(bot1), wrap_bot(bot2), wrap_bot(bot3), wrap_bot(bot4)], env.configuration)[
            0]
    return results


def play_game_against_bot(genome1, bot):
    env.reset(4)
    bot1 = HaliteBot(genome1)
    bot2 = HaliteBot(genome1)
    results = \
        evaluate("halite", [wrap_bot(bot1), "evolutionary/bots/" + bot + ".py", wrap_bot(bot2),
                            "evolutionary/bots/" + bot + ".py"], env.configuration)[
            0]
    return results


def play_game_against_bots(genome1, bot1, bot2, bot3):
    env.reset(4)
    bot = HaliteBot(genome1)
    shuffled_indices = np.random.permutation(4)
    bots = [wrap_bot(bot), "evolutionary/bots/" + bot1 + ".py", "evolutionary/bots/" + bot2 + ".py",
            "evolutionary/bots/" + bot3 + ".py"]
    bots[:] = [bots[i] for i in shuffled_indices]

    results = evaluate("halite", bots, env.configuration)[0]
    results[:] = [results[i] for i in shuffled_indices]
    return results


def wrap_bot(bot):
    return lambda obs, config: bot.step(Board(obs, config), obs)


def determine_fitness(genome, best_genome=first_genome):
    print("Determining fitness for a genome")
    print(genome)
    score = 0
    for i in range(4):
        env.configuration['randomSeed'] = randrange((1 << 32) - 1)
        if i > 0:
            print("Current score: %i" % score)
        # not optimal
        try:
            result = play_game_against_bot(genome, "uninstalllol3")
            print(result)
            standings = np.argsort(result)
            for place, agent in enumerate(standings):
                if agent % 2 == 0:
                    score += place * 100000
                else:
                    score -= place * 100000
            score += result[0] - result[1] + result[2] - result[3]

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


if __name__ == '__main__':
    optimize()
