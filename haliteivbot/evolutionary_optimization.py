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
    'max_ship_advantage': ('int', (-5, 10)),
    'map_blur_sigma': ('float', (0.15, 0.8)),
    'map_blur_gamma': ('float', (0.4, 0.95)),
    'max_deposits_per_shipyard': ('int', (2, 8)),
    'end_return_extra_moves': ('int', (6, 15)),
    'ending_halite_threshold': ('int', (5, 30)),
    'end_start': ('int', (380, 390)),
    'cell_score_enemy_halite': ('float', (0.15, 0.5)),
    'cell_score_neighbour_discount': ('float', (0.45, 0.8)),
    'move_preference_base': ('int', (175, 210)),
    'move_preference_return': ('int', (190, 230)),
    'move_preference_mining': ('int', (225, 275)),
    'move_preference_hunting': ('int', (160, 200)),
    'cell_score_ship_halite': ('float', (0.0005, 0.001)),
    'fight_map_alpha': ('float', (1.1, 2.5)),
    'fight_map_sigma': ('float', (0.2, 0.8)),
    'fight_map_zeta': ('float', (0.2, 0.9))
}

first_genome = {
    'spawn_till': 310,
    'spawn_step_multiplier': 0,
    'min_ships': 25,
    'ship_spawn_threshold': 0.6514033017687603,
    'shipyard_conversion_threshold': 1.9207293700980244,
    'ships_shipyards_threshold': 0.3572482485104553,
    'shipyard_stop': 294,
    'min_shipyard_distance': 7,
    'min_mining_halite': 20,
    'convert_when_attacked_threshold': 387,
    'max_halite_attack_shipyard': 93,
    'mining_score_alpha': 0.9689375743168729,
    'mining_score_beta': 0.8856409409897609,
    'mining_score_gamma': 0.9860073895543661,
    'hunting_threshold': 2.5,
    'hunting_halite_threshold': 0,
    'disable_hunting_till': 80,
    'hunting_score_alpha': 0.5,
    'hunting_score_gamma': 0.98,
    'return_halite': 771,
    'max_ship_advantage': 5,
    'map_blur_sigma': 0.4767363109968698,
    'map_blur_gamma': 0.5407113040457792,
    'max_deposits_per_shipyard': 3,
    'end_return_extra_moves': 8,
    'ending_halite_threshold': 15,
    'end_start': 380,
    'cell_score_enemy_halite': 0.3891021527805018,
    'cell_score_neighbour_discount': 0.637723836164392,
    'move_preference_base': 193,
    'move_preference_return': 206,
    'move_preference_mining': 250,
    'move_preference_hunting': 161,
    'cell_score_ship_halite': 0.0005462803359757412,
    'fight_map_alpha': 1.5,
    'fight_map_sigma': 0.5,
    'fight_map_zeta': 0.4
}

second_genome = {
    'spawn_till': 310,
    'spawn_step_multiplier': 2,
    'min_ships': 23,
    'ship_spawn_threshold': 0.6514033017687603,
    'shipyard_conversion_threshold': 1.9207293700980244,
    'ships_shipyards_threshold': 0.3572482485104553,
    'shipyard_stop': 294,
    'min_shipyard_distance': 9,
    'min_mining_halite': 33,
    'convert_when_attacked_threshold': 387,
    'max_halite_attack_shipyard': 93,
    'mining_score_alpha': 0.9689375743168729,
    'mining_score_beta': 0.8856409409897609,
    'mining_score_gamma': 0.9860073895543661,
    'hunting_threshold': 2.5,
    'hunting_halite_threshold': 0,
    'disable_hunting_till': 50,
    'hunting_score_alpha': 0.7,
    'hunting_score_gamma': 0.8662616356575497,
    'return_halite': 771,
    'max_ship_advantage': 5,
    'map_blur_sigma': 0.4767363109968698,
    'map_blur_gamma': 0.5407113040457792,
    'max_deposits_per_shipyard': 3,
    'end_return_extra_moves': 8,
    'ending_halite_threshold': 15,
    'end_start': 380,
    'cell_score_enemy_halite': 0.3891021527805018,
    'cell_score_neighbour_discount': 0.637723836164392,
    'move_preference_base': 193,
    'move_preference_return': 206,
    'move_preference_mining': 250,
    'move_preference_hunting': 161,
    'cell_score_ship_halite': 0.0005462803359757412,
    'fight_map_alpha': 1.4,
    'fight_map_sigma': 0.45,
    'fight_map_zeta': 0.5
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
            result = play_game_against_bot(genome,
                                           "uninstalllol1")  # TODO: add support for multiple genomes to be tested
            standings = np.argsort(result)
            for place, agent in enumerate(standings):
                if agent % 2 == 0:
                    score += place * 10000
                else:
                    score -= place * 10000
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


optimize()
