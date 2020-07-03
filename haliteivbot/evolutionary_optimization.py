import os
import pickle
from datetime import datetime
from random import random, choice

import numpy as np
from kaggle_environments import evaluate
from kaggle_environments import make

from haliteivbot.bot import HaliteBot, Board

MUTATION_PROBABILITY = 0.15
POOL_SIZE = 15
SELECTION_CAP = 5  # take the fittest five genomes of a generation
IGNORE_SELECTION_PROBABILITY = 0.1  # the probability to let another genome survive
NB_PARENTS = 4

POOL_NAME = ""

hyperparameters = {
    'spawn_till': ('int', (100, 390)),
    'spawn_step_multiplier': ('int', (0, 30)),
    'min_ships': ('int', (8, 40)),
    'ship_spawn_threshold': ('float', (0.1, 4.0)),
    'shipyard_conversion_threshold': ('float', (0.3, 10)),
    'shipyard_stop': ('int', (100, 380)),
    'min_shipyard_distance': ('int', (0, 35)),
    'mining_threshold': ('float', (0.2, 5.0)),
    'mining_decay': ('float', (-0.05, 0)),
    'min_mining_halite': ('int', (1, 30)),
    'return_halite': ('float', (3.0, 30.0)),
    'return_halite_decay': ('float', (-0.08, 0)),
    'min_return_halite': ('float', (0, 3.0)),
    'exploring_window_size': ('int', (1, 15)),
    'convert_when_attacked_threshold': ('int', (100, 600)),
    'distance_penalty': ('float', (0.01, 3.0)),
}

first_genome = {
    'spawn_till': 351,
    'spawn_step_multiplier': 0,
    'min_ships': 10,
    'ship_spawn_threshold': 0.4,
    'shipyard_conversion_threshold': 2,
    'shipyard_stop': 189,
    'min_shipyard_distance': 7,
    'mining_threshold': 1.30,
    'mining_decay': 0.0,
    'min_mining_halite': 1,
    'return_halite': 3.0,
    'return_halite_decay': -0.00552,
    'min_return_halite': 0.0,
    'exploring_window_size': 6,
    'convert_when_attacked_threshold': 359,
    'distance_penalty': 0.19172389911637758
}

second_genome = {
    'spawn_till': 314,
    'spawn_step_multiplier': 1,
    'min_ships': 20,
    'ship_spawn_threshold': 0.2,
    'shipyard_conversion_threshold': 0.4,
    'shipyard_stop': 187,
    'min_shipyard_distance': 7,
    'mining_threshold': 1.4906584887790534,
    'mining_decay': -0.001,
    'min_mining_halite': 6,
    'return_halite': 3.1,
    'return_halite_decay': 0.0,
    'min_return_halite': 0.121,
    'exploring_window_size': 6,
    'convert_when_attacked_threshold': 340,
    'distance_penalty': 0.19172389911637758}

env = make("halite", configuration={"size": 21, "startingHalite": 5000}, debug=False)


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
        result = play_game(genome, best_genome, best_genome,
                           genome)  # TODO: add support for multiple genomes to be tested
        score += result[0] - result[1] - result[2] + result[3]
    print("Final score: %i" % score)
    return score


def save_current_pool(pool):
    pickle.dump(pool, open('evolutionary/genomes/' + str(datetime.now().strftime("%Y-%m-%d %H-%M")), 'wb'))


def load_pool(name):
    return pickle.load(open('evolutionary/genomes/' + name, 'rb'))


optimize()
