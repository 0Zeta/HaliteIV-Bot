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
    'spawn_till': ('int', (200, 390)),
    'spawn_step_multiplier': ('int', (0, 30)),
    'min_ships': ('int', (8, 40)),
    'ship_spawn_threshold': ('float', (0.1, 4.0)),
    'shipyard_conversion_threshold': ('float', (0.3, 10)),
    'ships_shipyards_threshold': ('float', (0.05, 1.2)),
    'shipyard_stop': ('int', (200, 390)),
    'min_shipyard_distance': ('int', (0, 35)),
    'mining_threshold': ('float', (0.4, 15.0)),
    'mining_decay': ('float', (-0.05, 0)),
    'min_mining_halite': ('int', (1, 30)),
    'return_halite': ('float', (3.0, 30.0)),
    'return_halite_decay': ('float', (-0.08, 0)),
    'min_return_halite': ('float', (0, 3.0)),
    'convert_when_attacked_threshold': ('int', (100, 600)),
    'max_halite_attack_shipyard': ('int', (10, 250)),
    'mining_score_alpha': ('float', (0.4, 0.9)),
    'mining_score_gamma': ('float', (0.92, 0.999))
}

first_genome = {
    'spawn_till': 280,
    'spawn_step_multiplier': 0,
    'min_ships': 20,
    'ship_spawn_threshold': 1.040101225356633,
    'shipyard_conversion_threshold': 2.8,
    'ships_shipyards_threshold': 0.4,
    'shipyard_stop': 364,
    'min_shipyard_distance': 10,
    'mining_threshold': 10.0068566355668074,
    'mining_decay': 0,
    'min_mining_halite': 5,
    'return_halite': 4.804004233593588,
    'return_halite_decay': 0.0,
    'min_return_halite': 0.0,
    'convert_when_attacked_threshold': 399,
    'max_halite_attack_shipyard': 20,
    'mining_score_alpha': 0.5,
    'mining_score_gamma': 0.98
}

second_genome = {
    'spawn_till': 350,
    'spawn_step_multiplier': 0,
    'min_ships': 25,
    'ship_spawn_threshold': 0.9,
    'shipyard_conversion_threshold': 2.0,
    'ships_shipyards_threshold': 0.8,
    'shipyard_stop': 364,
    'min_shipyard_distance': 8,
    'mining_threshold': 8,
    'mining_decay': 0,
    'min_mining_halite': 10,
    'return_halite': 3.2,
    'return_halite_decay': 0.0,
    'min_return_halite': 0.0,
    'convert_when_attacked_threshold': 300,
    'max_halite_attack_shipyard': 100,
    'mining_score_alpha': 0.3,
    'mining_score_gamma': 0.97
}

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
