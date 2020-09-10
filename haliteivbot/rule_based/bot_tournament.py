import logging
import traceback
from multiprocessing import Process, Queue
from random import randrange, sample

import numpy as np
from kaggle_environments import evaluate
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Board
from trueskill import Rating, rate

from haliteivbot.rule_based.bot import HaliteBot


def wrap_bot(bot):
    return lambda obs, config: bot.step(Board(obs, config), obs)


class Tournament(object):

    def __init__(self, bots):
        self.ratings = {i: Rating(mu=600, sigma=50) for i in range(len(bots))}
        self.bots = bots
        self.bot_to_idx = dict()
        for index, bot in enumerate(bots):
            if isinstance(bot, str):
                self.bot_to_idx[bot] = index
            else:
                self.bot_to_idx[bot['evo_id']] = index

    def bot_to_index(self, bot):
        if isinstance(bot, str):
            return self.bot_to_idx[bot]
        else:
            return self.bot_to_idx[bot['evo_id']]

    def play_game(self, bots, queue):
        try:
            env = make("halite", configuration={"size": 21, 'randomSeed': randrange((1 << 32) - 1), "agentTimeout": 90,
                                                "actTimeout": 18, "runTimeout": 36000}, debug=True)
            env.reset(4)
            shuffled_indices = np.random.permutation(4)
            bots[:] = [bots[i] for i in shuffled_indices]
            results = evaluate("halite", [wrap_bot(HaliteBot(bot)) if isinstance(bot, dict) else bot for bot in bots],
                               env.configuration)[0]
            results[:] = [results[i] for i in shuffled_indices]
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = -1000
                    logging.critical("An error occurred with bot " + str(self.bots[self.bot_to_index(bots[i])]) + ".")
                    logging.critical(results)
            standings = 3 - np.argsort(results)
            queue.put((bots, standings))
        except Exception as exception:
            logging.critical("An error occurred.")
            print(exception)
            queue.put((bots, [0, 0, 0, 0]))

    def play_tournament(self, rounds):
        for round in range(rounds):
            print("Starting round {}/{}".format(round + 1, rounds))
            round_bots = sample(self.bots, k=len(self.bots))
            games_per_round = len(self.bots) // 4
            queue = Queue()
            processes = []
            for game in range(games_per_round):
                bots = round_bots[game * 4:(game + 1) * 4]
                p = Process(target=self.play_game, args=(bots, queue))
                processes.append(p)
                p.start()
            # for process in processes:
            #     process.join()
            for _ in range(games_per_round):
                result = queue.get()
                bots = result[0]
                standings = result[1]
                new_ratings = rate([[self.ratings[self.bot_to_index(bot)]] for bot in bots], ranks=standings)
                for i, bot in enumerate(bots):
                    self.ratings[self.bot_to_index(bot)] = new_ratings[i][0]
            print([(idx, rating) if not isinstance(self.bots[idx], str) else (
                self.bots[idx].replace('evolutionary/bots/', '').replace('.py', ''), rating) for idx, rating in
                   sorted(self.ratings.items(), key=lambda item: item[1].mu, reverse=True)])

        return [self.bots[bot_index] for bot_index, _ in
                sorted(self.ratings.items(), key=lambda item: item[1].mu, reverse=True)]


class EarlyTournament(Tournament):

    def __init__(self, bots, steps):
        super().__init__(bots)
        self.steps = steps

    def get_fitness(self, step_data):
        results = []
        for player in range(4):
            halite = step_data['observation']['players'][player][0]
            shipyard_count = len(step_data['observation']['players'][player][1])
            ship_count = len(step_data['observation']['players'][player][2])
            cargo = sum([ship_data[1] for ship_data in step_data['observation']['players'][player][2].values()])

            results.append(halite + shipyard_count * 1200 + ship_count * 500 + cargo * 0.7)
        return results

    def play_game(self, bots, queue):
        try:
            env = make("halite", configuration={"size": 21, 'randomSeed': randrange((1 << 32) - 1), "agentTimeout": 90,
                                                "actTimeout": 18, "runTimeout": 36000, "episodeSteps": self.steps},
                       debug=True)
            env.reset(4)
            shuffled_indices = np.random.permutation(4)
            bots[:] = [bots[i] for i in shuffled_indices]
            env.run([wrap_bot(HaliteBot(bot)) if isinstance(bot, dict) else bot for bot in bots])
            results = self.get_fitness(env.steps[-1][0])
            results[:] = [results[i] for i in shuffled_indices]
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = -1000
                    logging.critical("An error occurred with bot " + str(self.bots[self.bot_to_index(bots[i])]) + ".")
                    logging.critical(results)
            standings = 3 - np.argsort(results)
            queue.put((bots, standings))
        except Exception as exception:
            logging.critical("An error occurred.")
            print(exception)
            traceback.print_exc()
            queue.put((bots, [0, 0, 0, 0]))
