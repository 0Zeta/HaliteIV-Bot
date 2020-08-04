import logging
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
        self.env = make("halite", configuration={"size": 21, "startingHalite": 5000}, debug=True)
        self.bots = bots
        self.bot_to_idx = {bot: index for index, bot in enumerate(bots)}

    def play_game(self, bots):
        try:
            self.env.configuration['randomSeed'] = randrange((1 << 32) - 1)
            self.env.reset(4)
            shuffled_indices = np.random.permutation(4)
            bots[:] = [bots[i] for i in shuffled_indices]
            results = evaluate("halite", [wrap_bot(HaliteBot(bot)) if isinstance(bot, dict) else bot for bot in bots],
                               self.env.configuration)[0]
            results[:] = [results[i] for i in shuffled_indices]
            standings = 3 - np.argsort(results)
        except Exception as exception:
            logging.critical("An error occurred.")
            print(exception)
            return [0, 0, 0, 0]
        return standings

    def play_tournament(self, rounds):
        for round in range(rounds):
            round_bots = sample(self.bots, k=len(self.bots))
            games_per_round = len(self.bots) // 4
            for game in range(games_per_round):
                print("Playing game {}/{} of round {}/{}".format(game + 1, games_per_round, round + 1, rounds))
                bots = round_bots[game * 4:(game + 1) * 4]
                standings = self.play_game(bots)
                new_ratings = rate([[self.ratings[self.bot_to_idx[bot]]] for bot in bots], ranks=standings)
                for i, bot in enumerate(bots):
                    self.ratings[self.bot_to_idx[bot]] = new_ratings[i][0]
                print([(idx, rating) if not isinstance(self.bots[idx], str) else (
                    self.bots[idx].replace('evolutionary/bots/', '').replace('.py', ''), rating) for idx, rating in
                       sorted(self.ratings.items(), key=lambda item: item[1].mu, reverse=True)])

        print([(self.bots[idx], rating) for idx, rating in self.ratings.items()])
        return [self.bots[bot_index] for bot_index, _ in
                sorted(self.ratings.items(), key=lambda item: item[1].mu, reverse=True)]
