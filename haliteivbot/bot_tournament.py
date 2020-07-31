import logging
from random import randrange, sample

import numpy as np
from kaggle_environments import evaluate
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import Board
from trueskill import Rating, rate

from haliteivbot.bot import HaliteBot


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

    def play_tournament(self, games):
        for game in range(games):
            # TODO: make every player play nearly the same number of games
            mean_sigma = np.mean([rating.sigma for rating in self.ratings.values()])
            print("Playing game " + str(game + 1) + " of " + str(games))
            candidates = [self.bots[bot_index] for bot_index, rating in self.ratings.items() if
                          rating.sigma >= 0.75 * mean_sigma]
            if len(candidates) >= 4:
                bots = sample(candidates, 4)
            else:
                bots = candidates + sample([bot for bot in self.bots if bot not in candidates], 4 - len(candidates))
            standings = self.play_game(bots)
            new_ratings = rate([[self.ratings[self.bot_to_idx[bot]]] for bot in bots], ranks=standings)
            for i, bot in enumerate(bots):
                self.ratings[self.bot_to_idx[bot]] = new_ratings[i][0]
            print(sorted(self.ratings.items(), key=lambda item: item[1].mu - 3 * item[1].sigma, reverse=True))
        print([(self.bots[idx], rating) for idx, rating in self.ratings.items()])
        return [self.bots[bot_index] for bot_index, _ in
                sorted(self.ratings.items(), key=lambda item: item[1].mu - 1.5 * item[1].sigma,
                       reverse=True)]  # only subtract 1.5 * sigma as long as the number of games each bot plays varies greatly
