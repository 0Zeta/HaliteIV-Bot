from random import randrange

from kaggle_environments import make, evaluate

from haliteivbot.rule_based.bot import HaliteBot
from haliteivbot.rule_based.evolutionary_optimization import wrap_bot

if __name__ == '__main__':
    genome = {
        'cell_score_enemy_halite': 0.32555198341306124,
        'cell_score_neighbour_discount': 0.5884688910982244,
        'cell_score_ship_halite': 0.0005,
        'conflict_map_alpha': 1.4911687759755472,
        'conflict_map_sigma': 0.7102895212248723,
        'conflict_map_zeta': 0.834025988173528,
        'convert_when_attacked_threshold': 343,
        'disable_hunting_till': 89,
        'dominance_map_medium_radius': 5,
        'dominance_map_medium_sigma': 0.36913948907204935,
        'dominance_map_small_radius': 3,
        'dominance_map_small_sigma': 0.1527570414981133,
        'end_return_extra_moves': 8,
        'end_start': 380,
        'ending_halite_threshold': 23,
        'hunting_halite_threshold': 3,
        'hunting_score_alpha': 0.7748086187292644,
        'hunting_score_beta': 2.9702041344806904,
        'hunting_score_delta': 0.8,
        'hunting_score_gamma': 0.9155724342094896,
        'hunting_threshold': 1.9241957270147425,
        'map_blur_gamma': 0.447176685485112,
        'map_blur_sigma': 0.5272776436703535,
        'max_deposits_per_shipyard': 2,
        'max_halite_attack_shipyard': 83,
        'max_ship_advantage': 8,
        'max_shipyard_distance': 11,
        'min_mining_halite': 50,
        'min_ships': 28,
        'min_shipyard_distance': 1,
        'mining_score_alpha': 0.9324461144707071,
        'mining_score_beta': 0.99,
        'mining_score_gamma': 0.9823723236887304,
        'move_preference_base': 107,
        'move_preference_hunting': 106,
        'move_preference_mining': 131,
        'move_preference_return': 120,
        'return_halite': 782,
        'ship_spawn_threshold': 1.0323492495068471,
        'ships_shipyards_threshold': 0.31629693540968523,
        'shipyard_conversion_threshold': 1.3178552873143363,
        'shipyard_guarding_min_dominance': 3.660357949387156,
        'shipyard_min_dominance': 5.395535965563704,
        'shipyard_stop': 258,
        'spawn_min_dominance': 5.159993027190642,
        'spawn_step_multiplier': 0,
        'spawn_till': 271
    }


    def run_game():
        agent_count = 4
        env = make("halite", configuration={"size": 21, "startingHalite": 5000}, debug=True)
        env.configuration['randomSeed'] = randrange((1 << 32) - 1)
        info = env.reset(agent_count)
        bot1 = HaliteBot(genome)
        bot2 = HaliteBot(genome)
        results = evaluate("halite", [wrap_bot(bot1), "haliteivbot/evolutionary/bots/uninstalllol3.py", wrap_bot(bot2),
                                      "haliteivbot/evolutionary/bots/uninstalllol3.py"], env.configuration)[0]


    run_game()
