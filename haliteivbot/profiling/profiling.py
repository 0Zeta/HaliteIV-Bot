from random import randrange

from kaggle_environments import make, evaluate

from haliteivbot.rule_based.bot import HaliteBot
from haliteivbot.rule_based.bot_tournament import wrap_bot

if __name__ == '__main__':
    genome = {
        'cell_score_enemy_halite': 0.4715114974967425,
        'cell_score_neighbour_discount': 0.6103114250591687,
        'cell_score_ship_halite': 0.0006146935882821743,
        'conflict_map_alpha': 1.5582849784278563,
        'conflict_map_sigma': 0.7972236445783817,
        'conflict_map_zeta': 0.8625720662620574,
        'convert_when_attacked_threshold': 600,
        'disable_hunting_till': 66,
        'dominance_map_halite_clip': 350,
        'dominance_map_medium_radius': 5,
        'dominance_map_medium_sigma': 0.01,
        'dominance_map_small_radius': 3,
        'dominance_map_small_sigma': 0.6398185917717716,
        'end_return_extra_moves': 6,
        'end_start': 381,
        'ending_halite_threshold': 27,
        'hunting_halite_threshold': 13,
        'hunting_min_ships': 13,
        'hunting_score_alpha': 1.1279514150456798,
        'hunting_score_beta': 2.5942517199955524,
        'hunting_score_delta': 0.5316893166228572,
        'hunting_score_gamma': 0.948335898856567,
        'hunting_threshold': 5.923173510655287,
        'map_blur_gamma': 0.8013766949565679,
        'map_blur_sigma': 0.693995126086896,
        'max_halite_attack_shipyard': 220,
        'max_hunting_ships_per_direction': 2,
        'max_ship_advantage': 6,
        'max_shipyard_distance': 11,
        'max_shipyards': 2,
        'min_mining_halite': 43,
        'min_ships': 25,
        'min_shipyard_distance': 3,
        'mining_score_alpha': 0.99,
        'mining_score_beta': 0.9677377968883031,
        'mining_score_dominance_clip': 3.7917976330210568,
        'mining_score_dominance_norm': 0.5259866301386806,
        'mining_score_gamma': 0.99,
        'move_preference_base': 109,
        'move_preference_hunting': 113,
        'move_preference_longest_axis': 12,
        'move_preference_mining': 130,
        'move_preference_return': 115,
        'move_preference_stay_on_shipyard': -112,
        'return_halite': 2509,
        'ship_spawn_threshold': 0.8698654657611027,
        'ships_shipyards_threshold': 0.05,
        'shipyard_abandon_dominance': -3.299048786606774,
        'shipyard_conversion_threshold': 2.589602412992243,
        'shipyard_guarding_attack_probability': 0.7405296328744754,
        'shipyard_guarding_min_dominance': 6.869301077098315,
        'shipyard_min_dominance': 6.976792091761268,
        'shipyard_start': 48,
        'shipyard_stop': 271,
        'spawn_min_dominance': 5.01915987533356,
        'spawn_till': 312
    }


    def run_game():
        agent_count = 4
        env = make("halite", configuration={"size": 21}, debug=True)
        env.configuration['randomSeed'] = randrange((1 << 32) - 1)
        info = env.reset(agent_count)
        bot1 = HaliteBot(genome)
        bot2 = HaliteBot(genome)
        results = evaluate("halite", [wrap_bot(bot1), "haliteivbot/evolutionary/bots/uninstalllol3.py", wrap_bot(bot2),
                                      "haliteivbot/evolutionary/bots/uninstalllol3.py"], env.configuration)[0]


    run_game()
