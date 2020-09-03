import logging
from enum import Enum
from math import floor, ceil
from random import random

from kaggle_environments.envs.halite.helpers import Shipyard, Ship, Board, ShipyardAction

# from haliteivbot.display_utils import display_matrix, display_dominance_map
from haliteivbot.rule_based.utils import *

logging.basicConfig(level=logging.INFO)

PARAMETERS = {
    'cargo_map_halite_norm': 100,
    'cell_score_dominance': 1.986880839809644,
    'cell_score_enemy_halite': 0.4,
    'cell_score_farming': -100,
    'cell_score_neighbour_discount': 0.6691096641934141,
    'cell_score_ship_halite': 0.0006229108666303259,
    'convert_when_attacked_threshold': 489,
    'disable_hunting_till': 75,
    'dominance_map_halite_clip': 150,
    'dominance_map_medium_radius': 5,
    'dominance_map_medium_sigma': 2.8,
    'dominance_map_small_radius': 3,
    'dominance_map_small_sigma': 1.5,
    'early_second_shipyard': 25,
    'end_return_extra_moves': 7,
    'end_start': 382,
    'ending_halite_threshold': 10,
    'farming_end': 355,
    'farming_start': 1,
    'farming_start_shipyards': 2,
    'guarding_aggression_radius': 6,
    'guarding_end': 375,
    'guarding_max_distance_to_shipyard': 4,
    'guarding_max_ships_per_shipyard': 4,
    'guarding_min_distance_to_shipyard': 2,
    'guarding_norm': 0.65,
    'guarding_radius': 3,
    'guarding_radius2': 1,
    'guarding_ship_advantage_norm': 20,
    'guarding_stop': 342,
    'max_guarding_ships_per_target': 2,
    'harvest_threshold_alpha': 0.13268007586429212,
    'harvest_threshold_hunting_norm': 0.670907810006537,
    'hunting_halite_threshold': 0.04077647561190107,
    'hunting_max_group_distance': 5,
    'hunting_max_group_size': 1,
    'hunting_min_ships': 10,
    'hunting_proportion': 0.42990710071649985,
    'hunting_proportion_after_farming': 0.26923397840562785,
    'hunting_score_alpha': 1.0700236651908992,
    'hunting_score_beta': 2.391546761028965,
    'hunting_score_cargo_clip': 2.7903622542244086,
    'hunting_score_delta': 0.7152003820018653,
    'hunting_score_farming_position_penalty': 0.6589914665212904,
    'hunting_score_gamma': 0.9272782335536588,
    'hunting_score_halite_norm': 203,
    'hunting_score_hunt': 2,
    'hunting_score_intercept': 1.25,
    'hunting_score_iota': 0.5535210187640481,
    'hunting_score_kappa': 0.36442395827098567,
    'hunting_score_ship_bonus': 190,
    'hunting_score_ypsilon': 2,
    'hunting_score_zeta': 1.1452680492519223,
    'hunting_threshold': 7,
    'map_blur_gamma': 0.94,
    'map_blur_sigma': 0.3579575706817798,
    'map_ultra_blur': 1.25,
    'max_halite_attack_shipyard': 0,
    'max_hunting_ships_per_direction': 2,
    'max_ship_advantage': 25,
    'max_shipyard_distance': 7,
    'max_shipyards': 10,
    'min_enemy_shipyard_distance': 5,
    'min_mining_halite': 5,
    'min_ships': 19,
    'min_shipyard_distance': 6,
    'mining_score_alpha': 1,
    'mining_score_beta': 0.9964875995375948,
    'mining_score_dominance_clip': 4,
    'mining_score_dominance_norm': 0.3,
    'mining_score_farming_penalty': 0.01,
    'mining_score_gamma': 0.9766714280531774,
    'mining_score_juicy': 0.33770148976371134,
    'mining_score_juicy_end': 0.1,
    'mining_score_start_returning': 53,
    'move_preference_base': 94,
    'move_preference_block_shipyard': -192,
    'move_preference_constructing': 144,
    'move_preference_construction_guarding': 130,
    'move_preference_guarding': 98,
    'move_preference_guarding_stay': -99,
    'move_preference_hunting': 107,
    'move_preference_longest_axis': 10,
    'move_preference_mining': 125,
    'move_preference_return': 119,
    'move_preference_stay_on_shipyard': -61,
    'return_halite': 862,
    'second_shipyard_min_ships': 15,
    'second_shipyard_step': 30,
    'ship_spawn_threshold': 0.05,
    'ships_shipyards_threshold': 0.16,
    'shipyard_abandon_dominance': -25,
    'shipyard_conversion_threshold': 2.5,
    'shipyard_guarding_attack_probability': 0.35,
    'shipyard_guarding_min_dominance': -15,
    'shipyard_min_dominance': -2,
    'shipyard_min_population': 1.5,
    'shipyard_min_ship_advantage': -12,
    'shipyard_start': 180,
    'shipyard_stop': 250,
    'spawn_min_dominance': -2.0,
    'spawn_till': 270,
    'third_shipyard_min_ships': 18,
    'third_shipyard_step': 56
}

OPTIMAL_MINING_STEPS_TENSOR = [
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [5, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [6, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1], [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 6, 6, 5, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1], [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [9, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
     [10, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [10, 9, 8, 7, 7, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1], [11, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1]],
    [[3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [5, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [6, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
     [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1], [8, 6, 6, 5, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1], [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 7, 7, 6, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [10, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 7, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
     [11, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1], [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1]],
    [[4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [5, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [5, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [6, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [6, 5, 5, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1], [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1], [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [9, 7, 7, 6, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1], [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
     [10, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [10, 9, 8, 7, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1], [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [11, 10, 9, 8, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1]],
    [[4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [5, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [5, 4, 4, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [6, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [6, 5, 5, 4, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1], [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
     [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1], [8, 6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 7, 7, 6, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [10, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 7, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
     [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [11, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 8, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1], [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1]],
    [[5, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [5, 4, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [6, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [6, 5, 5, 4, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1], [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1], [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [9, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
     [10, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1], [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [11, 10, 9, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1], [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1]],
    [[5, 4, 4, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [6, 5, 5, 4, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1], [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
     [7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1], [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [10, 8, 8, 7, 6, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [11, 10, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1], [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
     [12, 11, 10, 9, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1], [13, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1]],
    [[6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [6, 5, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
     [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1], [7, 6, 6, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1], [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
     [10, 8, 8, 7, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1], [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
     [12, 10, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1], [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [13, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [13, 11, 10, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1, 1]],
    [[6, 5, 5, 4, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1], [7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
     [7, 6, 6, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1], [8, 7, 6, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [8, 7, 7, 6, 5, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 7, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1], [10, 8, 8, 7, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [11, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [11, 10, 9, 8, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1], [12, 10, 10, 9, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
     [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [13, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [13, 11, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1], [14, 12, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1]],
    [[7, 6, 5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1], [7, 6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [11, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 8, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1], [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
     [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [13, 11, 10, 9, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [13, 11, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
     [14, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1], [15, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1]],
    [[7, 6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1], [8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [8, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1], [10, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [11, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1], [11, 10, 9, 8, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1], [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [13, 11, 10, 10, 9, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
     [13, 11, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1], [14, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
     [15, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1], [15, 12, 12, 11, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1]],
    [[8, 7, 6, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
     [8, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1], [9, 8, 8, 7, 6, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [11, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1], [11, 10, 9, 9, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
     [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [13, 11, 10, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1], [13, 11, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
     [14, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1], [15, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [15, 12, 12, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [15, 13, 12, 11, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1, 1]],
    [[8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1], [8, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
     [9, 8, 8, 7, 6, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 9, 9, 8, 7, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1], [11, 10, 9, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [12, 11, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [13, 11, 10, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
     [13, 11, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1], [14, 12, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
     [15, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [15, 13, 12, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [15, 13, 12, 11, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1, 1], [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 1, 1, 1, 1, 1]],
    [[8, 7, 7, 6, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1, 1], [9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1],
     [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1], [9, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1], [11, 9, 9, 8, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 9, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1], [11, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [12, 11, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
     [13, 11, 10, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1], [13, 11, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
     [14, 12, 11, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1], [15, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [15, 13, 12, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 1, 1, 1, 1, 1],
     [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1], [15, 14, 13, 12, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1]],
    [[9, 8, 7, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1, 1], [9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1],
     [9, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 9, 9, 8, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1], [11, 10, 9, 9, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
     [11, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [12, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [12, 11, 10, 9, 9, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1], [13, 11, 10, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
     [13, 11, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1], [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [15, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [15, 13, 12, 11, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1], [15, 14, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 14, 13, 12, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1], [15, 15, 13, 12, 11, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1]],
    [[9, 8, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1], [9, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1], [11, 9, 9, 8, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [11, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [12, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1], [12, 11, 10, 9, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
     [13, 11, 10, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1], [13, 11, 11, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
     [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [15, 12, 11, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 1, 1, 1, 1, 1], [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 14, 13, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1], [15, 14, 13, 12, 11, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 15, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1], [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1]],
    [[9, 8, 8, 7, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1],
     [10, 9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1], [10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 8, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1], [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [11, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [12, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
     [12, 11, 10, 9, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1], [13, 11, 11, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
     [13, 11, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [15, 12, 11, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1], [15, 14, 13, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 14, 13, 12, 11, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1], [15, 15, 14, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
     [15, 15, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1], [15, 15, 15, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1]],
    [[10, 9, 8, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1, 1], [10, 9, 8, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1],
     [10, 9, 9, 8, 7, 7, 5, 4, 3, 1, 1, 1, 1, 1, 1], [11, 10, 9, 8, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1],
     [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [11, 10, 10, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [12, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1], [12, 11, 10, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1],
     [13, 11, 11, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1], [13, 11, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [15, 12, 12, 11, 10, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1], [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 14, 13, 12, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1], [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
     [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1], [15, 15, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
     [15, 15, 15, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1], [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1]],
    [[10, 9, 9, 8, 7, 6, 5, 4, 3, 1, 1, 1, 1, 1, 1], [10, 9, 9, 8, 7, 7, 6, 4, 3, 1, 1, 1, 1, 1, 1],
     [11, 10, 9, 8, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [11, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1], [12, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
     [12, 11, 10, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1, 1], [13, 11, 11, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
     [13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [15, 12, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1], [15, 13, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1], [15, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1],
     [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1], [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
     [15, 15, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1], [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 2, 1, 1, 1, 1],
     [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1], [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1]],
    [[10, 9, 9, 8, 8, 7, 6, 4, 3, 2, 1, 1, 1, 1, 1], [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1],
     [11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [11, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
     [12, 11, 10, 9, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1], [12, 11, 10, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1],
     [13, 11, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [14, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [15, 12, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1], [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 14, 13, 12, 10, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1], [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
     [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1], [15, 15, 14, 13, 11, 10, 8, 7, 6, 4, 2, 1, 1, 1, 1],
     [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1], [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
     [15, 15, 15, 14, 12, 11, 9, 7, 6, 4, 3, 1, 1, 1, 1], [15, 15, 15, 15, 13, 11, 9, 8, 6, 4, 3, 1, 1, 1, 1]],
    [[11, 10, 9, 9, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1, 1], [11, 10, 9, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1],
     [11, 10, 10, 9, 8, 7, 6, 5, 4, 2, 1, 1, 1, 1, 1], [12, 11, 10, 9, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1, 1],
     [12, 11, 10, 10, 9, 8, 7, 5, 4, 3, 1, 1, 1, 1, 1], [13, 11, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1],
     [13, 12, 11, 10, 9, 8, 7, 6, 4, 3, 1, 1, 1, 1, 1], [14, 12, 11, 10, 10, 8, 7, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 12, 12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 1, 1, 1], [15, 13, 12, 11, 10, 9, 8, 6, 5, 3, 2, 1, 1, 1, 1],
     [15, 13, 12, 11, 10, 9, 8, 6, 5, 4, 2, 1, 1, 1, 1], [15, 14, 13, 12, 11, 9, 8, 7, 5, 4, 2, 1, 1, 1, 1],
     [15, 15, 13, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1], [15, 15, 14, 12, 11, 10, 8, 7, 5, 4, 2, 1, 1, 1, 1],
     [15, 15, 14, 13, 11, 10, 9, 7, 6, 4, 2, 1, 1, 1, 1], [15, 15, 15, 13, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1],
     [15, 15, 15, 14, 12, 10, 9, 7, 6, 4, 3, 1, 1, 1, 1], [15, 15, 15, 14, 12, 11, 9, 8, 6, 4, 3, 1, 1, 1, 1],
     [15, 15, 15, 15, 13, 11, 9, 8, 6, 5, 3, 1, 1, 1, 1], [15, 15, 15, 15, 13, 11, 9, 8, 6, 5, 3, 1, 1, 1, 1]]]

BOT = None


class ShipType(Enum):
    MINING = 1
    RETURNING = 2
    HUNTING = 3
    SHIPYARD_GUARDING = 4
    GUARDING = 5
    DEFENDING = 6  # guarding ships that hunt nearby enemies
    CONVERTING = 7
    CONSTRUCTING = 8  # ships on their way to a good shipyard location
    CONSTRUCTION_GUARDING = 9
    ENDING = 10


class HaliteBot(object):

    def __init__(self, parameters):
        self.parameters = parameters
        self.config = None
        self.size = 21
        self.me = None
        self.player_id = 0

        self.halite = 5000
        self.ship_count = 1
        self.shipyard_count = 0
        self.shipyard_positions = []
        self.second_shipyard_ship = None
        self.first_guarding_ship = None
        self.next_shipyard_position = None
        self.blurred_halite_map = None
        self.average_halite_per_cell = 0
        self.rank = 0

        self.planned_moves = list()  # a list of positions where our ships will be in the next step
        self.planned_shipyards = list()
        self.ship_position_preferences = None
        self.ship_types = dict()
        self.mining_targets = dict()
        self.deposit_targets = dict()
        self.hunting_targets = dict()
        self.guarding_shipyards = dict()
        self.friendly_neighbour_count = dict()
        self.shipyard_guards = list()
        self.spawn_cost = 500

        self.enemies = list()

        self.returning_ships = list()
        self.mining_ships = list()
        self.hunting_ships = list()
        self.guarding_ships = list()

        if OPTIMAL_MINING_STEPS_TENSOR is None:
            self.optimal_mining_steps = create_optimal_mining_steps_tensor(1, 1, 1)
        else:
            self.optimal_mining_steps = OPTIMAL_MINING_STEPS_TENSOR

        create_navigation_lists(self.size)
        self.distances = get_distance_matrix()
        self.positions_in_reach_list, self.positions_in_reach_indices = compute_positions_in_reach()
        self.farthest_directions_indices = get_farthest_directions_matrix()
        self.farthest_directions = get_farthest_directions_list()
        create_radius_lists(self.parameters['dominance_map_small_radius'],
                            self.parameters['dominance_map_medium_radius'])
        self.farming_radius_list = create_radius_list(ceil(self.parameters['max_shipyard_distance'] / 2))

    def step(self, board: Board, obs):
        if self.me is None:
            self.player_id = board.current_player_id
            self.config = board.configuration
            self.size = self.config.size
        self.observation = obs
        self.me = board.current_player
        self.opponents = board.opponents
        self.ships = self.me.ships
        self.halite = self.me.halite
        self.step_count = board.step
        self.ship_count = len(self.ships)
        self.shipyard_count = len(self.me.shipyards)
        self.shipyard_guards.clear()

        self.enemies = [ship for player in board.players.values() for ship in player.ships if
                        player.id != self.player_id]
        self.enemy_positions = [TO_INDEX[ship.position] for ship in self.enemies]

        self.average_halite_per_cell = sum([halite for halite in self.observation['halite']]) / self.size ** 2
        self.average_halite_population = sum(
            [1 if halite > 0 else 0 for halite in self.observation['halite']]) / self.size ** 2
        self.nb_cells_in_farming_radius = len(self.farming_radius_list[0])

        self.blurred_halite_map = get_blurred_halite_map(self.observation['halite'], self.parameters['map_blur_sigma'])
        self.ultra_blurred_halite_map = get_blurred_halite_map(self.observation['halite'],
                                                               self.parameters['map_ultra_blur'])

        self.shipyard_positions = []
        for shipyard in self.me.shipyards:
            self.shipyard_positions.append(TO_INDEX[shipyard.position])

        players = [self.me] + self.opponents
        ranking = np.argsort([self.calculate_player_score(player) for player in players])[::-1]
        self.rank = int(np.where(ranking == 0)[0])
        self.player_ranking = dict()

        map_presence_ranks = np.argsort(
            [self.calculate_player_map_presence(player) for player in players])[::-1]
        self.map_presence_rank = int(np.where(map_presence_ranks == 0)[0])
        self.map_presence_ranking = dict()
        map_presence = self.calculate_player_map_presence(self.me)
        self.map_presence_diff = {opponent.id: map_presence - self.calculate_player_map_presence(opponent) for opponent
                                  in self.opponents}

        halite_ranks = np.argsort([player.halite for player in players])[::-1]
        self.halite_ranking = dict()

        for i, player in enumerate(players):
            self.player_ranking[player.id] = int(np.where(ranking == i)[0])
            self.map_presence_ranking[player.id] = int(np.where(map_presence_ranks == i)[0])
            self.halite_ranking[player.id] = int(np.where(halite_ranks == i)[0])

        self.ship_advantage = len(self.me.ships) - max([len(player.ships) for player in self.opponents] + [0])

        self.enemy_hunting_proportion = sum(
            [sum([1 for ship in player.ships if ship.halite <= 1]) for player in self.opponents if
             len(player.ships) > 0]) / max(1, sum([len(player.ships) for player in self.opponents]))

        need_halite = ((self.parameters['second_shipyard_step'] - 4) <= self.step_count <= self.parameters[
            'second_shipyard_step']) or (
                                  (self.parameters['third_shipyard_step'] - 4) <= self.step_count <= self.parameters[
                              'third_shipyard_step'])

        # Distances to enemy ships and shipyard connections
        self.enemy_distances = dict()
        self.guarded_shipyards = list()
        self.max_shipyard_connections = 0
        self.nb_connected_shipyards = 0
        for shipyard_position in self.shipyard_positions:
            min_distance = 20
            connections = 0
            for enemy_position in self.enemy_positions:
                distance = get_distance(shipyard_position, enemy_position)
                if distance < min_distance:
                    min_distance = distance
            self.enemy_distances[shipyard_position] = min_distance
            for shipyard2_pos in self.shipyard_positions:
                con_distance = get_distance(shipyard_position, shipyard2_pos)
                if shipyard_position != shipyard2_pos and self.parameters['min_shipyard_distance'] <= con_distance <= \
                        self.parameters['max_shipyard_distance']:
                    connections += 1
                    if self.max_shipyard_connections < connections:
                        self.max_shipyard_connections = connections
            if connections > 0:
                self.nb_connected_shipyards += 1

        self.farming_positions = []
        self.real_farming_points = []
        self.guarding_positions = []
        if self.shipyard_count == 0:
            # There is no shipyard, but we still need to mine.
            self.shipyard_distances = [3] * self.size ** 2
        else:
            # Compute distances to the next shipyard, farming and guarding positions:
            self.shipyard_distances = []
            guarding_radius = ((self.parameters['max_shipyard_distance'] + 1) if self.max_shipyard_connections >= 2 else \
                                   self.parameters['max_shipyard_distance'] - 1) - 1 + self.parameters[
                                  'guarding_radius2']
            farming_radius = (self.parameters['max_shipyard_distance'] if self.max_shipyard_connections >= 2 else \
                                  self.parameters['max_shipyard_distance'] - 2) - 1
            required_in_range = min(3,
                                    max(self.parameters['farming_start_shipyards'], self.max_shipyard_connections + 1))
            for pos in range(0, SIZE ** 2):
                min_distance = float('inf')
                in_guarding_range = 0
                in_farming_range = 0
                guard = False
                for shipyard_position in self.shipyard_positions:  # TODO: consider planned shipyards
                    distance = get_distance(pos, shipyard_position)
                    if distance < min_distance:
                        min_distance = distance
                    if distance <= self.parameters['guarding_radius']:
                        guard = True
                    if self.parameters['farming_start'] <= self.step_count:
                        if distance <= farming_radius:
                            in_guarding_range += 1
                            in_farming_range += 1
                        elif distance <= guarding_radius:
                            in_guarding_range += 1
                self.shipyard_distances.append(min_distance)

                if guard or (
                        self.parameters['farming_start'] <= self.step_count and in_guarding_range >= required_in_range):
                    self.guarding_positions.append(pos)
                if self.parameters['farming_start'] <= self.step_count <= self.parameters[
                    'farming_end'] and pos not in self.shipyard_positions and in_farming_range >= required_in_range:
                    self.farming_positions.append(pos)
                    point = Point.from_index(pos, SIZE)
                    if board.cells[point].halite > 0:
                        self.real_farming_points.append(point)

        if len(self.me.ships) > 0:
            self.small_dominance_map = get_dominance_map(self.me, self.opponents,
                                                         self.parameters['dominance_map_small_sigma'], 20,
                                                         self.parameters['dominance_map_halite_clip'])
            self.medium_dominance_map = get_dominance_map(self.me, self.opponents,
                                                          self.parameters['dominance_map_medium_sigma'], 80,
                                                          self.parameters['dominance_map_halite_clip'])
            self.cargo_map = get_cargo_map(self.me.ships, self.me.shipyards, self.parameters['cargo_map_halite_norm'])

        self.planned_moves.clear()
        self.spawn_limit_reached = self.reached_spawn_limit(board)
        self.harvest_threshold = self.calculate_harvest_threshold()
        self.positions_in_reach = []
        for ship in self.me.ships:
            self.positions_in_reach.extend(self.positions_in_reach_list[ship.position])
        self.positions_in_reach = list(set(self.positions_in_reach))
        if board.step > self.parameters['end_start']:
            for shipyard in self.me.shipyards:
                if shipyard.position in self.positions_in_reach:
                    self.positions_in_reach.extend([shipyard.position for _ in range(3)])
        nb_positions_in_reach = len(self.positions_in_reach)
        nb_shipyard_conversions = self.halite // self.config.convert_cost + 1  # TODO: Ships with enough halite can also convert to a shipyard
        self.available_shipyard_conversions = nb_shipyard_conversions - 1  # without cargo
        self.ship_to_index = {ship: ship_index for ship_index, ship in enumerate(self.me.ships)}
        self.position_to_index = dict()
        for position_index, position in enumerate(self.positions_in_reach):
            if position in self.position_to_index.keys():
                self.position_to_index[position].append(position_index)
            else:
                self.position_to_index[position] = [position_index]
        self.ship_position_preferences = np.full(
            shape=(self.ship_count, nb_positions_in_reach + nb_shipyard_conversions),
            fill_value=-999999)  # positions + convert "positions"
        for ship_index, ship in enumerate(self.me.ships):
            if ship.halite >= self.parameters[
                'convert_when_attacked_threshold'] and ship.halite + self.halite >= self.config.convert_cost:
                self.ship_position_preferences[ship_index,
                nb_positions_in_reach:] = -self.parameters[
                    'convert_when_attacked_threshold']
            for position in self.positions_in_reach_list[ship.position]:
                self.ship_position_preferences[
                    ship_index, self.position_to_index[position]] = self.calculate_cell_score(ship,
                                                                                              board.cells[position])
            if ship.cell.shipyard is not None:
                self.ship_position_preferences[ship_index, self.position_to_index[ship.position]] += self.parameters[
                    'move_preference_stay_on_shipyard'] if self.step_count > 12 else -200  # don't block the shipyard in the early game

        self.cargo = sum([0] + [ship.halite for ship in self.me.ships])
        self.planned_shipyards.clear()
        self.guarding_shipyards.clear()
        self.ship_types.clear()
        self.mining_targets.clear()
        self.deposit_targets.clear()
        enemy_cargo = sorted([ship.halite for ship in self.enemies])
        self.hunting_halite_threshold = enemy_cargo[
            floor(len(enemy_cargo) * self.parameters['hunting_halite_threshold'])] if len(enemy_cargo) > 0 else 0

        self.debug()

        self.determine_vulnerable_enemies()

        if self.handle_special_steps(board):
            return self.me.next_actions

        self.guard_shipyards(board)
        self.build_shipyards(board)

        self.spawn_cost = 2 * self.config.spawn_cost if need_halite or ((self.next_shipyard_position is not None and (
                ShipType.CONSTRUCTING in self.ship_types.values() or (
                board.cells[Point.from_index(self.next_shipyard_position, SIZE)].ship is not None and
                board.cells[Point.from_index(self.next_shipyard_position,
                                             SIZE)].ship.player_id == self.player_id))) and self.step_count <
                                                                        self.parameters[
                                                                            'shipyard_stop']) else self.config.spawn_cost

        self.move_ships(board)
        self.spawn_ships(board)
        return self.me.next_actions

    def debug(self):
        if len(self.me.ships) > 0:
            logging.debug("avg cargo at step " + str(self.step_count) + ": " + str(
                sum([ship.halite for ship in self.me.ships]) / len(self.me.ships)))
            if self.step_count % 25 == 0:
                map = np.zeros((SIZE ** 2,), dtype=np.int)
                for pos in range(SIZE ** 2):
                    if pos in self.guarding_positions:
                        map[pos] += 1
                    if pos in self.farming_positions:
                        map[pos] += 1
                    if pos in self.shipyard_positions:
                        map[pos] += 5
                # small = np.array(self.small_dominance_map.reshape((21, 21)), dtype=np.int)
                # medium = np.array(self.medium_dominance_map.reshape((21, 21)), dtype=np.int)
                # display_dominance_map(get_new_dominance_map([self.me] + self.opponents, 1.2, 15, 50).reshape((4, 21, 21)))

    def handle_special_steps(self, board: Board) -> bool:
        step = board.step
        if step == 0:
            # Immediately spawn a shipyard
            self.me.ships[0].next_action = ShipAction.CONVERT
            logging.debug("Ship " + str(self.me.ships[0].id) + " converts to a shipyard at the start of the game.")
            return True
        elif step == 1:
            # Immediately spawn a ship
            self.me.shipyards[0].next_action = ShipyardAction.SPAWN
            return True
        return False

    def plan_shipyard_position(self):
        if len(self.me.shipyards) == 0:
            return
        possible_positions = []
        if self.max_shipyard_connections == 0:
            shipyard = self.me.shipyards[0]
            shipyard_pos = TO_INDEX[shipyard.position]
            enemy_shipyard_positions = [TO_INDEX[enemy_shipyard.position] for player in self.opponents for
                                        enemy_shipyard in player.shipyards]
            early_second_shipyard = self.step_count <= self.parameters['early_second_shipyard']
            for pos in range(SIZE ** 2):
                if self.parameters['min_shipyard_distance'] <= get_distance(shipyard_pos, pos) <= self.parameters[
                    'max_shipyard_distance'] and min(
                    [20] + [get_distance(pos, enemy_pos) for enemy_pos in enemy_shipyard_positions]) >= self.parameters[
                    'min_enemy_shipyard_distance']:
                    if early_second_shipyard:
                        possible_positions.append((pos, self.ultra_blurred_halite_map[pos]))
                    else:
                        point = Point.from_index(pos, SIZE)
                        half = 0.5 * get_vector(shipyard.position, point)
                        half = Point(round(half.x), round(half.y))
                        midpoint = (shipyard.position + half) % SIZE
                        possible_positions.append((pos, self.get_populated_cells_in_radius_count(TO_INDEX[midpoint])))
        else:
            require_dominance = self.nb_connected_shipyards > 2 and (self.map_presence_rank != 0 or self.rank != 0)
            avoid_positions = [TO_INDEX[enemy_shipyard.position] for player in self.opponents for
                               enemy_shipyard in player.shipyards if self.map_presence_diff[player.id] < 2]
            for pos in range(SIZE ** 2):
                if require_dominance and self.small_dominance_map[pos] < self.parameters[
                    'shipyard_min_dominance'] * 1.8:
                    continue
                if len(avoid_positions) > 0:
                    for ap in avoid_positions:
                        if get_distance(pos, ap) < self.parameters['min_enemy_shipyard_distance']:
                            continue
                shipyard_distance = self.shipyard_distances[pos]
                if shipyard_distance < self.parameters['min_shipyard_distance'] or self.parameters[
                    'max_shipyard_distance'] < shipyard_distance:
                    continue
                point = Point.from_index(pos, SIZE)
                good_distance = [shipyard.position for shipyard in self.me.shipyards if
                                 self.parameters['min_shipyard_distance'] <= get_distance(pos, TO_INDEX[
                                     shipyard.position]) <= self.parameters['max_shipyard_distance']]
                if len(good_distance) >= 2:
                    for i in range(len(good_distance)):
                        for j in range(i + 1, len(good_distance)):
                            pos1, pos2 = good_distance[i], good_distance[j]
                            if (pos1.x == pos2.x == point.x) or (
                                    pos1.y == pos2.y == point.y):  # rays don't intersect
                                possible_positions.append((pos, self.get_populated_cells_in_radius_count(pos)))
                            else:
                                midpoint = TO_INDEX[get_excircle_midpoint(pos1, pos2, point)]
                                possible_positions.append((pos, self.get_populated_cells_in_radius_count(midpoint)))
        if len(possible_positions) > 0:
            possible_positions.sort(key=lambda data: data[1], reverse=True)
            self.next_shipyard_position = possible_positions[0][0]
            logging.info(
                "Planning to place the next shipyard at " + str(Point.from_index(self.next_shipyard_position, SIZE)))

    def build_shipyards(self, board: Board):
        avoid_positions = [TO_INDEX[enemy_shipyard.position] for player in self.opponents for
                           enemy_shipyard in player.shipyards if
                           self.map_presence_diff[player.id] < (2 if self.step_count > 130 else 1)]
        if self.next_shipyard_position is not None and not (
                self.parameters['min_shipyard_distance'] <= self.shipyard_distances[self.next_shipyard_position] <=
                self.parameters['max_shipyard_distance'] and (
                        len(avoid_positions) == 0 or self.next_shipyard_position not in avoid_positions)):
            self.plan_shipyard_position()
        converting_disabled = self.parameters['shipyard_start'] > self.step_count or self.step_count > self.parameters[
            'shipyard_stop']
        if self.step_count < self.parameters['shipyard_stop'] and ((self.parameters[
                                                                        'third_shipyard_step'] <= self.step_count < 200 and self.max_shipyard_connections <= 1 and self.ship_advantage > -10 and self.ship_count >= \
                                                                    self.parameters['third_shipyard_min_ships']) or (
                                                                           self.parameters[
                                                                               'second_shipyard_step'] <= self.step_count and self.max_shipyard_connections == 0 and self.ship_advantage > -18 and self.ship_count >=
                                                                           self.parameters['second_shipyard_min_ships']) \
                                                                   or ((max(self.nb_connected_shipyards, 1) + 1) / len(
                    self.me.ships) <= self.parameters[
                                                                           'ships_shipyards_threshold'] and self.ship_advantage >
                                                                       self.parameters[
                                                                           'shipyard_min_ship_advantage'] and self.max_shipyard_connections > 1)):
            if self.next_shipyard_position is None:
                self.plan_shipyard_position()
            elif self.small_dominance_map[self.next_shipyard_position] >= -3:
                ships = [ship for ship in self.me.ships if
                         ship.halite <= self.hunting_halite_threshold and ship.id not in self.ship_types.keys()]
                ships.sort(key=lambda ship: get_distance(TO_INDEX[ship.position], self.next_shipyard_position))
                cell = board.cells[Point.from_index(self.next_shipyard_position, SIZE)]
                if len(ships) > 0 and (cell.ship is None or cell.ship.player_id != self.player_id):
                    self.ship_types[ships[0].id] = ShipType.CONSTRUCTING
                if len(ships) > 1 and self.halite > 300 and self.small_dominance_map[self.next_shipyard_position] < 0.5:
                    self.ship_types[ships[1].id] = ShipType.CONSTRUCTION_GUARDING
            else:
                logging.debug("Dominance of " + str(self.small_dominance_map[
                                                        self.next_shipyard_position]) + " at the shipyard construction position is too low.")
        elif converting_disabled:
            return
        for ship in sorted(self.me.ships, key=lambda s: s.halite, reverse=True):
            if self.should_convert(ship) and (
                    ship.id not in self.ship_types.keys() or self.ship_types[ship.id] != ShipType.CONVERTING) and (
                    not converting_disabled or (self.next_shipyard_position is not None and TO_INDEX[
                ship.position] == self.next_shipyard_position)):
                self.convert_to_shipyard(ship)
                return  # only build one shipyard per step

    def spawn_ships(self, board: Board):
        # Spawn a ship if there are none left
        if len(self.me.ships) == 0 and self.halite >= self.config.spawn_cost:
            if len(self.me.shipyards) > 0:
                self.spawn_ship(self.me.shipyards[0])
        shipyards = self.me.shipyards
        shipyards.sort(key=lambda shipyard: self.calculate_spawning_score(TO_INDEX[shipyard.position]), reverse=True)
        for shipyard in shipyards:
            if self.halite < self.spawn_cost:  # save halite for the next shipyard
                return
            if shipyard.position in self.planned_moves:
                continue
            dominance = self.medium_dominance_map[TO_INDEX[shipyard.position]]
            if self.reached_spawn_limit(board):
                continue
            if self.ship_count >= self.parameters['min_ships'] and self.average_halite_per_cell / self.ship_count < \
                    self.parameters['ship_spawn_threshold']:
                continue
            if dominance < self.parameters['spawn_min_dominance'] and board.step > 75 and self.shipyard_count > 1:
                continue
            self.spawn_ship(shipyard)

    def reached_spawn_limit(self, board: Board):
        return board.step > self.parameters['spawn_till'] or ((self.ship_count >= max(
            [len(player.ships) for player in board.players.values() if player.id != self.player_id]) +
                                                               self.parameters[
                                                                   'max_ship_advantage']) and self.ship_count >=
                                                              self.parameters['min_ships'])

    def move_ships(self, board: Board):
        if len(self.me.ships) == 0:
            return
        self.returning_ships.clear()
        self.mining_ships.clear()
        self.hunting_ships.clear()
        self.guarding_ships.clear()

        if self.shipyard_count == 0:
            ship = max(self.me.ships, key=lambda ship: ship.halite)  # TODO: choose the ship with the safest position
            if ship.halite + self.halite >= self.config.convert_cost:
                self.convert_to_shipyard(ship)
                self.ship_types[ship.id] = ShipType.CONVERTING

        ships = self.me.ships.copy()

        for ship in ships:
            if ship.cell.shipyard is not None and 30 < self.step_count < self.parameters[
                'farming_end'] and ship.id not in self.ship_types.keys():
                self.guarding_ships.append(ship)
                self.ship_types[ship.id] = ShipType.GUARDING

        for ship in ships:
            ship_type = self.get_ship_type(ship, board)
            self.ship_types[ship.id] = ship_type
            if ship_type == ShipType.MINING:
                self.mining_ships.append(ship)
            elif ship_type == ShipType.RETURNING or ship_type == ShipType.ENDING:
                self.returning_ships.append(ship)

        self.assign_ship_targets(board)  # also converts some ships to hunting/returning ships

        logging.info(
            "*** Ship type breakdown for step " + str(self.step_count) + " (" + str(
                self.me.halite) + " halite) (ship advantage: " + str(self.ship_advantage) + ") ***")
        ship_types_values = list(self.ship_types.values())
        for ship_type in set(ship_types_values):
            type_count = ship_types_values.count(ship_type)
            logging.info(str(ship_type).replace("ShipType.", "") + ": " + str(type_count) + " (" + str(
                round(type_count / len(self.me.ships) * 100, 1)) + "%)")
        while len(ships) > 0:
            ship = ships[0]
            ship_type = self.ship_types[ship.id]
            if ship_type == ShipType.MINING:
                self.handle_mining_ship(ship)
            elif ship_type == ShipType.RETURNING or ship_type == ShipType.ENDING:
                self.handle_returning_ship(ship, board)
            elif ship_type == ShipType.HUNTING or ship_type == ShipType.DEFENDING:
                self.handle_hunting_ship(ship)
            elif ship_type == ShipType.GUARDING:
                self.handle_guarding_ship(ship)
            elif ship_type == ShipType.CONSTRUCTING:
                self.handle_constructing_ship(ship)
            elif ship_type == ShipType.CONSTRUCTION_GUARDING:
                self.handle_construction_guarding_ship(ship)

            ships.remove(ship)

        row, col = scipy.optimize.linear_sum_assignment(self.ship_position_preferences, maximize=True)
        for ship_index, position_index in zip(row, col):
            ship = self.me.ships[ship_index]
            if position_index >= len(self.positions_in_reach):
                # The ship wants to convert to a shipyard
                if self.ship_position_preferences[ship_index, position_index] > 5000:
                    # immediately convert
                    ship.next_action = ShipAction.CONVERT  # self.halite has already been reduced
                else:
                    ship.next_action = ShipAction.CONVERT
                    self.halite -= self.config.convert_cost
                    self.planned_shipyards.append(ship.position)
            else:
                target = self.positions_in_reach[position_index]
                if target != ship.position:
                    ship.next_action = get_direction_to_neighbour(TO_INDEX[ship.position], TO_INDEX[target])
                self.planned_moves.append(target)

    def assign_ship_targets(self, board: Board):
        # Mining assignment adapted from https://www.kaggle.com/solverworld/optimus-mine-agent
        if self.ship_count == 0:
            return

        ship_targets = {}
        mining_positions = []
        dropoff_positions = set()

        id_to_ship = {ship.id: ship for ship in self.mining_ships}

        halite_map = self.observation['halite']
        for position, halite in enumerate(halite_map):
            if halite >= self.parameters['min_mining_halite']:
                mining_positions.append(position)

        for shipyard in self.me.shipyards:
            shipyard_pos = TO_INDEX[shipyard.position]
            # Maybe only return to safe shiypards
            # Add each shipyard once for each distance to a ship
            for ship in self.mining_ships:
                dropoff_positions.add(shipyard_pos + get_distance(TO_INDEX[ship.position], shipyard_pos) * 1000)
        dropoff_positions = list(dropoff_positions)

        self.mining_score_beta = self.parameters['mining_score_beta'] if self.step_count >= self.parameters[
            'mining_score_start_returning'] else self.parameters[
            'mining_score_juicy']  # Don't return too often early in the game

        if self.parameters['farming_end'] < self.step_count < self.parameters['end_start']:
            self.mining_score_beta = self.parameters['mining_score_juicy_end']

        mining_scores = np.zeros((len(self.mining_ships), len(mining_positions) + len(dropoff_positions)))
        for ship_index, ship in enumerate(self.mining_ships):
            ship_pos = TO_INDEX[ship.position]
            for position_index, position in enumerate(mining_positions + dropoff_positions):
                if position >= 1000:
                    distance_to_shipyard = position // 1000
                    position = position % 1000
                    if distance_to_shipyard != get_distance(ship_pos, position):
                        mining_scores[ship_index, position_index] = -999999
                        continue

                mining_scores[ship_index, position_index] = self.calculate_mining_score(
                    ship_pos, position, halite_map[position],
                    self.blurred_halite_map[position], ship.halite)

        row, col = scipy.optimize.linear_sum_assignment(mining_scores, maximize=True)
        target_positions = mining_positions + dropoff_positions

        assigned_scores = [mining_scores[r][c] for r, c in zip(row, col)]
        assigned_scores.sort()
        hunting_proportion = self.parameters['hunting_proportion'] if self.step_count < self.parameters[
            'farming_end'] else self.parameters['hunting_proportion_after_farming']
        logging.debug("assigned mining scores mean: {}".format(np.mean(assigned_scores)))
        hunting_enabled = board.step > self.parameters['disable_hunting_till'] and (self.ship_count >= self.parameters[
            'hunting_min_ships'] or board.step > self.parameters['spawn_till'])
        hunting_threshold = max(np.mean(assigned_scores) - np.std(assigned_scores) * self.parameters[
            'hunting_score_alpha'], assigned_scores[
                                    ceil(len(assigned_scores) * hunting_proportion) - 1]) if len(
            assigned_scores) > 0 else -1

        for r, c in zip(row, col):
            if (mining_scores[r][c] < self.parameters['hunting_threshold'] or (
                    mining_scores[r][c] <= hunting_threshold and self.mining_ships[
                r].halite <= self.hunting_halite_threshold)) and hunting_enabled:
                continue
            if target_positions[c] >= 1000:
                ship_targets[self.mining_ships[r].id] = target_positions[c] % 1000
            else:
                ship_targets[self.mining_ships[r].id] = target_positions[c]

        # Convert indexed positions to points
        for ship_id, target_pos in ship_targets.items():
            ship = id_to_ship[ship_id]
            if target_pos in self.shipyard_positions:
                self.returning_ships.append(ship)
                self.mining_ships.remove(ship)
                self.ship_types[ship_id] = ShipType.RETURNING
                self.deposit_targets[ship_id] = Point.from_index(target_pos, self.size)
                logging.debug("Ship " + str(ship.id) + " returns.")
                continue
            self.mining_targets[ship_id] = Point.from_index(target_pos, self.size)
            logging.debug(
                "Assigning target " + str(Point.from_index(target_pos, self.size)) + " to ship " + str(ship.id))

        for ship in self.mining_ships:
            if ship.id not in self.mining_targets.keys():
                if ship.halite <= self.hunting_halite_threshold:
                    self.hunting_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.HUNTING
                else:
                    self.returning_ships.append(ship)
                    self.ship_types[ship.id] = ShipType.RETURNING

        encoded_dirs = [1, 2, 4, 8]
        possible_enemy_targets = [(dir, ship) for ship in self.enemies for dir in encoded_dirs for _ in
                                  range(self.parameters['max_hunting_ships_per_direction'])]
        hunting_scores = np.zeros(
            (len(self.hunting_ships), len(self.enemies) * 4 * self.parameters['max_hunting_ships_per_direction']))
        hunting_ship_to_idx = {ship.id: idx for idx, ship in enumerate(self.hunting_ships)}
        for ship_index, ship in enumerate(self.hunting_ships):
            ship_pos = TO_INDEX[ship.position]
            for enemy_index, (direction, enemy_ship) in enumerate(possible_enemy_targets):
                farthest_dirs = self.farthest_directions_indices[ship_pos][TO_INDEX[enemy_ship.position]]
                if farthest_dirs == direction or (farthest_dirs - direction) in encoded_dirs:
                    hunting_scores[ship_index, enemy_index] = self.calculate_hunting_score(ship, enemy_ship)
                else:
                    hunting_scores[ship_index, enemy_index] = -999999

        assigned_hunting_scores = []
        row, col = scipy.optimize.linear_sum_assignment(hunting_scores, maximize=True)
        for r, c in zip(row, col):
            self.hunting_targets[self.hunting_ships[r].id] = possible_enemy_targets[c][1]
            assigned_hunting_scores.append(hunting_scores[r, c])

        if len(self.me.shipyards) > 0:
            guarding_targets = [ship for ship in self.enemies if TO_INDEX[ship.position] in self.guarding_positions] + \
                               [shipyard for player in self.opponents for shipyard in player.shipyards if
                                TO_INDEX[shipyard.position] in self.guarding_positions]

            # Guarding ships
            assigned_hunting_scores.sort()
            guarding_threshold_index = max(
                min(ceil(((1 - clip(self.ship_advantage, 0, self.parameters['guarding_ship_advantage_norm']) /
                           self.parameters['guarding_ship_advantage_norm']) * (
                                  clip(self.enemy_hunting_proportion, 0, self.parameters['guarding_norm']) /
                                  self.parameters['guarding_norm'])) * len(assigned_hunting_scores)) - 1,
                    self.parameters['guarding_max_ships_per_shipyard'] * len(self.me.shipyards) - 1,
                    0 if self.step_count >= self.parameters['guarding_end'] else 500),
                min(len(assigned_hunting_scores) - 1, len(self.me.shipyards))) - len(self.guarding_ships)
            if guarding_threshold_index > 0:
                guarding_threshold = assigned_hunting_scores[guarding_threshold_index]
                for r, c in zip(row, col):
                    target_pos = TO_INDEX[possible_enemy_targets[c][1].position]
                    ship_pos = TO_INDEX[self.hunting_ships[r].position]
                    if hunting_scores[r, c] < guarding_threshold:
                        if target_pos not in self.guarding_positions or get_distance(
                                ship_pos, target_pos) > self.parameters[
                            'guarding_aggression_radius'] or ship_pos in self.shipyard_positions:
                            self.guarding_ships.append(self.hunting_ships[r])
                        else:
                            self.ship_types[self.hunting_ships[r].id] = ShipType.DEFENDING

                unassigned_defending_ships = [ship for ship in self.guarding_ships if
                                              ship.id not in self.shipyard_guards]
                assigned_defending_ships = []
                if len(guarding_targets) > 0 and len(unassigned_defending_ships) > 0:
                    defending_targets = np.full(shape=(len(unassigned_defending_ships),
                                                       len(guarding_targets) * self.parameters[
                                                           'max_guarding_ships_per_target']), fill_value=99999,
                                                dtype=np.int)
                    for ship_index, ship in enumerate(unassigned_defending_ships):
                        ship_pos = TO_INDEX[ship.position]
                        for target_index, target in enumerate(guarding_targets):
                            distance = get_distance(ship_pos, TO_INDEX[target.position])
                            if distance <= self.parameters['guarding_aggression_radius']:
                                defending_targets[ship_index, target_index * 2:target_index * 2 + 1] = distance
                    row, col = scipy.optimize.linear_sum_assignment(defending_targets, maximize=False)
                    for r, c in zip(row, col):
                        if defending_targets[r, c] > self.parameters['guarding_aggression_radius']:
                            continue
                        ship = unassigned_defending_ships[r]
                        assigned_defending_ships.append(ship.id)
                        self.ship_types[ship.id] = ShipType.DEFENDING
                        self.hunting_targets[ship.id] = guarding_targets[c // self.parameters[
                            'max_guarding_ships_per_target']]  # hunt the target (ship is still in self.guarding_ships and self.hunting_ships)

                for ship in self.guarding_ships:
                    if ship.id in assigned_defending_ships:
                        continue
                    # move to a shipyard
                    if ship in self.hunting_ships:
                        self.hunting_ships.remove(ship)
                    self.ship_types[ship.id] = ShipType.GUARDING

            self.guarding_ships = [ship for ship in self.guarding_ships if ship not in self.hunting_ships]
            available_guarding_ships = [ship for ship in self.guarding_ships if ship.id not in self.shipyard_guards]
            if len(available_guarding_ships) > 0:
                shipyards_to_protect = [shipyard_position for shipyard_position in self.shipyard_positions if
                                        self.medium_dominance_map[shipyard_position] > self.parameters[
                                            'shipyard_abandon_dominance']]
                if len(shipyards_to_protect) == 0:
                    shipyards_to_protect.append(self.shipyard_positions[0])  # guard at least one shipyard
                guarding_ships_per_shipyard = ceil(len(available_guarding_ships) / len(shipyards_to_protect))
                guarding_scores = np.zeros(
                    (len(available_guarding_ships), len(shipyards_to_protect) * guarding_ships_per_shipyard))
                for ship_index, ship in enumerate(available_guarding_ships):
                    ship_pos = TO_INDEX[ship.position]
                    for shipyard_index, shipyard_position in enumerate(shipyards_to_protect):
                        guarding_scores[ship_index,
                        guarding_ships_per_shipyard * shipyard_index:guarding_ships_per_shipyard * shipyard_index + guarding_ships_per_shipyard] = get_distance(
                            ship_pos, shipyard_position) + self.small_dominance_map[shipyard_position]
                row, col = scipy.optimize.linear_sum_assignment(guarding_scores, maximize=False)
                for r, c in zip(row, col):
                    self.guarding_shipyards[available_guarding_ships[r].id] = shipyards_to_protect[
                        c // guarding_ships_per_shipyard]

        available_hunting_ships = [ship for ship in self.hunting_ships if self.ship_types[ship.id] == ShipType.HUNTING]
        if len(available_hunting_ships) > 0 and self.parameters['hunting_max_group_size'] > 1:
            hunting_groups = group_ships(available_hunting_ships, self.parameters['hunting_max_group_size'],
                                         self.parameters['hunting_max_group_distance'])
            hunting_group_scores = np.zeros((len(hunting_groups), len(self.enemies) * 2))
            step = 4 * self.parameters['max_hunting_ships_per_direction']
            for group_idx, group in enumerate(hunting_groups):
                combined_hunting_scores = np.zeros((len(self.enemies) * 2,))
                for ship in group:
                    idx = hunting_ship_to_idx[ship.id]
                    for i in range(len(combined_hunting_scores) // 2):
                        combined_hunting_scores[i * 2:(i + 1) * 2] += clip(
                            np.max(hunting_scores[idx, i * step:(i + 1) * step]), 0, 999999)
                combined_hunting_scores /= len(group)
                hunting_group_scores[group_idx] = combined_hunting_scores
            row, col = scipy.optimize.linear_sum_assignment(hunting_group_scores, maximize=True)
            for r, c in zip(row, col):
                for ship in hunting_groups[r]:
                    self.hunting_targets[ship.id] = possible_enemy_targets[step * c // 2][1]

    def get_ship_type(self, ship: Ship, board: Board) -> ShipType:
        if ship.id in self.ship_types.keys():
            return self.ship_types[ship.id]
        if board.step >= self.parameters['end_start']:
            if self.shipyard_distances[TO_INDEX[ship.position]] + board.step + self.parameters[
                'end_return_extra_moves'] >= 398 and ship.halite >= self.parameters['ending_halite_threshold']:
                return ShipType.ENDING
        if ship.halite >= self.parameters['return_halite']:
            return ShipType.RETURNING
        else:
            return ShipType.MINING

    def handle_returning_ship(self, ship: Ship, board: Board):
        if self.ship_types[ship.id] == ShipType.ENDING:
            destination = self.get_nearest_shipyard(ship.position)
            if destination is not None:
                destination = destination.position
        else:
            if ship.id in self.deposit_targets.keys():
                destination = self.deposit_targets[ship.id]
            else:
                destination = self.get_nearest_shipyard(ship.position)
                if destination is not None:
                    destination = destination.position
        if destination is None:
            if self.halite + ship.halite >= self.config.convert_cost:
                if self.shipyard_count == 0:
                    self.convert_to_shipyard(ship)
                    logging.debug("Returning ship " + str(
                        ship.id) + " has no shipyard and converts to one at position " + str(ship.position) + ".")
                    return
                else:
                    destination = board.cells[self.planned_shipyards[0]].position
            else:
                # TODO: mine
                logging.debug("Returning ship " + str(ship.id) + " has no shipyard to go to.")
                return

        ship_pos = TO_INDEX[ship.position]
        destination_pos = TO_INDEX[destination]
        if self.ship_types[ship.id] == ShipType.ENDING:
            if get_distance(ship_pos, destination_pos) == 1:
                self.change_position_score(ship, destination, 9999)  # probably unnecessary
                logging.debug("Ending ship " + str(ship.id) + " returns to a shipyard at position " + str(destination))
                return

        self.prefer_moves(ship, navigate(ship.position, destination, self.size),
                          self.farthest_directions[ship_pos][destination_pos],
                          self.parameters['move_preference_return'], destination=destination)

    def handle_mining_ship(self, ship: Ship):
        if ship.id not in self.mining_targets.keys():
            logging.error("Mining ship " + str(ship.id) + " has no valid mining target.")
            return
        target = self.mining_targets[ship.id]
        ship_pos = TO_INDEX[ship.position]
        target_pos = TO_INDEX[target]
        reduce_farming_penalty = target_pos not in self.farming_positions
        if target != ship.position:
            self.prefer_moves(ship, nav(ship_pos, target_pos), self.farthest_directions[ship_pos][target_pos],
                              self.parameters['move_preference_base'], reduce_farming_penalty=reduce_farming_penalty,
                              destination=target)
            if self.shipyard_distances[ship_pos] == 1:
                for neighbour in get_neighbours(ship.cell):
                    if neighbour.shipyard is not None and neighbour.shipyard.player_id == self.player_id:
                        if self.step_count <= 11 and self.halite >= self.config.spawn_cost:
                            # We really want to get our ships out
                            self.change_position_score(ship, neighbour.position,
                                                       5 * self.parameters['move_preference_stay_on_shipyard'])
                        else:
                            self.change_position_score(ship, neighbour.position,
                                                       self.parameters['move_preference_stay_on_shipyard'])

        else:
            self.change_position_score(ship, target, self.parameters['move_preference_mining'])
            self.prefer_moves(ship, [], [], self.parameters['move_preference_mining'],
                              reduce_farming_penalty=reduce_farming_penalty)

    def handle_hunting_ship(self, ship: Ship):
        ship_pos = TO_INDEX[ship.position]
        ship_position = ship.position
        penalize_farming = not (
                    self.ship_types[ship.id] == ShipType.DEFENDING and ship.id in self.hunting_targets.keys() and
                    TO_INDEX[self.hunting_targets[ship.id].position] in self.farming_positions)
        if self.step_count >= self.parameters['end_start'] and ship.halite == 0:
            enemy_shipyards = [shipyard for player in self.opponents for shipyard in player.shipyards if
                               self.step_count + get_distance(ship_pos, TO_INDEX[shipyard.position]) <= 398]
            if len(enemy_shipyards) > 0:
                enemy_shipyards.sort(key=lambda shipyard: (30 - self.halite_ranking[shipyard.player_id] * 10 if
                                                           self.halite_ranking[self.player_id] <= 1 else
                                                           self.halite_ranking[shipyard.player_id] * 10) - get_distance(
                    ship_pos, TO_INDEX[shipyard.position]), reverse=True)
                target = enemy_shipyards[0]
                self.prefer_moves(ship, navigate(ship_position, target.position, self.size),
                                  self.farthest_directions[ship_pos][TO_INDEX[target.position]],
                                  self.parameters['move_preference_hunting'] * 2, penalize_farming,
                                  destination=target.position)
        if len(self.enemies) > 0:
            if ship.id in self.hunting_targets.keys():
                target = self.hunting_targets[ship.id]
            else:
                target = max(self.enemies, key=lambda enemy: self.calculate_hunting_score(ship, enemy))
            if (isinstance(target, Shipyard) and ship.halite <= self.parameters[
                'max_halite_attack_shipyard']) or (isinstance(target, Ship) and target.halite > ship.halite):
                target_position = target.position
                self.prefer_moves(ship, navigate(ship_position, target_position, self.size),
                                  self.farthest_directions[ship_pos][TO_INDEX[target_position]],
                                  self.parameters['move_preference_hunting'], penalize_farming,
                                  destination=target.position)

    def handle_guarding_ship(self, ship: Ship):
        if ship.id not in self.guarding_shipyards.keys():
            logging.error("Guarding ship " + str(ship.id) + " has no shipyard to guard.")
            return
        ship_pos = TO_INDEX[ship.position]
        shipyard_position = self.guarding_shipyards[ship.id]
        current_distance = get_distance(ship_pos, shipyard_position)
        if ship.id in self.shipyard_guards:
            if ship_pos != shipyard_position:
                self.prefer_moves(ship, nav(ship_pos, shipyard_position),
                                  self.farthest_directions[ship_pos][shipyard_position],
                                  self.parameters['move_preference_guarding'] * 2, False,
                                  destination=Point.from_index(shipyard_position, SIZE))
            else:
                self.change_position_score(ship, ship.position,
                                           self.parameters['move_preference_guarding'] - self.parameters[
                                               'move_preference_stay_on_shipyard'])
        elif current_distance <= self.parameters['guarding_max_distance_to_shipyard']:
            # stay near the shipyard
            if ship.cell.shipyard is not None:
                if self.halite < self.spawn_cost or self.step_count > \
                        self.parameters['spawn_till']:
                    self.change_position_score(ship, ship.position,
                                               self.parameters['move_preference_guarding'] - self.parameters[
                                                   'move_preference_stay_on_shipyard'])
            elif ship.cell.halite > 0:
                self.change_position_score(ship, ship.position, self.parameters['move_preference_guarding_stay'])
            elif self.parameters['guarding_min_distance_to_shipyard'] <= current_distance:
                self.change_position_score(ship, ship.position, self.parameters['move_preference_guarding'])
            for position in get_neighbouring_positions(ship.position):
                if self.parameters['guarding_min_distance_to_shipyard'] <= get_distance(TO_INDEX[position],
                                                                                        shipyard_position) <= \
                        self.parameters['guarding_max_distance_to_shipyard'] and (
                        position not in self.real_farming_points):
                    self.change_position_score(ship, position, self.parameters['move_preference_guarding'])
        else:
            self.prefer_moves(ship, nav(ship_pos, shipyard_position),
                              self.farthest_directions[ship_pos][shipyard_position],
                              self.parameters['move_preference_guarding'],
                              destination=Point.from_index(shipyard_position, SIZE))

    def handle_constructing_ship(self, ship: Ship):
        if self.next_shipyard_position is None:
            logging.error("Constructing ship " + str(ship.id) + " has no construction target.")
            return
        shipyard_point = Point.from_index(self.next_shipyard_position, SIZE)
        logging.debug(
            "Constructing ship " + str(ship.id) + " at position " + str(ship.position) + " is on the way to " + str(
                shipyard_point) + ".")
        self.prefer_moves(ship, navigate(ship.position, shipyard_point, self.size),
                          self.farthest_directions[TO_INDEX[ship.position]][self.next_shipyard_position],
                          self.parameters['move_preference_constructing'], destination=shipyard_point)

    def handle_construction_guarding_ship(self, ship: Ship):
        if self.next_shipyard_position is None:
            if len(self.planned_shipyards) == 0:
                logging.error("Construction guarding ship  " + str(ship.id) + " has no construction target.")
                return
            else:
                shipyard_point = self.planned_shipyards[0]
        else:
            shipyard_point = Point.from_index(self.next_shipyard_position, SIZE)
        self.prefer_moves(ship, navigate(ship.position, shipyard_point, self.size),
                          self.farthest_directions[TO_INDEX[ship.position]][TO_INDEX[shipyard_point]],
                          self.parameters['move_preference_construction_guarding'], destination=shipyard_point)

    def guard_shipyards(self, board: Board):
        shipyard_guards = dict()
        for shipyard in self.me.shipyards:
            if shipyard.position in self.planned_moves:
                continue
            shipyard_position = TO_INDEX[shipyard.position]
            dominance = self.medium_dominance_map[shipyard_position]

            min_distance = 20
            for ship in [ship for ship in self.me.ships if ship.halite <= self.hunting_halite_threshold and (
                    ship.id not in self.ship_types.keys() or self.ship_types[ship.id] not in [ShipType.CONVERTING,
                                                                                              ShipType.CONSTRUCTING])]:
                distance = get_distance(shipyard_position, TO_INDEX[ship.position])
                if distance < min_distance:
                    min_distance = distance
                    shipyard_guards[shipyard_position] = ship
            enemy_distance = self.enemy_distances[shipyard_position]
            if dominance < self.parameters['shipyard_abandon_dominance']:
                logging.debug("Abandoning shipyard " + str(shipyard.id))
            elif enemy_distance - 1 <= min_distance and shipyard_position in shipyard_guards.keys():
                guard = shipyard_guards[shipyard_position]
                self.shipyard_guards.append(guard.id)
                self.guarding_ships.append(guard)
                self.guarding_shipyards[guard.id] = shipyard_position
                self.ship_types[guard.id] = ShipType.GUARDING
            elif enemy_distance - 2 <= min_distance and shipyard_position in shipyard_guards.keys():
                guard = shipyard_guards[shipyard_position]
                if guard.cell.halite > 0:
                    self.change_position_score(guard, guard.position, -500)  # don't mine

            enemies = set(filter(lambda cell: cell.ship is not None and cell.ship.player_id != self.player_id,
                                 get_neighbours(shipyard.cell)))
            max_halite = min([cell.ship.halite for cell in enemies]) if len(enemies) > 0 else 500

            if len(enemies) > 0 and min_distance != 1:
                # TODO: maybe don't move on the shipyard if the dominance score is too low
                if shipyard.cell.ship is not None:
                    self.ship_types[shipyard.cell.ship.id] = ShipType.SHIPYARD_GUARDING
                    if self.halite < self.spawn_cost or (
                            self.step_count > self.parameters['spawn_till'] and (
                            self.shipyard_count > 1 or self.step_count > 385)) or dominance < \
                            self.parameters[
                                'shipyard_guarding_min_dominance'] or random() > self.parameters[
                        'shipyard_guarding_attack_probability'] or self.step_count >= self.parameters['guarding_stop']:
                        if dominance > self.parameters['shipyard_abandon_dominance']:
                            self.change_position_score(shipyard.cell.ship, shipyard.cell.position, 10000)
                            logging.debug("Ship " + str(shipyard.cell.ship.id) + " stays at position " + str(
                                shipyard.position) + " to guard a shipyard.")
                    else:
                        self.spawn_ship(shipyard)
                        for enemy in enemies:
                            logging.debug("Attacking a ship near our shipyard")
                            self.change_position_score(shipyard.cell.ship, enemy.position,
                                                       500)  # equalize to crash into the ship even if that means we also lose our ship
                            self.attack_position(
                                enemy.position)  # Maybe also do this if we don't spawn a ship, but can move one to the shipyard
                else:
                    potential_guards = [neighbour.ship for neighbour in get_neighbours(shipyard.cell) if
                                        neighbour.ship is not None and neighbour.ship.player_id == self.player_id and neighbour.ship.halite <= max_halite]
                    if len(potential_guards) > 0 and (
                            self.reached_spawn_limit(board) or self.halite < self.spawn_cost):
                        guard = sorted(potential_guards, key=lambda ship: ship.halite)[0]
                        self.change_position_score(guard, shipyard.position, 8000)
                        self.ship_types[guard.id] = ShipType.SHIPYARD_GUARDING
                        logging.debug("Ship " + str(guard.id) + " moves to position " + str(
                            shipyard.position) + " to protect a shipyard.")
                    elif self.halite > self.spawn_cost and (
                            dominance >= self.parameters[
                        'shipyard_guarding_min_dominance'] or board.step <= 25 or self.shipyard_count == 1) and (
                            self.step_count <
                            self.parameters['guarding_stop'] or (
                                    self.shipyard_count == 1 and self.step_count < self.parameters['end_start'])):
                        logging.debug("Shipyard " + str(shipyard.id) + " spawns a ship to defend the position.")
                        self.spawn_ship(shipyard)
                    else:
                        logging.info("Shipyard " + str(shipyard.id) + " cannot be protected.")

    def determine_vulnerable_enemies(self):
        hunting_matrix = get_hunting_matrix(self.me.ships)
        self.vulnerable_ships = dict()
        for ship in self.enemies:
            ship_pos = TO_INDEX[ship.position]
            escape_positions = [int(pos) for pos in np.argwhere(hunting_matrix >= ship.halite) if
                                int(pos) in self.positions_in_reach_indices[ship_pos]]
            if len(escape_positions) == 0:
                self.vulnerable_ships[ship.id] = -2
            elif len(escape_positions) == 1:
                if escape_positions[0] == ship_pos:
                    self.vulnerable_ships[ship.id] = -1  # stay still
                else:
                    self.vulnerable_ships[ship.id] = get_direction_to_neighbour(ship_pos, escape_positions[0])
        logging.debug("Number of vulnerable ships: " + str(len(self.vulnerable_ships)))

    def should_convert(self, ship: Ship):
        if self.halite + ship.halite < self.config.convert_cost:
            return False
        ship_pos = TO_INDEX[ship.position]
        if ship_pos == self.next_shipyard_position and self.step_count <= self.parameters['shipyard_stop']:
            return True
        if self.shipyard_count == 0 and (self.step_count <= self.parameters[
            'end_start'] or ship.halite >= self.config.convert_cost or self.cargo >= 1200):
            return True  # TODO: choose best ship
        if self.nb_connected_shipyards >= self.parameters['max_shipyards']:
            return False
        if self.average_halite_per_cell / self.shipyard_count < self.parameters[
            'shipyard_conversion_threshold'] or (max(self.nb_connected_shipyards, 1) + 1) / self.ship_count >= \
                self.parameters[
                    'ships_shipyards_threshold']:
            return False
        if self.medium_dominance_map[ship_pos] < self.parameters['shipyard_min_dominance']:
            return False
        return self.creates_good_triangle(ship.position)

    def creates_good_triangle(self, point):
        ship_pos = TO_INDEX[point]
        distance_to_nearest_shipyard = self.shipyard_distances[ship_pos]
        if self.parameters['min_shipyard_distance'] <= distance_to_nearest_shipyard <= self.parameters[
            'max_shipyard_distance']:
            good_distance = []
            for shipyard_position in self.shipyard_positions:
                if self.parameters['min_shipyard_distance'] <= get_distance(ship_pos, shipyard_position) <= \
                        self.parameters['max_shipyard_distance']:
                    good_distance.append(Point.from_index(shipyard_position, SIZE))
            if len(good_distance) == 0:
                return False
            midpoints = []
            if self.max_shipyard_connections == 0:
                half = 0.5 * get_vector(point, good_distance[0])
                half = Point(round(half.x), round(half.y))
                midpoints.append((point + half) % SIZE)
            else:
                for i in range(len(good_distance)):
                    for j in range(i + 1, len(good_distance)):
                        pos1, pos2 = good_distance[i], good_distance[j]
                        if (pos1.x == pos2.x == point.x) or (
                                pos1.y == pos2.y == point.y):  # rays don't intersect
                            midpoints.append(point)
                        else:
                            midpoints.append(get_excircle_midpoint(pos1, pos2, point))
            threshold = self.parameters[
                            'shipyard_min_population'] * self.average_halite_population * self.nb_cells_in_farming_radius
            if any([self.get_populated_cells_in_radius_count(TO_INDEX[midpoint]) >= threshold for midpoint in
                    set(midpoints)]):
                return True
        return False

    def get_populated_cells_in_radius_count(self, position):
        return sum([1 if self.observation['halite'][cell] > 0 else 0 for cell in self.farming_radius_list[position]])

    def calculate_mining_score(self, ship_position: int, cell_position: int, halite, blurred_halite,
                               ship_halite) -> float:
        distance_from_ship = get_distance(ship_position, cell_position)
        distance_from_shipyard = self.shipyard_distances[cell_position]
        halite_val = (1 - self.parameters['map_blur_gamma'] ** distance_from_ship) * blurred_halite + self.parameters[
            'map_blur_gamma'] ** distance_from_ship * halite
        if cell_position in self.enemy_positions and distance_from_ship > 1:
            halite_val *= 0.75 ** (distance_from_ship - 1)
        else:
            halite_val = min(1.02 ** distance_from_ship * halite_val, 500)
        farming_activated = self.parameters['farming_start'] <= (self.step_count + distance_from_ship) < \
                            self.parameters['farming_end']
        if distance_from_shipyard > 20:
            # There is no shipyard.
            distance_from_shipyard = 20
        if ship_halite == 0:
            ch = 0
        elif halite_val == 0:
            ch = 14
        else:
            ch = int(math.log(self.mining_score_beta * ship_halite / halite_val) * 2.5 + 5.5)
            ch = clip(ch, 0, 14)
        if distance_from_shipyard == 0:
            mining_steps = 0
            if distance_from_ship == 0:
                return 0  # We are on the shipyard
        elif cell_position in self.farming_positions and halite >= self.harvest_threshold and farming_activated:  # halite not halite_val because we cannot be sure the cell halite regenerates
            mining_steps = ceil(math.log(self.harvest_threshold / halite_val, 0.75))
        else:
            mining_steps = self.optimal_mining_steps[max(distance_from_shipyard - 1, 0)][
                max(int(round(self.parameters['mining_score_alpha'] * distance_from_shipyard) - 1), 0)][ch]
        if self.step_count >= self.parameters['end_start']:
            ending_steps = self.step_count + distance_from_ship + mining_steps + distance_from_shipyard + \
                           self.parameters['end_return_extra_moves'] // 2 - 398
            if ending_steps > 0:
                mining_steps = max(mining_steps - ending_steps, 0)
        dominance = self.parameters['mining_score_dominance_norm'] * clip(
            self.small_dominance_map[cell_position] + self.parameters['mining_score_dominance_clip'], 0,
            1.5 * self.parameters['mining_score_dominance_clip']) / (
                                1.5 * self.parameters['mining_score_dominance_clip'])
        if self.step_count < self.parameters['mining_score_start_returning']:
            dominance /= 1.5
            dominance += self.parameters['mining_score_dominance_norm'] / 3
        dominance += 1 - self.parameters['mining_score_dominance_norm'] / 2
        score = self.parameters['mining_score_gamma'] ** (distance_from_ship + mining_steps) * (
                self.mining_score_beta * ship_halite + (1 - 0.75 ** mining_steps) * halite_val) * dominance / max(
            distance_from_ship + mining_steps + self.parameters['mining_score_alpha'] * distance_from_shipyard, 1)
        if distance_from_shipyard == 0 and self.step_count <= 11:
            score *= 0.1  # We don't want to block the shipyard.
        if halite < self.harvest_threshold and cell_position in self.farming_positions and farming_activated:  # halite not halite_val because we cannot be sure the cell halite regenerates
            score *= self.parameters['mining_score_farming_penalty']
        return score

    def calculate_hunting_score(self, ship: Ship, enemy: Ship) -> float:
        d_halite = enemy.halite - ship.halite
        ship_pos = TO_INDEX[ship.position]
        enemy_pos = TO_INDEX[enemy.position]
        distance = get_distance(ship_pos, enemy_pos)
        if d_halite < 0:
            halite_score = -1
        elif d_halite == 0:
            halite_score = 0.25 * self.parameters['hunting_score_ship_bonus'] * (1 - self.step_count / 398) / \
                           self.parameters['hunting_score_halite_norm']
        else:
            ship_bonus = self.parameters['hunting_score_ship_bonus'] * (1 - self.step_count / 398)
            halite_score = (ship_bonus + d_halite) / self.parameters['hunting_score_halite_norm']
        player_score = 1 + self.parameters['hunting_score_kappa'] * (
            3 - self.player_ranking[ship.player_id] if self.rank <= 1 else self.player_ranking[ship.player_id])
        score = self.parameters['hunting_score_gamma'] ** distance * halite_score * (
                self.parameters['hunting_score_delta'] + self.parameters['hunting_score_beta'] * clip(
            self.medium_dominance_map[enemy_pos] + 15, 0, 30) / 30) * player_score * (
                        1 + (self.parameters['hunting_score_iota'] * clip(self.blurred_halite_map[enemy_pos], 0,
                                                                          500) / 500)) * (
                        1 + (self.parameters['hunting_score_zeta'] * clip(self.cargo_map[enemy_pos], 0,
                                                                          self.parameters['hunting_score_cargo_clip']) /
                             self.parameters['hunting_score_cargo_clip'])
                )
        if self.next_shipyard_position is not None and get_distance(enemy_pos, self.next_shipyard_position) <= 3:
            # Clear space for a shipyard
            score *= self.parameters['hunting_score_ypsilon']

        if enemy.id in self.vulnerable_ships.keys():
            safe_direction = self.vulnerable_ships[enemy.id]
            if distance <= 2:
                score *= self.parameters['hunting_score_hunt']
            elif safe_direction != -1 and safe_direction != -2:  # The ship can only stay at it's current position or it has no safe position.
                if safe_direction == ShipAction.WEST or safe_direction == ShipAction.EAST:
                    # chase along the x-axis
                    interception_pos = TO_INDEX[Point(ship.position.x, enemy.position.y)]
                else:
                    # chase along the y-axis
                    interception_pos = TO_INDEX[Point(enemy.position.x, ship.position.y)]
                target_dir = nav(enemy_pos, interception_pos)
                if (len(target_dir) > 0 and target_dir[0] == safe_direction) and get_distance(enemy_pos,
                                                                                              interception_pos) >= get_distance(
                    ship_pos, interception_pos):
                    # We can intercept the target
                    score *= self.parameters['hunting_score_intercept']
        if len(self.real_farming_points) > 0 and enemy_pos not in self.farming_positions:
            farming_positions_in_the_way = min(
                [self.get_farming_positions_count_in_between(ship.position, enemy.position, dir) for dir in
                 nav(ship_pos, enemy_pos)])
            score *= self.parameters['hunting_score_farming_position_penalty'] ** farming_positions_in_the_way
        return score

    def calculate_cell_score(self, ship: Ship, cell: Cell) -> float:
        score = 0
        if cell.position in self.planned_moves:
            score -= 1500
            return score
        if cell.shipyard is not None:
            shipyard = cell.shipyard
            owner_id = shipyard.player_id
            if shipyard.player_id != self.player_id:
                if shipyard.player.halite < self.config.spawn_cost and cell.ship is None and ship.halite < 30 and sum(
                        [1 for c in get_neighbours(cell) if c.ship is not None and
                                                            c.ship.player_id == owner_id and c.ship.halite <= ship.halite]) == 0:  # The shipyard cannot be protected by the owner
                    score += 300
                elif ship.halite > self.parameters['max_halite_attack_shipyard']:
                    score -= (400 + ship.halite)
                elif ship.halite == 0 and (
                        (self.rank == 0 and self.ship_advantage > 0) or self.step_count >= self.parameters[
                    'end_start'] or TO_INDEX[cell.position] in self.farming_positions) or self.shipyard_distances[
                    TO_INDEX[cell.position]] <= 2:
                    score += 400  # Attack the enemy shipyard
                else:
                    score -= 300
            elif self.halite >= self.spawn_cost and self.shipyard_count == 1 and not self.spawn_limit_reached:
                if self.step_count <= 100 or self.medium_dominance_map[TO_INDEX[shipyard.position]] >= self.parameters[
                    'spawn_min_dominance']:
                    score += self.parameters['move_preference_block_shipyard']
        if cell.shipyard is None:  # don't distract guarding ships
            if cell.ship is not None and cell.ship.player_id != self.player_id:
                if cell.ship.halite < ship.halite:
                    score -= (500 + ship.halite - 0.5 * cell.ship.halite)
                elif cell.ship.halite == ship.halite:
                    if TO_INDEX[cell.position] not in self.guarding_positions and (
                            self.next_shipyard_position is None or get_distance(TO_INDEX[cell.position],
                                                                                self.next_shipyard_position) > 2):
                        score -= 350
                else:
                    score += cell.ship.halite * self.parameters['cell_score_enemy_halite']
            neighbour_value = 0
            for neighbour in get_neighbours(cell):
                if neighbour.ship is not None and neighbour.ship.player_id != self.player_id:
                    if neighbour.ship.halite < ship.halite:  # We really don't want to go to that cell unless it's necessary.
                        neighbour_value = -(500 + ship.halite) * self.parameters['cell_score_neighbour_discount']
                        break
                    elif neighbour.ship.halite == ship.halite:
                        if TO_INDEX[neighbour.position] not in self.guarding_positions and (
                                self.next_shipyard_position is None or get_distance(TO_INDEX[neighbour.position],
                                                                                    self.next_shipyard_position) > 2):
                            neighbour_value -= 350 * self.parameters['cell_score_neighbour_discount']
                    else:
                        neighbour_value += neighbour.ship.halite * self.parameters['cell_score_enemy_halite'] * \
                                           self.parameters['cell_score_neighbour_discount']
            score += neighbour_value
            score += self.parameters['cell_score_dominance'] * self.small_dominance_map[TO_INDEX[cell.position]]
            if TO_INDEX[cell.position] in self.farming_positions and 0 < cell.halite < self.harvest_threshold:
                score += self.parameters['cell_score_farming']
        return score * (1 + self.parameters['cell_score_ship_halite'] * ship.halite)

    def calculate_player_score(self, player):
        return player.halite + len(player.ships) * 500 * (1 - self.step_count / 398) + len(player.shipyards) * 750 * (
                1 - self.step_count / 398) + sum(
            [ship.halite / 4 for ship in player.ships] if len(player.ships) > 0 else [0])

    def calculate_spawning_score(self, shipyard_position: int):
        if self.step_count <= self.parameters['farming_start']:
            return self.ultra_blurred_halite_map[shipyard_position]
        dominance = self.medium_dominance_map[shipyard_position]
        if dominance < self.parameters['shipyard_abandon_dominance']:
            return -999
        return -dominance

    def calculate_player_map_presence(self, player):
        return len(player.ships) + len(player.shipyards)

    def prefer_moves(self, ship, directions, longest_axis, weight, penalize_farming=True, reduce_farming_penalty=False,
                     destination=None):
        for dir in directions:
            position = (ship.position + dir.to_point()) % self.size
            w = weight
            if dir in longest_axis:
                w += self.parameters['move_preference_longest_axis']
            self.change_position_score(ship, position, weight)
        if destination is not None and len(directions) >= 2 and len(self.real_farming_points) > 0:
            axis1_farming_positions = self.get_farming_positions_count_in_between(ship.position, destination,
                                                                                  get_axis(directions[0]))
            axis2_farming_positions = self.get_farming_positions_count_in_between(ship.position, destination,
                                                                                  get_axis(directions[1]))
            if axis1_farming_positions > axis2_farming_positions:
                position = (ship.position + directions[0].to_point()) % self.size
                self.change_position_score(ship, position, int(-weight // 2))
            elif axis1_farming_positions < axis2_farming_positions:
                position = (ship.position + directions[1].to_point()) % self.size
                self.change_position_score(ship, position, int(-weight // 2))
        for dir in get_inefficient_directions(directions):
            position = (ship.position + dir.to_point()) % self.size
            self.change_position_score(ship, position, int(-weight // 1.2))
        if not penalize_farming:
            for cell in (get_neighbours(ship.cell) + [ship.cell]):
                if TO_INDEX[cell.position] in self.farming_positions and 0 < cell.halite < self.harvest_threshold:
                    self.change_position_score(ship, cell.position, -self.parameters['cell_score_farming'])
        elif reduce_farming_penalty:
            for cell in (get_neighbours(ship.cell) + [ship.cell]):
                if TO_INDEX[cell.position] in self.farming_positions and 0 < cell.halite < self.harvest_threshold:
                    self.change_position_score(ship, cell.position, int(-self.parameters[
                        'cell_score_farming'] // 1.5))

    def get_farming_positions_count_in_between(self, source, destination, axis):
        count = 0
        source_coordinate1 = source.x if axis == 'x' else source.y
        source_coordinate2 = source.y if axis == 'x' else source.x
        destination_coordinate = destination.y if axis == 'x' else destination.x
        distance = dist(source_coordinate2, destination_coordinate)
        for farming_position in self.real_farming_points:
            farming_coordinate1 = farming_position.x if axis == 'x' else farming_position.y
            if source_coordinate1 != farming_coordinate1:
                continue
            farming_coordinate2 = farming_position.y if axis == 'x' else farming_position.x
            if dist(source_coordinate2, farming_coordinate2) < distance and dist(destination_coordinate,
                                                                                 farming_coordinate2) < distance:
                count += 1
        return count

    def change_position_score(self, ship: Ship, position: Point, delta: float):
        self.ship_position_preferences[self.ship_to_index[ship], self.position_to_index[position]] += delta

    def get_nearest_shipyard(self, pos: Point):
        min_distance = float('inf')
        nearest_shipyard = None
        for shipyard in self.me.shipyards:
            distance = calculate_distance(pos, shipyard.position)
            if distance < min_distance:
                min_distance = distance
                nearest_shipyard = shipyard
        return nearest_shipyard

    def get_friendly_neighbour_count(self, cell: Cell):
        return sum(1 for _ in
                   filter(lambda n: n.ship is not None and n.ship.player_id == self.player_id, get_neighbours(cell)))

    def convert_to_shipyard(self, ship: Ship):
        assert self.halite + ship.halite >= self.config.convert_cost
        # ship.next_action = ShipAction.CONVERT
        self.ship_types[ship.id] = ShipType.CONVERTING
        self.ship_position_preferences[self.ship_to_index[ship],
        len(self.position_to_index):] = 9999999  # TODO: fix the amount of available shipyard conversions
        self.halite += ship.halite
        self.halite -= self.config.convert_cost
        self.ship_count -= 1
        self.shipyard_count += 1
        self.planned_shipyards.append(ship.position)
        if TO_INDEX[ship.position] == self.next_shipyard_position:
            self.next_shipyard_position = None

    def spawn_ship(self, shipyard: Shipyard):
        assert self.halite >= self.config.spawn_cost
        shipyard.next_action = ShipyardAction.SPAWN
        self.planned_moves.append(shipyard.position)
        self.halite -= self.config.spawn_cost
        self.ship_count += 1
        logging.debug("Spawning ship on position " + str(shipyard.position) + " (shipyard " + str(shipyard.id) + ")")

    def attack_position(self, position: Point):
        self.ship_position_preferences[:, self.position_to_index[position]][
            self.ship_position_preferences[:, self.position_to_index[position]] > -50] += 900

    def calculate_harvest_threshold(self):
        threshold = clip(0.001298 * self.step_count ** 2 + 1.3769 * self.step_count + 35, 80, 500)
        if self.map_presence_rank == 0 and self.ship_advantage >= 3:
            threshold += 15
        elif self.map_presence_rank == 3 and self.ship_advantage <= -7:
            threshold -= 10
        threshold *= (1 - (self.parameters['harvest_threshold_alpha'] / 2) + (
                self.parameters['harvest_threshold_alpha'] * (
                1 - clip(self.enemy_hunting_proportion, 0, self.parameters['harvest_threshold_hunting_norm']) /
                self.parameters['harvest_threshold_hunting_norm'])))
        return int(clip(threshold, 110, 450))


def agent(obs, config):
    global BOT
    if BOT is None:
        BOT = HaliteBot(PARAMETERS)
    board = Board(obs, config)
    logging.debug("Begin step " + str(board.step))
    return BOT.step(board, obs)
