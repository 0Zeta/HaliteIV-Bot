def sort_dictionary(dictionary):
    print("{")
    for c, key in enumerate(sorted(dictionary)):
        if c != len(dictionary) - 1:
            print("    '%s': %s" % (key, dictionary[key]) + ",")
        else:
            print("    '%s': %s" % (key, dictionary[key]))
    print("}")


d = {'spawn_till': 271, 'spawn_step_multiplier': 0, 'min_ships': 29, 'ship_spawn_threshold': 1.3403290844728477,
     'shipyard_conversion_threshold': 1.1805116725126426, 'ships_shipyards_threshold': 0.4962206074630996,
     'shipyard_stop': 278, 'min_shipyard_distance': 0, 'max_shipyard_distance': 11,
     'shipyard_min_dominance': 5.3120293563992265, 'shipyard_guarding_min_dominance': 3.976501897934859,
     'spawn_min_dominance': 3.5410149974032716, 'min_mining_halite': 38, 'convert_when_attacked_threshold': 304,
     'max_halite_attack_shipyard': 83, 'mining_score_alpha': 0.970426309367264, 'mining_score_beta': 0.8947538239676384,
     'mining_score_gamma': 0.9855172248244773, 'hunting_threshold': 2.694982160409251, 'hunting_halite_threshold': 2,
     'disable_hunting_till': 83, 'hunting_score_alpha': 0.7748086187292644, 'hunting_score_beta': 2.6375171424007227,
     'hunting_score_gamma': 0.8785218214543145, 'hunting_score_delta': 0.8604248471255606, 'return_halite': 464,
     'max_ship_advantage': 8, 'map_blur_sigma': 0.43709504234499663, 'map_blur_gamma': 0.4,
     'max_deposits_per_shipyard': 2, 'end_return_extra_moves': 7, 'ending_halite_threshold': 17, 'end_start': 380,
     'cell_score_enemy_halite': 0.35063481489408443, 'cell_score_neighbour_discount': 0.5884688910982244,
     'move_preference_base': 107, 'move_preference_return': 116, 'move_preference_mining': 131,
     'move_preference_hunting': 110, 'cell_score_ship_halite': 0.0005, 'conflict_map_alpha': 1.5679996502526856,
     'conflict_map_sigma': 0.7522651298172364, 'conflict_map_zeta': 0.7542051442915627,
     'dominance_map_small_sigma': 0.16697623378393822, 'dominance_map_medium_sigma': 0.41567319281444853,
     'dominance_map_small_radius': 3, 'dominance_map_medium_radius': 5}
sort_dictionary(d)
