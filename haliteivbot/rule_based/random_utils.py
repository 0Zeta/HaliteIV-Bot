from haliteivbot.rule_based.utils import create_optimal_mining_steps_tensor


def sort_dictionary(dictionary):
    print("{")
    for c, key in enumerate(sorted(dictionary)):
        if c != len(dictionary) - 1:
            print("    '%s': %s" % (key, dictionary[key]) + ",")
        else:
            print("    '%s': %s" % (key, dictionary[key]))
    print("}")


def generate_optimal_mining_steps_tensor(parameters):
    print(create_optimal_mining_steps_tensor(parameters['mining_score_alpha'], parameters['mining_score_beta'],
                                             parameters['mining_score_gamma']))
