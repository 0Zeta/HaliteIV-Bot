def sort_dictionary(dictionary):
    print("{")
    for c, key in enumerate(sorted(dictionary)):
        if c != len(dictionary) - 1:
            print("    '%s': %s" % (key, dictionary[key]) + ",")
        else:
            print("    '%s': %s" % (key, dictionary[key]))
    print("}")
