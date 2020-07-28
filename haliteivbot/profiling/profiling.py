from kaggle_environments import make

if __name__ == '__main__':
    def run_game():
        agent_count = 4
        env = make("halite", configuration={"size": 21, "startingHalite": 5000}, debug=True)
        info = env.reset(agent_count)
        results = env.run(["../evolutionary/bots/uninstalllol3.py", "../evolutionary/bots/uninstalllol3.py",
                           "../evolutionary/bots/uninstalllol3.py", "../evolutionary/bots/uninstalllol3.py"])


    run_game()
