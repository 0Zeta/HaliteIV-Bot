# simple rule-based bot writeup [?th place solution]
## Introduction
From July to September 2020, my teammate David Frank and I participated in an international AI programming challenge ["Halite by Two Sigma"](https://www.kagle.com/c/halite) which was hosted on Kaggle. I'll represent my team in this post to publish some information on our solution. First of all we want to thank Kaggle and Two Sigma for hosting this competition and providing excellent support which made the whole competition something special. Halite IV was the first programming competition we seriously participated in, so it was a genuinely new and exciting experience for us. The friendly atmosphere and the helpful community made this highly competitive challenge a truly fun way to spend our free time on something meaningful.

## Game rules
[INSERT GAMEPLAY]
The full game rules can be found on the [Kaggle webpage](https://www.kaggle.com/c/halite/overview/halite-rules), but we'll explain the most important features of the game in a simplified way.
- Halite is a four-player game taking place on a 21x21 grid map over 400 turns in which the players choose their actions simultaneously.
- Each player starts with 5000 halite (the central resource in the game) and one ship
- Ships can choose from six actions: they can move in four directions (north, east, south and west), stand still or convert into a shipyard for 500 halite
- A shipyard can spawn one ship per turn for a fee of 500 halite.
- Halite is placed symmetrically on many of the cells of the game map and when a ship stays still on a cell with halite it mines 25% of the cell's halite each turn receiving this amount as cargo.
- Each turn halite cells with no ships on them regenrate 2% of their halite value up to a maximum of 500 halite.
- Ships must return their cargo by moving on top of a shipyard. Then the cargo gets transferred to the bank of the player from which it can be spent.
- When multiple ships move to the same cell at the same turn, the ship with the least cargo survives and receives the cargo of the other ships which get destroyed. All ships are destroyed in case of a tie.
- Ships crashing into enemy shipyards destroy themselves and the shipyard if the shipyard doesn'thave a ship on top of it or spawns one.
- When a player has no ships and shipyards left (or only shipyards and not enough halite to spawn a ship) he is immediately eliminated from the game.
- At the end of the game the surviving players are ranked based on the halite in their bank at the last game step.

## Rule-based approach
### Parameters
First of all I have to admit that our approach utilizes no really fancy ML techniques or otherwise disproportionate complex algorithms. Instead our bot is a 100% rule based bot with lots of parameters. Initially we considered using deep reinforcement learning, but fiured out the problem was not optimal for this approach and with our poor hardware specs it would have been no good. Therefore we settled for a bot with many parameters that would then be tuned by a simple evolutionary algorithm which worked out quite well in the first half of the competition, but in the end we abandoned the evolutionary optimization because of problems with overfitting and symmetry. Therefore the behaviour of our final bot is dictated by more than hundred suboptimal parameters we chose more or less based on our intuition. These parameters control when we stop spawning ships, where to place our shipyards, how different factors influence mining and hunting, etc. (basically everything)

## Strategy
On a high level our bot aims to place shipyards in triangles around areas with a high population of halite cells (Not all cells contain halite. In fact most of them don't.) creating plantations. Then we wait for the halite to grow and mine on the cells when their halite values reach a certain harvest threshold. Shortly before the game ends we try to completely clear our whole plantation. We do our best to protect our plantation, scaring enemies away, but we also aim to let a few enemies mine inside our plantation and catch them afterwards causing ship losses among our opponents.

## 