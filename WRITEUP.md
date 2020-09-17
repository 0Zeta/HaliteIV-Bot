# Simple rule-based bot writeup for Halite IV [?th place solution]
## Introduction
From July to September 2020, my teammate David Frank and I participated in an international AI programming challenge ["Halite by Two Sigma"](https://www.kagle.com/c/halite) which was hosted on Kaggle. I'll represent my team in this post to publish some information on our solution. First of all we want to thank Kaggle and Two Sigma for hosting this competition and providing excellent support which made the whole competition something special. Halite IV was the first programming competition we seriously participated in, so it was a genuinely new and exciting experience for us. We also want to thank the whole community for the friendly and helpful atmosphere, which made this highly competitive challenge a truly fun way to spend our free time on something meaningful.

## Game rules
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

![an example game](https://user-images.githubusercontent.com/9535190/93367888-6a6e3880-f84d-11ea-85fa-527fd3afb62b.gif)


## Our rule-based approach
### Parameters
First of all I have to admit that our approach utilizes no really fancy ML techniques or otherwise disproportionate complex algorithms. Instead our bot is a 100% rule based bot with lots of parameters. Initially we considered using deep reinforcement learning, but fiured out the problem was not optimal for this approach and with our poor hardware specs it would have been no good. Therefore we settled for a bot with many parameters that would then be tuned by a simple evolutionary algorithm which worked out quite well in the first half of the competition, but in the end we abandoned the evolutionary optimization because of problems with overfitting and symmetry. Therefore the behaviour of our final bot is dictated by more than hundred suboptimal parameters we chose more or less based on our intuition. These parameters control when we stop spawning ships, where to place our shipyards, how different factors influence mining and hunting, etc. (basically everything)

### General strategy
On a high level our bot aims to place shipyards in triangles around areas with a high population of halite cells (Not all cells contain halite. In fact most of them don't.) creating halite plantations. Then we wait for the halite to grow and mine on the cells when their halite values reach a certain harvest threshold. Shortly before the game ends we try to completely clear our whole plantation. We do our best to protect our plantation, scaring enemies away, but we also aim to let a few enemies mine inside our plantation and catch them afterwards causing ship losses among our opponents.
At this point I want to thank [Fei Wang](https://www.kaggle.com/flynnfei) whose early success with his farming strategy, which also employs shipyards placed in triangles, made us recognize the impact of halite regeneration and the viability of this approach to the game.

### Features
In the following I will give an overview of the most important features of our bot while trying to maintain a good balance between oversimplificating interesting stuff and going too much into detail when it comes to the boring specifities of the implementation. In general we decided against programming a high-level framework for our bot because we felt like there were so many possible situations requiring special handling that we could accomplish our tasks best without much abstraction although this approach was very time-consuming, produced messy code and caused many bugs.

#### Stateless vs stateful agent
Until very late in the competition our agent was completely stateless, completely ignoring past turns at each step, and our final bot also keeps only very few information between turns, e.g. positions of planned shipyards, past intruders in our region, the number of our shipyards. This allowed us to minimize the impact of bad decisions in the past on newer ones.

#### Ship types
At the start of each turn our bot assigns each ship a ship type based on different metrics. Our bot uses 10 ship types in total, but these are the most important:
* MINING ships travel to halite-rich cells and mine halite.
* RETURNING ships are ships with high cargo values that travel to a shipyard to deposit their cargo.
* HUNTING ships try to crash into enemy ships, stealing their cargo. Most of them have a cargo of 0 halite.
* GUARDING ships patrol along the borders of our region with the most enemies and protect shipyards. They also have 0 halite and when an enemy approaches our territory they become
* DEFENDING ships that are basically HUNTING ships attacking enemies in regions dominated by us.

#### Scoring functions
Our bot using different scoring functions, which get computed for each ship, in combination with linear sum assignments for basically everything. Mining ships get assigned the mining target that maximizes the sum of all mining scores, hunting ships choose the target with the highest hunting score, guarding ships patrol on the border with the highest guarding score, etc. This approach proved to be simple and quite effective.

#### Move preferences
At the end of each turn we use a linear sum assignment to assign each ship an action simultaneously without any order. Specifically for every ship we calculate a move preference score for each of the six available actions. We chose this approach because it allows us to flexibly handle special cases, give different ship types different priorities and easily avoid collisions with enemies (well, at least in theory). These move preference scores are affected by many factors other than the destination of a ship, e.g. the number of safe cells around a position, halite, farming positions on this axis, etc. By handling all ships simultaneously we could improve our bot significantly compared to our old approach, which computed a move order for our ships.

#### Mining
We built upon the formula in [this](https://www.kaggle.com/solverworld/optimal-mining-with-carried-halite) excellent notebook by [SolverWorld](https://www.kaggle.com/solverworld), but we tuned the approach a bit and added some factors to the calculation. For example we delay the returning of our mining ships when there is still a lot of halite around them to improve mining. We also added many discount factors to express the uncertainty that comes with predicting future states and changed the mining scores of cells according to their respective safety, which describes how many of our ships are near the cell compared to our opponents taking into account how dangerous the ships are (Ships with less halite exert greater dominance). This safety matrix treats all opponents as a single one making our mining ships avoid regions of conflict and zones that are dominated by our opponents.

#### Hunting
Ships with bad mining scores and ships that are very likely to catch an enemy ship become hunting ships that try to destroy the ships of our opponents. The maximum amount of hunting ships targeting an enemy from one direction is limited which forced hunting ships to approach their targets from multiple sides increasing the probability of a catch. While our hunting score calculation is wuite complex the most important factors are distance, player score, danger to our ships and shipyards, halite near them and most important dominance. For each position on the map we calculate the dominance as player as their ship advantage (taking cargo values into account) over the other players in the region which is quite similar to the safety matrix with the difference that we treat each player separate.

![dominance_map](https://user-images.githubusercontent.com/9535190/93397072-19743980-f879-11ea-91e3-c4cd2e2010d2.png)

##### Chasing and intercepting targets
We boost the hunting scores of targets that can move safely (without risking to get destroyed or getting into situations that lead to them being surrounded by us) to fewer than two cells (especially for nearby hunting ships) and if a target can move in only one direction safely we calculate possible interceptions. For this calculation we simply check whether one of our ships can intercept the target, which is forced to move in one direction, by reaching a cell in front of the target in time.

![interceptions](https://user-images.githubusercontent.com/9535190/93397985-fd719780-f87a-11ea-913d-433413c83e7d.png)

#### Guarding
Of course guarding our plantation is a very important part of our strategy. Therefore a certain proportion of our hunting ships, which is determined by many factors like our player score, is turned into guarding ships (those with the worst hunting scores).

##### Shipyard guarding
We always try to have two ships that have at least the same distance to our shipyard as the two nearest enemies for all of our shipyards to defend them. Apparently this asimple algorithm works quite well when somebody tries to crash into our shipyard with only one ship, but often we cannot defend our shipyards against multiple ships attacking in a short time window as the following guarding ships often have to wait on cells which makes them mine halite, thus they cannot defend the shipyard against ships with 0 halite.

##### Patrolling along borders
Our border guarding ships fulfill multiple purposes, the most important points being an increase of our dominance, which scares away opponents, and a good positioning of 0 halite-ships around our plantation, which enables promising attacks on enemies within our borders. Similarly to the other scoring functions we calculate a guarding score for every border position for every ship and maximize the combined scores. We improve the guarding scores of border positions near our ship, near shipyards and in regions with bad dominance. Therefore our guarding ships actively improve the dominance in these regions and limit the amount of enemies that get into our plantation.

#### Shipyard placement
Shipyards are easily one of the most important aspects of our strategy as they are crucial for efficient use of our plantation. Well-placed shipyards exert dominance, allow short return distances for mining ships and serve as an entry point into hostile territory. The most important factor for lucrative plantations is the halite population of an area, which is the amount of cells that contain halite, as only cells with more than 0 halite produce halite. Therefore our bot calculates the next shipyard location each time our ship count exceeds different thresholds. After planning a new shipyard we increase the hunting scores of enemy targets near the planned shipyard site, slowly securing the position (if we succeed). Then we send two ships, one guarding ship that makes sure the newly constructed shipyard isn't immediately destroyed and a constructing ship that converts into a shipyard once it reaches the position. The way we choose our next shipyard position greatly varies with our current shipyard count, our player score, etc.
- We place the first shipyard near clusters of halite-rich cells to get a good start in the game.
- The second shipyard is also placed in a halite-rich region, but we also take the halite population in a circle that covers the area between our two shipyards into account.
- The third shipyard is placed so that the included area maximizes the halite population, but we also try not to place it near enemy shipyards of players that aren't completely out of the game.
- After the third shipyard we try to maximize halite population and dominance values (using a really blurry dominance map).

#### Special regions
When we have more than one shipyard, we define different regions in the space between/around our shipyards:
* Farming zone: These positions are the cells we use for growing and harvesting halite.
* Minor farming zone: These are also farming positions, but they aren't as easy to protect as the real farming positions. Therefore we don't let them grow for as long as the others and we don't penalize ships for travelling over them.
* Guarding zone: The hunting scores of enemies on these positions get increased and our bot takes 1:1 trades when an enemy ships stays there for too long.
* Border positons: This is simply a border around our plantations along which our guarding ships patrol.
[INSERT REGIONS GRAPHIC]

#### Spawning
For ship spawning we almost always use the simple rule: always spawn until step x; spawn ships at the shipyards with the lowest dominance to reinforce their defenders

### Some strategic considerations
#### ship trades
Generally speaking taking 1:1 trades by letting a ship crash into an enemy ship with equal cargo isn't the best thing to do in a four player game because you weaken yourself and only one of the opponents. Despite this our bot often takes 1:1 trades for several reasons:
- In the early game we greedily ignore to which cells enemy ships with equal cargo could move, essentially hoping that our opponent isn't as mad as we are and avoids crashes.
- On the directly neighbouring cells of our shipyard we also take 1:1 trades to defend our shipyards.
- When we are far ahead in the game, our bot is also willing to take trades as the benefits of weakening our opponent outweigh the downsides of losing one of many ships.
- Ships that stay on our farming plantations for too long without mining are also targeted by our hunting ships to remove them from our farming positions.
Due to our depence on well-placed shipyards and the fact that we cannot simply move our stationary halite plantations as the exponential halite regeneration takes a lot of time to let halite grow, we are forced to defend our shipyards at all costs and destroy shipyards in our farming/guarding zone (unless we have many shipyards and can afford to lose one plantation), which is a huge downside of our strategy as we often drain our material so much that we and the opponent who attacked us/was attacked by us finish in the last two places.

#### Player scores and their implications
Every turn we calculate different sets of player scores, taking different metrics into account, to see who is most likely to win, who has the biggest navy, who has the greatest map presence, etc.
The resulting rankings are used for several decisions, for example we don't want to place a shipyard next to a player that has way more ships, but we want to punish greedy bots with a low ship count that start banking up halite too early by attacking them. Another important point is that we boost the hunting scores for players that are competing for the same place as we are. For example we attack the player with the second highest player when we are first and we try to at least finish in the third place by attacking the worst bot besides us when we are far behind.

### Tuning parameters
At first we used a simple (badly implemented) evolutionary algorithm to tune our parameters, but as our bot got better and better we couldn't use some quickly implemented bots with different strategies anymore and had to play only against our own bots. For a while we could still achieve quite okay-ish results by limiting the games to 110 turns, thus improving our early game, but it became clear that we couldn't overcome the problems with this approach, namely symmetry, overfitting and little noise leading to some bot snowballing and winning by chance.
In the end we manually tuned some parameters and wanted to wait until the last day of the competition to optimize them, but sadly I introduced a critical bug the day before, resulting in very poor performance of our final bots.