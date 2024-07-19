# Solving Baseball
This project applies game theory to baseball, computing mixed strategies for 
pitchers with Markov games.

## Model Overview
State is made up of the current count, the number of outs, and 
the occupied bases. The pitcher's actions are made up of the pitch type and location, 
in a 5x5 grid. The batter's actions are simply to swing or take. 

The transition probability distribution takes a few factors into account, while also
heavily simplifying what actually happens in a game. We calculate a pitcher control distribution
to map an intended zone to a distribution of actual zones. We compute a batter patience
distribution and override the batter's decision if the pitch results in a "borderline" non-strike 
zone. If the batter swings, we use an inferred swing outcome distribution to determine the outcome.

The outcome, which is either a hit (single, double, triple, home-run), an out, a foul, or a ball,
is used to transition to a new state. Our transition function heavily simplifies the actual
workings of the game. We assume each player advances the number of bases corresponding to the hit.

Of course, the reward of an action over a state is simply the expected number of runs.

## Limitations
- Our model is currently oblivious to plays that happen on field, besides the number of bases achieved
by the runner and whether the batter gets out. This is a significant part of the game that we are ignoring. ↓↓

- The Kaggle dataset does have limited information on at-bat events, but it's not clear how 
to incorporate this into our model. We can learn a distribution of field results given a 
pitch result (making our entire transition function stochastic), however this should probably incorporate
the batter's intentions, which seems difficult.

- The pitch types simplify on the data given, where perhaps more of the numeric information could be of use. 

- In predicting swing results, we do not differentiate between the batter directly getting out and outs on the field.

- The pitcher's options are limited in a way that might not be representative of the actual game. The pitcher cannot
aim for a borderline zone and can only aim for 5x5 centers.

## TODO / Ideas
- Finish the poster. Clean up the code. Make a write-up.
  - Batter selection
  - Stochastic games

- Run against a cardinals lineup. Analyze the runs that we record between our strategies and see if there are any patterns, 
  why are some lineups improved more, what do the winning ones look like?

- Perhaps a batter selection strategy can be developed which directly uses the symmetry of the game, since running the game
  gives us values for each rotation of the batting order.

- Investigate ways to improve convergence on two-strike positions (where fouls act as self-loops). I got minor improvement
  by iterating an additional time, which is currently still in the code.

- Incorporate a distribution for on-field outcomes. We can start with the empirical distribution
