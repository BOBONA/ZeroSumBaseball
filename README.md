# Solving Baseball
This project applies game theory to baseball, computing mixed strategies for 
pitchers with Markov games.

## Model Overview
State is made up of the current count, the number of runs, the number of outs, and 
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

Of course, the value of a state is simply the number of runs.

## Limitations
- Our model is currently oblivious to plays that happen on field, besides the number of bases achieved
by the runner and whether the batter gets out. This is significant part of the game that we are ignoring. ↓↓

- The Kaggle dataset does have limited information on at-bat events, but it's not clear how 
to incorporate this into our model. We can learn a distribution of field results given a 
pitch result (making our entire transition function stochastic), however this should probably incorporate
the batter's intentions, which seems difficult.

- The pitch types simplify on the data given, where perhaps more of the numeric information could be of use. 

- In predicting swing results, we do not differentiate between the batter directly getting out and outs on the field.