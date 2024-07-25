# Zero-Sum Baseball

![A research poster of this repo's work](presentation/poster.png)

This project demonstrates how a zero-sum stochastic game model of baseball can be
used to approach more complex baseball problems. We explore some strategies for optimizing 
batter lineups and achieve some interesting results.

Read the write-up [here](presentation/writeup.pdf) for the full details.

### Getting Started
- Install the requirements with `pip install -r requirements.txt`
- Or manually install PyTorch, CVXPY, Pandas, Matplotlib, blosc2 and tqdm

### Project Structure
- `model_weights/` contains pre-trained models for the distributions
- `presentation/` contains the research poster and write-up
- `raw_data/` contains a script for fetching the Statcast data
- `src/` contains the made codebase
  - `src/data/` contains the data processing scripts and Pytorch datasets
  - `src/distributions/` contains the Pytorch models for learning the distributions
  - `src/model/` contains the object classes for the game model, like players, zones, pitches, etc.
  - `src/policy/` contains the zero-sum stochastic game model and work on batting lineup optimization
  - `src/statistics.ipynb` contains a bunch of different visualizations 