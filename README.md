# Zero-Sum Baseball

![A research poster of this repo's work](presentation/poster-small.jpg)

This project demonstrates how a zero-sum stochastic game model of baseball can be
used to approach more complex baseball problems. We explore some strategies for optimizing 
batter lineups and achieve some interesting results.

Read the write-up [here](presentation/writeup.pdf) for the full details.

### Getting Started
1. Install the requirements with `pip install -r requirements.txt`
   - Or manually install PyTorch, CVXPY, Pandas, Matplotlib, blosc2, and tqdm
2. Fetch the raw data with `raw_data/fetch_data.py`
3. Process the data with `src/data/data_loading.py`
4. Try out the zero-sum stochastic game model with `src/policy/optimal_policy.py`
5. Try the batting lineup optimization scripts with `src/policy/batting_order_optimization.py`
6. Check out some visualizations with `src/statistics.ipynb` and `src/policy/batting_order.ipnyb`
7. Feel free to load the data with `bd = BaseballData()` and experiment!

### Project Structure
- `model_weights/` contains pre-trained models for the distributions
- `presentation/` contains the research poster and write-up
- `src/` contains the made codebase
  - `src/data/` contains the data processing scripts and Pytorch datasets
  - `src/distributions/` contains the Pytorch models for learning the distributions
  - `src/model/` contains the object classes for the game model, like players, zones, pitches, etc.
  - `src/policy/` contains the zero-sum stochastic game model and work on batting lineup optimization
