# Raw data
This project currently relies on [MLB pitch data from 2015-2018](https://www.kaggle.com/datasets/pschale/mlb-pitch-data-20152018), scraped from their site by Paul Schale.
Documentation is written on the site, and you can see how this project uses it in [data_loading.py](../src/data/data_loading.py).

Please include the uncompressed files in this directory and the BaseballData class should take care of the details. Note that it includes an option to cache itself.