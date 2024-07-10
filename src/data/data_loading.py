import os.path
import pickle
from collections import defaultdict
from typing import NamedTuple

import blosc2
import numpy as np
import torch
from pandas import DataFrame
from torch import nan_to_num, Tensor
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from src.data.code_mappings import pitch_type_mapping, pitch_result_mapping, at_bat_event_mapping
from src.model.at_bat import AtBatState, PitchResult
from src.model.pitch import Pitch
from src.model.pitch_type import PitchType
from src.model.players import Batter, Pitcher
from src.model.zones import Zones, default


def load_blosc2(path: str):
    with open(path, 'rb') as f:
        return pickle.loads(blosc2.decompress(f.read()))


def save_blosc2(data, path: str):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'wb') as f:
        f.write(blosc2.compress(pickle.dumps(data)))


class BaseballData:
    """
    This class loads, processes, and stores the baseball data used for training and evaluation.
    More specific datasets are created for training specific models.

    As we add more data processing to this class, the time to load the data from scratch
    increases. Use load_with_cache to load the data once and cache it for future use.

    We do not use all the data available in this dataset, but the interesting column names
    are still kept.

    See https://www.kaggle.com/datasets/pschale/mlb-pitch-data-20152018/
    """

    default_processed_data_dir = '../../processed_data/'

    def __init__(self, load_pitches: bool = True, load_players: bool = True,
                 processed_data_dir: str = default_processed_data_dir):
        self.pitches = None
        self.players = None

        if load_pitches:
            self.pitches = []
            for year in range(2015, 2019):
                self.pitches.extend(load_blosc2(processed_data_dir + f'{year}.blosc2'))

        if load_players:
            players = load_blosc2(processed_data_dir + 'players.blosc2')
            self.pitchers: defaultdict[int, Pitcher] = players['pitchers']
            self.batters: defaultdict[int, Batter] = players['batters']

    @classmethod
    def process_data(cls, raw_data_dir: str = '../../raw_data/statcast/', processed_data_dir: str = default_processed_data_dir):
        """
        Load raw baseball data from the specified directory.
        """

        print('Generating baseball data (this will only happen once)...')

        # Keep track of players and statistics
        pitchers: defaultdict[int, Pitcher] = defaultdict(Pitcher)
        batters: defaultdict[int, Batter] = defaultdict(Batter)

        pitcher_all_at_bats = defaultdict(set)
        pitcher_hits_against = defaultdict(int)

        pitch_statistics_shape = (len(PitchType), Zones.DIMENSION, Zones.DIMENSION)
        pitcher_statistics = defaultdict(lambda: {
            'total_thrown': torch.zeros(pitch_statistics_shape),
            'total_velocity': torch.zeros(pitch_statistics_shape)
        })
        pitch_locations_30 = defaultdict(lambda: defaultdict(list))  # To calculate pitcher control

        batter_all_at_bats = defaultdict(set)
        batter_hits = defaultdict(int)

        batter_statistics = defaultdict(lambda: {
            'total_encountered': torch.zeros(pitch_statistics_shape),
            'total_swung': torch.zeros(pitch_statistics_shape),
            'total_hits': torch.zeros(pitch_statistics_shape)
        })

        velocities = []  # To normalize the velocity data

        # Load the pitches, year by year
        for year in tqdm(range(2015, 2019)):
            pitch_data: DataFrame = load_blosc2(f'{raw_data_dir}{year}.blosc2')
            pitch_data = pitch_data.replace({np.nan: None})

            pitches = []

            for row in pitch_data.itertuples(index=False):
                row: NamedTuple  # Consult https://baseballsavant.mlb.com/csv-docs for column names
                state = AtBatState(balls=row.balls, strikes=row.strikes, runs=row.bat_score,
                                   outs=row.outs_when_up, first=bool(row.on_1b),
                                   second=bool(row.on_2b), third=bool(row.on_3b))

                pitch_type = pitch_type_mapping.get(row.pitch_type, None)

                zone_idx = None
                plate_x, plate_z = None, None
                if row.sz_top is not None:
                    plate_x, plate_z = 12 * row.plate_x, 12 * row.plate_z  # Convert from feet to inches
                    zones = default if row.sz_bot < 0.3 else Zones(sz_bottom=12 * row.sz_bot, sz_top=12 * row.sz_top)
                    zone_idx, zone = zones.get_zone(plate_x, plate_z)

                pitch_outcome = pitch_result_mapping.get(row.description, None)
                if pitch_outcome == PitchResult.HIT_SINGLE:
                    pitch_outcome = at_bat_event_mapping.get(row.events, PitchResult.HIT_SINGLE)

                pitch = Pitch(at_bat_state=state, batter_id=row.batter, pitcher_id=row.pitcher,
                              pitch_type=pitch_type, location=zone_idx, pitch_result=pitch_outcome,
                              speed=row.release_speed, plate_x=plate_x, plate_z=plate_z,
                              game_id=row.game_pk, at_bat_num=row.at_bat_number,
                              pitch_num=row.pitch_number)

                # Update the pitch entry
                pitches.append(pitch)

                # Update player statistics
                if pitch.is_valid() and pitch.speed is not None:
                    def increment_statistic(statistic: Tensor, amount: float = 1):
                        for x, y in default.COMBINED_ZONES[pitch.zone_idx].coords:
                            statistic[pitch.type, x, y] += amount

                    # Pitcher
                    pitcher_all_at_bats[pitch.pitcher_id].add((pitch.game_id, pitch.at_bat_num))
                    if pitch.result.batter_hit():
                        pitcher_hits_against[pitch.pitcher_id] += 1

                    increment_statistic(pitcher_statistics[pitch.pitcher_id]['total_thrown'])
                    increment_statistic(pitcher_statistics[pitch.pitcher_id]['total_velocity'], pitch.speed)
                    velocities.append(pitch.speed)

                    if pitch.at_bat_state.balls == 3 and pitch.at_bat_state.strikes == 0:
                        pitch_locations_30[pitch.pitcher_id][pitch.type].append((pitch.plate_x, pitch.plate_z))

                    # Batter
                    batter_all_at_bats[pitch.batter_id].add((pitch.game_id, pitch.at_bat_num))
                    if pitch.result.batter_hit():
                        batter_hits[pitch.batter_id] += 1

                    increment_statistic(batter_statistics[pitch.batter_id]['total_encountered'])
                    if pitch.result.batter_swung():
                        increment_statistic(batter_statistics[pitch.batter_id]['total_swung'])
                    if pitch.result.batter_hit():
                        increment_statistic(batter_statistics[pitch.batter_id]['total_hits'])

            save_blosc2(pitches, processed_data_dir + f'{year}.blosc2')

        # Add the aggregate statistics to the players, replacing blank statistics with zeros
        velocity_mean = torch.mean(torch.tensor(velocities))
        velocity_std = torch.std(torch.tensor(velocities))

        # Aggregate pitcher statistics
        for pitcher_id, stats in pitcher_statistics.items():
            pitcher = pitchers[pitcher_id]
            pitcher.set_throwing_frequency_data(stats['total_thrown'])

            avg_velocity = stats['total_velocity'] / stats['total_thrown']
            normalized_velocity = nan_to_num((avg_velocity - velocity_mean) / velocity_std)

            pitcher.set_average_velocity_data(normalized_velocity)

        # Pitch control distributions
        jitter = torch.eye(2) * 1e-5  # Helps with positive definiteness
        for pitcher_id, pitch_counts in pitch_locations_30.items():
            for pitch_type, locations in pitch_counts.items():
                if len(locations) > 5:
                    locations_tensor = torch.tensor(locations, dtype=torch.float32)
                    mean = torch.mean(locations_tensor, dim=0)
                    covar = torch.cov(locations_tensor.T) + jitter
                    try:
                        pitchers[pitcher_id].estimated_control[pitch_type] = MultivariateNormal(mean, covar)
                    except ValueError:  # If the covariance matrix is not positive definite
                        pass

        # Aggregate batter statistics
        for batter_id, stats in batter_statistics.items():
            batter = batters[batter_id]
            batter.set_swinging_frequency_data(nan_to_num(stats['total_swung']))
            batter.set_batting_average_data(nan_to_num(stats['total_hits'] / stats['total_encountered']))

        # Add the OBP statistics to the players
        batter_obp_list = sorted([(batter_id, batter_hits[batter_id] / len(batter_all_at_bats[batter_id])) for batter_id in batters.keys()], key=lambda x: x[1])
        pitcher_obp_list = sorted([(pitcher_id, pitcher_hits_against[pitcher_id] / len(pitcher_all_at_bats[pitcher_id])) for pitcher_id in pitchers.keys()], key=lambda x: x[1])
        for idx, (player_id, obp) in enumerate(batter_obp_list):
            batters[player_id].obp = obp
            batters[player_id].obp_percentile = idx / len(batter_obp_list)
            batters[player_id].num_at_bats = len(batter_all_at_bats[player_id])
        for idx, (player_id, obp) in enumerate(pitcher_obp_list):
            pitchers[player_id].obp = obp
            pitchers[player_id].obp_percentile = idx / len(pitcher_obp_list)
            pitchers[player_id].num_batters_faced = len(pitcher_all_at_bats[player_id])

        players = {
            'pitchers': pitchers,
            'batters': batters
        }

        save_blosc2(players, processed_data_dir + 'players.blosc2')

        print('Done')


if __name__ == '__main__':
    BaseballData.process_data()
