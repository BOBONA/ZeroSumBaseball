import csv
import os.path
import pickle
from collections import defaultdict

import blosc2
import torch
from torch import nan_to_num, Tensor
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from src.data.code_mappings_old import old_at_bat_event_mapping, old_pitch_type_mapping, old_pitch_result_mapping
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

    AT_BATS = 'atbats.csv'
    PITCHES = 'pitches.csv'

    def __init__(self, load_pitches: bool = True, load_players: bool = True,
                 processed_data_dir: str = default_processed_data_dir):
        self.pitches = None
        self.players = None

        if load_pitches:
            self.pitches = load_blosc2(processed_data_dir + 'all_pitches.blosc2')

        if load_players:
            players = load_blosc2(processed_data_dir + 'players.blosc2')
            self.pitchers: defaultdict[int, Pitcher] = players['pitchers']
            self.batters: defaultdict[int, Batter] = players['batters']

    @classmethod
    def process_data(cls, raw_data_dir: str = '../../raw_data/kaggle/', processed_data_dir: str = default_processed_data_dir):
        """
        Load raw baseball data from the specified directory.
        """

        print('Generating baseball data (this will only happen once)...')

        at_bat_data: dict[int, tuple[int, int, int, PitchResult]] = {}
        pitchers: defaultdict[int, Pitcher] = defaultdict(Pitcher)
        batters: defaultdict[int, Batter] = defaultdict(Batter)

        # Load at-bat data
        with open(raw_data_dir + cls.AT_BATS) as f:
            at_bats = csv.reader(f, delimiter=',')

            batter_obp = defaultdict(lambda: (0, 0))  # Tracks the number of plate appearances and hits for each batter
            pitcher_obp = defaultdict(lambda: (0, 0))  # Tracks the number of batters faced and hits allowed for each pitcher

            next(at_bats)
            for (at_bat_id, batter_id, event, game_id, inning, outs, pitcher_team_score,
                 pitcher_hand, pitcher_id, batter_side, top_inning) in tqdm(at_bats, desc='Loading at-bat data'):
                at_bat_id = int(at_bat_id)
                batter_id = int(batter_id)
                pitcher_id = int(pitcher_id)
                game_id = int(game_id)

                pitch_result = old_at_bat_event_mapping.get(event, None)
                at_bat_data[at_bat_id] = (game_id, pitcher_id, batter_id, pitch_result)

                # Update the OBP statistics
                add_hit = pitch_result is not None
                batter_obp[batter_id] = (batter_obp[batter_id][0] + 1, batter_obp[batter_id][1] + add_hit)
                pitcher_obp[pitcher_id] = (pitcher_obp[pitcher_id][0] + 1, pitcher_obp[pitcher_id][1] + add_hit)

            # Add the OBP statistics to the players
            batter_obp_list = sorted([(batter_id, hits / pa) for batter_id, (pa, hits) in batter_obp.items()], key=lambda x: x[1])
            pitcher_obp_list = sorted([(pitcher_id, hits / bf) for pitcher_id, (bf, hits) in pitcher_obp.items()], key=lambda x: x[1], reverse=True)
            for idx, (player_id, obp) in enumerate(batter_obp_list):
                batters[player_id].obp = obp
                batters[player_id].obp_percentile = idx / len(batter_obp_list)
                batters[player_id].num_at_bats = batter_obp[player_id][0]
            for idx, (player_id, obp) in enumerate(pitcher_obp_list):
                pitchers[player_id].obp = obp
                pitchers[player_id].obp_percentile = idx / len(pitcher_obp_list)
                pitchers[player_id].num_batters_faced = pitcher_obp[player_id][0]

        all_pitches: list[Pitch] = []

        # Load individual pitch data
        with (open(raw_data_dir + cls.PITCHES) as f):
            pitches = csv.reader(f, delimiter=',')

            # Keep track of player statistics
            pitch_statistics_shape = (len(PitchType), Zones.DIMENSION, Zones.DIMENSION)
            pitcher_statistics = defaultdict(lambda: {
                'total_thrown': torch.zeros(pitch_statistics_shape),
                'total_velocity': torch.zeros(pitch_statistics_shape)
            })
            pitch_locations_30 = defaultdict(lambda: defaultdict(list))  # To calculate pitcher control

            batter_statistics = defaultdict(lambda: {
                'total_encountered': torch.zeros(pitch_statistics_shape),
                'total_swung': torch.zeros(pitch_statistics_shape),
                'total_hits': torch.zeros(pitch_statistics_shape)
            })
            velocities = []  # To normalize the velocity data

            # For now, we ignore most of the columns in the CSV
            next(pitches)
            for (pos_x, pos_z, start_speed, end_speed, spin_rate, spin_dir,
                 _, _, _, _, _, _, sz_bottom, sz_top, _, _, _, _, _, _, _, _, _, _, _, _, _,
                 pitch_result_code, _, pitch_type, _, batter_score, at_bat_id,
                 balls, strikes, outs, pitch_number, player_on_1b, on_2b, on_3b) in tqdm(pitches, desc='Loading pitch data'):

                game_id, pitcher_id, batter_id, ab_pitch_result = at_bat_data.get(int(float(at_bat_id)))

                state = AtBatState(balls=int(float(balls)), strikes=int(float(strikes)), runs=int(float(batter_score)),
                                   outs=int(float(outs)), first=bool(int(float(player_on_1b))),
                                   second=bool(int(float(on_2b))), third=bool(int(float(on_3b))))

                loc = (None if not pos_x else 12 * float(pos_x),  # Convert from feet to inches
                       None if not pos_z else 12 * float(pos_z))

                if loc[0] is not None and loc[1] is not None:
                    zones_bottom = 0 if sz_bottom == '' else float(sz_bottom)
                    zones = default if zones_bottom < 0.3 else Zones(sz_bottom=12 * zones_bottom, sz_top=12 * float(sz_top))
                    zone_idx, zone = zones.get_zone(*loc)

                pitch_type = old_pitch_type_mapping.get(pitch_type, None)
                pitch_result = old_pitch_result_mapping.get(pitch_result_code, None)

                if pitch_result == PitchResult.HIT_SINGLE:
                    pitch_result = ab_pitch_result  # The more fine-grained event is stored in the at-bat

                pitch = Pitch(at_bat_state=state, batter_id=batter_id, pitcher_id=pitcher_id,
                              pitch_type=pitch_type, location=zone_idx, pitch_result=pitch_result,
                              speed=None if not start_speed else float(start_speed), plate_x=loc[0], plate_z=loc[1],
                              game_id=game_id, at_bat_num=at_bat_id,
                              pitch_num=int(float(pitch_number)))

                # Update the pitch entry
                all_pitches.append(pitch)

                # Update player statistics
                if pitch.is_valid():
                    def increment_statistic(statistic: Tensor, amount: float = 1):
                        for x, y in zone.coords:
                            statistic[pitch_type, x, y] += amount

                    # Pitcher
                    increment_statistic(pitcher_statistics[pitcher_id]['total_thrown'])
                    increment_statistic(pitcher_statistics[pitcher_id]['total_velocity'], float(start_speed))
                    velocities.append(float(start_speed))

                    if pitch.at_bat_state.balls == 3 and pitch.at_bat_state.strikes == 0:
                        pitch_locations_30[pitcher_id][pitch.type].append(loc)

                    # Batter
                    increment_statistic(batter_statistics[batter_id]['total_encountered'])
                    if pitch.result.batter_swung():
                        increment_statistic(batter_statistics[batter_id]['total_swung'])
                    if pitch.result.batter_hit():
                        increment_statistic(batter_statistics[batter_id]['total_hits'])

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

        players = {
            'pitchers': pitchers,
            'batters': batters
        }

        save_blosc2(all_pitches, processed_data_dir + 'all_pitches.blosc2')
        save_blosc2(players, processed_data_dir + 'players.blosc2')

        print('Done')


if __name__ == '__main__':
    BaseballData.process_data()
