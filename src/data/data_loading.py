import csv
import os.path
from typing import Self

import torch
from torch import nan_to_num, Tensor
from tqdm import tqdm

from src.model.at_bat import AtBat, AtBatState, AtBatOutcome, PitchResult
from src.model.game import Game
from src.model.pitch import Pitch
from src.model.pitch_type import PitchType
from src.model.players import Batter, Pitcher
from src.model.zones import get_zone, ZONES_DIMENSION, Zone

# Some definitions are necessary to map the raw data to our less precise format
on_base_events = ['Single', 'Double', 'Triple', 'Home Run', 'Walk', 'Intent Walk',
                  'Hit By Pitch', 'Field Error', 'Catcher Interference']

pitch_type_mapping = {
    'CH': PitchType.CHANGEUP,
    'CU': PitchType.CURVE,
    'EP': PitchType.CURVE,  # Eephus is a type of slow curveball
    'FC': PitchType.CUTTER,
    'FF': PitchType.FOUR_SEAM,
    'FO': None,  # Pitchout doesn't fit in the given categories
    'FS': PitchType.CHANGEUP,  # Splitter is often considered a type of changeup
    'FT': PitchType.TWO_SEAM,
    'IN': None,  # Intentional ball doesn't fit in the given categories
    'KC': PitchType.CURVE,
    'KN': PitchType.CURVE,  # Knuckleball is distinct but closer to a curve in action
    'PO': None,  # Pitchout doesn't fit in the given categories
    'SC': PitchType.CHANGEUP,  # Screwball acts like a changeup with reverse break
    'SI': PitchType.TWO_SEAM,
    'SL': PitchType.SLIDER,
    'UN': None,  # Unknown types cannot be classified
}

pitch_result_mapping = {
    'B': PitchResult.CALLED_BALL,  # Ball
    '*B': PitchResult.CALLED_BALL,  # Ball in dirt
    'I': PitchResult.CALLED_BALL,  # Intentional ball
    'P': PitchResult.CALLED_BALL,  # Pitchout
    'C': PitchResult.CALLED_STRIKE,  # Called strike
    'S': PitchResult.SWINGING_STRIKE,  # Swinging strike
    'W': PitchResult.SWINGING_STRIKE,  # Swinging strike (blocked)
    'Q': PitchResult.SWINGING_STRIKE,  # Swinging pitchout
    'T': PitchResult.SWINGING_STRIKE,  # Foul tip
    'L': PitchResult.SWINGING_STRIKE,  # Foul bunt
    'F': PitchResult.SWINGING_FOUL,  # Regular foul
    'R': PitchResult.SWINGING_FOUL,  # Foul pitchout
    'D': PitchResult.HIT,  # Hit
    'E': PitchResult.HIT,  # Hit, with runs scored
    'H': PitchResult.HIT,  # Hit by pitch, perhaps should be a separate category
    'X': PitchResult.OUT,  # In play, out(s)
}


class BaseballData:
    """
    This class loads, processes, and stores the baseball data used for training and evaluation.
    More specific datasets are created for training specific models.

    We do not use all the data available in this dataset, but the interesting column names
    are still kept.

    See https://www.kaggle.com/datasets/pschale/mlb-pitch-data-20152018/
    """

    PLAYER_NAMES = 'player_names.csv'
    GAMES = 'games.csv'
    AT_BATS = 'atbats.csv'
    PITCHES = 'pitches.csv'
    EJECTIONS = 'ejections.csv'

    @classmethod
    def load_with_cache(cls, processed_data: str = '../../processed_data/baseball_data.pth', raw_data_dir: str = '../../raw_data/') -> Self:
        """Cache the processed data if it doesn't exist, or load it"""

        print('Loading baseball data... ', end='')

        if os.path.isfile(processed_data):
            data = torch.load(processed_data)
            print('done')
            return data
        else:
            data = cls(raw_data_dir=raw_data_dir)
            torch.save(data, processed_data)
            return data

    def __init__(self, raw_data_dir: str = '../../raw_data/'):
        """
        Load raw baseball data from the specified directory.
        :param raw_data_dir: The directory containing the raw baseball data.
        """

        self.player_names: dict[int, str] = {}
        self.games: dict[int, Game] = {}
        self.at_bats: dict[int, AtBat] = {}
        self.pitchers: dict[int, Pitcher] = {}
        self.batters: dict[int, Batter] = {}
        self.pitches: list[Pitch] = []

        # Load player names
        with open(raw_data_dir + self.PLAYER_NAMES) as f:
            player_names = csv.reader(f, delimiter=',')
            next(player_names)  # Skip the header
            for person_id, first_name, last_name in tqdm(player_names, desc='Loading player names'):
                self.player_names[int(person_id)] = f'{first_name} {last_name}'

        # Load game data
        with open(raw_data_dir + self.GAMES) as f:
            games = csv.reader(f, delimiter=',')
            next(games)
            for (_, away_score, _, _, _, game_id, home_score,
                 _, _, _, _, _, _, _, weather, wind, _) in tqdm(games, desc='Loading game data'):
                self.games[int(game_id)] = Game(int(home_score), int(away_score))

        # Load at-bat data
        with open(raw_data_dir + self.AT_BATS) as f:
            at_bats = csv.reader(f, delimiter=',')
            next(at_bats)
            for (at_bat_id, batter_id, event, game_id, inning, outs, pitcher_team_score,
                 pitcher_hand, pitcher_id, batter_side, top_inning) in tqdm(at_bats, desc='Loading at-bat data'):

                at_bat_id = int(at_bat_id)
                batter_id = int(batter_id)
                pitcher_id = int(pitcher_id)
                game_id = int(game_id)

                if batter_id not in self.batters:
                    self.batters[batter_id] = Batter()
                if pitcher_id not in self.pitchers:
                    self.pitchers[pitcher_id] = Pitcher()

                outcome = AtBatOutcome.BASE if event in on_base_events else AtBatOutcome.OUT

                self.at_bats[at_bat_id] = AtBat(self.games.get(game_id, None), self.pitchers[pitcher_id],
                                                self.batters[batter_id], outcome_state=AtBatState(outcome),
                                                ab_id=at_bat_id)

        # Load individual pitch data
        with open(raw_data_dir + self.PITCHES) as f:
            pitches = csv.reader(f, delimiter=',')

            # Keep track of player statistics
            pitcher_statistics = {}
            batter_statistics = {}
            pitch_statistics_shape = (len(PitchType), ZONES_DIMENSION, ZONES_DIMENSION)
            velocities = []  # To normalize the velocity data

            # For now, we ignore most of the columns in the CSV
            next(pitches)
            for (pos_x, pos_z, start_speed, end_speed, spin_rate, spin_dir,
                 _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _,
                 pitch_result_code, _, pitch_type, _, batter_score, at_bat_id,
                 balls, strikes, outs, pitch_number, player_on_1b, on_2b, on_3b) in tqdm(pitches, desc='Loading pitch data'):

                at_bat = self.at_bats.get(int(float(at_bat_id)), None)
                zone = get_zone(None if not pos_x else 12 * float(pos_x),
                                None if not pos_z else 12 * float(pos_z))  # Convert from feet to inches
                pitch_type = pitch_type_mapping.get(pitch_type, None)
                pitch_result = pitch_result_mapping.get(pitch_result_code, None)

                pitch = Pitch(AtBatState(AtBatOutcome.NONE, int(float(balls)), int(float(strikes))),
                              at_bat, zone, pitch_type, start_speed, pitch_result)

                # Update the pitch entry
                if at_bat is not None and pitch.is_valid():
                    self.pitches.append(pitch)

                    # Update player statistics
                    def increment_statistic(statistic: Tensor, amount: float = 1):
                        for x, y in zone.coords:
                            statistic[pitch_type, x, y] += amount

                    if at_bat.pitcher not in pitcher_statistics:
                        pitcher_statistics[at_bat.pitcher] = {'total_thrown': torch.zeros(pitch_statistics_shape),
                                                              'total_velocity': torch.zeros(pitch_statistics_shape)}
                    increment_statistic(pitcher_statistics[at_bat.pitcher]['total_thrown'])
                    increment_statistic(pitcher_statistics[at_bat.pitcher]['total_velocity'], float(start_speed))
                    velocities.append(float(start_speed))

                    # Batter
                    if at_bat.batter not in batter_statistics:
                        batter_statistics[at_bat.batter] = {'total_encountered': torch.zeros(pitch_statistics_shape),
                                                            'total_swung': torch.zeros(pitch_statistics_shape),
                                                            'total_hits': torch.zeros(pitch_statistics_shape)}
                    increment_statistic(batter_statistics[at_bat.batter]['total_encountered'])
                    if pitch.result.batter_swung():
                        increment_statistic(batter_statistics[at_bat.batter]['total_swung'])
                    if pitch.result == PitchResult.HIT:
                        increment_statistic(batter_statistics[at_bat.batter]['total_hits'])

            # Add the aggregate statistics to the players, replacing blank statistics with zeros
            velocity_mean = torch.mean(torch.tensor(velocities))
            velocity_std = torch.std(torch.tensor(velocities))

            for pitcher, stats in pitcher_statistics.items():
                pitcher.set_throwing_frequency_data(stats['total_thrown'])

                avg_velocity = stats['total_velocity'] / stats['total_thrown']
                normalized_velocity = nan_to_num((avg_velocity - velocity_mean) / velocity_std)

                pitcher.set_average_velocity_data(normalized_velocity)

            for batter, stats in batter_statistics.items():
                batter.set_swinging_frequency_data(nan_to_num(stats['total_swung']))
                batter.set_batting_average_data(nan_to_num(stats['total_hits'] / stats['total_encountered']))


if __name__ == '__main__':
    data = BaseballData.load_with_cache()
