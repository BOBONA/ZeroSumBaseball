import os

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

from src.data.data_loading import BaseballData
from src.data.datasets import SwingResult, PitchDataset
from src.model.pitch import PitchType, Pitch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SwingOutcome(nn.Module):
    """
    This network learns the distribution of a swing based on the pitcher, batter, pitch, and current count.
    """

    def __init__(self):
        super(SwingOutcome, self).__init__()

        assert len(PitchType) == 6  # Edit this line if you add more pitch types

        self.p_dropout_1 = nn.Dropout(0.2)
        self.p_conv_1 = nn.Conv2d(2 * len(PitchType), 64, 3)
        self.p_conv_2 = nn.Conv2d(64, 128, 3)
        self.p_linear = nn.Linear(128, 128)
        self.p_dropout_2 = nn.Dropout(0.25)

        self.b_dropout_1 = nn.Dropout(0.2)
        self.b_conv_1 = nn.Conv2d(2 * len(PitchType), 64, 3)
        self.b_conv_2 = nn.Conv2d(64, 128, 3)
        self.b_linear = nn.Linear(128, 128)
        self.b_dropout_2 = nn.Dropout(0.25)

        self.pitch_conv_1 = nn.Conv2d(len(PitchType), 32, 3)
        self.pitch_conv_2 = nn.Conv2d(32, 64, 3)
        self.pitch_linear = nn.Linear(64, 64)

        self.linear_1 = nn.Linear(128 + 128 + 64 + 1 + 1, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, len(SwingResult))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pitcher: Tensor, batter: Tensor, pitch: Tensor, strikes: Tensor, balls: Tensor) -> Tensor:
        pitcher = self.p_dropout_1(pitcher)
        pitcher = F.relu(self.p_conv_1(pitcher))
        pitcher = F.relu(self.p_conv_2(pitcher))
        pitcher = pitcher.flatten(1)
        pitcher = F.relu(self.p_linear(pitcher))
        pitcher = self.p_dropout_2(pitcher)

        batter = self.b_dropout_1(batter)
        batter = F.relu(self.b_conv_1(batter))
        batter = F.relu(self.b_conv_2(batter))
        batter = batter.flatten(1)
        batter = F.relu(self.b_linear(batter))
        batter = self.b_dropout_2(batter)

        pitch = F.relu(self.pitch_conv_1(pitch))
        pitch = F.relu(self.pitch_conv_2(pitch))
        pitch = pitch.flatten(1)
        pitch = F.relu(self.pitch_linear(pitch))

        strikes = strikes.unsqueeze(1)
        balls = balls.unsqueeze(1)

        output = torch.cat((pitcher, batter, pitch, strikes, balls), dim=1)
        output = F.relu(self.linear_1(output))
        output = F.relu(self.linear_2(output))
        output = F.relu(self.linear_3(output))
        output = self.softmax(self.output(output))
        return output


def map_swing_outcome(idx: int, pitch: Pitch):
    return (idx, (pitch.at_bat.pitcher.data, pitch.at_bat.batter.data,
                  pitch.get_one_hot_encoding(),
                  torch.tensor(pitch.at_bat_state.strikes, dtype=torch.float32),
                  torch.tensor(pitch.at_bat_state.balls, dtype=torch.float32)),
            SwingResult.from_pitch_result(pitch.result).get_one_hot_encoding())


def get_swing_outcome_dataset(data: BaseballData) -> [PitchDataset, PitchDataset]:
    return PitchDataset.get_split_on_attribute(
        data, 0.2,
        attribute=lambda p: p.at_bat.pitcher,  # Group by pitcher
        filter_on=lambda p: p.result.batter_swung(),
        map_to=map_swing_outcome,
        seed=80
    )


def train(epochs: int = 30, batch_size: int = 64, learning_rate: float = 0.001, gamma: float = 2.75,
          path: str = '../../model_weights/swing_outcome.pth'):
    data = BaseballData.load_with_cache()
    training_set, testing_set = get_swing_outcome_dataset(data)

    training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    testing_dataloader = DataLoader(testing_set, batch_size=batch_size)

    # If your system supports Triton, Torch 2.0 has a compile method that can speed up the model
    # compiled_model = torch.compile(SwingOutcome())
    # model = compiled_model.to(device)
    model = SwingOutcome().to(device)

    print(f'Using device: {device}')

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    # I am struggling to understand why the network won't learn the distribution
    # with CrossEntropyLoss. It predicts HITS at 0%. Focal loss seems to work, and
    # some testing yielded 2.75 as the best value. However, I am concerned that this
    # is merely patching up some other problem
    # criterion = nn.CrossEntropyLoss()
    criterion = lambda x, y: sigmoid_focal_loss(x, y, reduction='mean', gamma=gamma)

    loader_length = len(training_dataloader)

    for epoch in range(epochs):
        model.train()
        training_loss = 0
        for i, (pitch_idx, (pitcher, batter, pitch, strikes, balls), target) in tqdm(enumerate(training_dataloader),
                                                                                     leave=True,
                                                                                     total=loader_length,
                                                                                     desc=f'Epoch {epoch + 1}'):
            pitcher, batter, pitch, strikes, balls = (pitcher.to(device), batter.to(device), pitch.to(device),
                                                      strikes.to(device), balls.to(device))
            target: Tensor = target.to(device)

            optimizer.zero_grad()
            output = model.forward(pitcher, batter, pitch, strikes, balls)
            loss: Tensor = criterion(output, target)
            loss.backward()
            optimizer.step()
            training_loss += loss

            _, predicted = torch.max(output.data, 1)

        torch.save(model.state_dict(), path if path else 'model.pth')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (pitch_idx, (pitcher, batter, pitch, strikes, balls), result) in enumerate(testing_dataloader):
                pitcher, batter, pitch, strikes, balls = (pitcher.to(device), batter.to(device), pitch.to(device),
                                                          strikes.to(device), balls.to(device))
                result: Tensor = result.to(device)

                output = model(pitcher, batter, pitch, strikes, balls)
                total_loss += criterion(output, result)

            print(f'Epoch {epoch + 1}, '
                  f'training loss: {1000 * training_loss / len(training_set)}, '
                  f'testing loss: {1000 * total_loss / len(testing_set)}')

        scheduler.step()


if __name__ == '__main__':
    # Testing could probably find a slightly better gamma value between 2.6 and 3.0
    # However, this kind of testing really requires another split of the data
    train(epochs=10, learning_rate=0.0001, batch_size=512, gamma=2.75, path=f'../../model_weights/swing_outcome.pth')
