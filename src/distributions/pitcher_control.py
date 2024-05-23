import os

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_loading import BaseballData
from src.data.datasets import PitchDataset
from src.model.pitch_type import PitchType


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PitcherControl(nn.Module):
    """
    This network is used to learn a pitcher's control, assumed to be a bivariate normal distribution.
    """

    def __init__(self):
        super(PitcherControl, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.conv_1 = nn.Conv2d(len(PitchType), 32, 3)
        self.conv_2 = nn.Conv2d(32, 64, 3)

        self.linear_1 = nn.Linear(64, 64)
        self.linear_2 = nn.Linear(64, 64)

        # Concatenate the pitcher and pitch embeddings
        self.linear_3 = nn.Linear(70, 32)

        self.mu_x = nn.Linear(32, 1)
        self.mu_y = nn.Linear(32, 1)
        self.var_x = nn.Linear(32, 1)
        self.var_y = nn.Linear(32, 1)
        self.covar_xy = nn.Linear(32, 1)

    def forward(self, pitcher: Tensor, pitch: Tensor) -> Tensor:
        pitcher = self.dropout(pitcher)
        pitcher = F.relu(self.conv_1(pitcher))
        pitcher = F.relu(self.conv_2(pitcher))
        pitcher = pitcher.flatten(1)
        pitcher = F.relu(self.linear_1(pitcher))
        pitcher = torch.cat((pitcher, pitch), dim=1)
        pitcher = F.relu(self.linear_2(pitcher))

        mu_x = self.mu_x(pitcher)
        mu_y = self.mu_y(pitcher)
        var_x = F.softplus(self.var_x(pitcher))
        var_y = F.softplus(self.var_y(pitcher))
        covar_xy = self.covar_xy(pitcher)

        return torch.cat((mu_x, mu_y, var_x, var_y, covar_xy), dim=1)


def train(epochs: int = 30, batch_size: int = 64, learning_rate: float = 0.001,
          path: str = '../../model_weights/pitcher_control.pth'):
    data = BaseballData.load_with_cache()

    # Filter the dataset to only include pitches where the pitcher has a 3-0 count
    training_dataset, validation_dataset = PitchDataset.get_split_on_attribute(
        data,
        val_split=0.2,
        attribute=lambda pitch: pitch.pitcher,
        filter_on=lambda pitch: pitch.at_bat_state.balls == 3 and pitch.at_bat_state.strikes == 0,
        map_to=lambda pitch: ((pitch.pitcher.data, pitch.type.get_one_hot_encoding()), pitch.pitcher.estimated_control)
    )

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    model = PitcherControl().to(device)

    print(f'Using device: {device}')

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    loader_length = len(training_dataloader)

    for epoch in range(epochs):
        print(f'Running epoch {epoch + 1}\n')

        model.train()
        training_loss = 0
        for i, ((pitcher, pitch_type), distribution) in tqdm(enumerate(training_dataloader), leave=True, total=loader_length):
            pitcher, pitch_type, distribution = pitcher.to(device), pitch_type.to(device), distribution.to(device)

            optimizer.zero_grad()
            output = model.forward(pitcher, pitch_type)
            loss: Tensor = criterion(output, distribution)
            loss.backward()
            optimizer.step()
            training_loss += loss

            _, predicted = torch.max(output.data, 1)

        torch.save(model.state_dict(), path if path else 'model.pth')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, ((pitcher, pitch_type), distribution) in enumerate(validation_dataloader):
                pitcher, pitch_type, distribution = pitcher.to(device), pitch_type.to(device), distribution.to(device)

                output = model.forward(pitcher, pitch_type)
                total_loss += criterion(output, distribution)

            print(f'Epoch {epoch + 1}, '
                  f'training loss: {1000 * training_loss / len(training_dataset)}, '
                  f'average loss: {1000 * total_loss / len(validation_dataset)}')

        scheduler.step()


if __name__ == '__main__':
    train()
