import os

import torch
from torch import nn, Tensor
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from src.data.data_loading import BaseballData
from src.data.datasets import PitchSwingDataset, SwingResult
from src.model.pitch import PitchType, Pitch
from src.model.zones import ZONES_DIMENSION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SwingOutcomeLateFusion(nn.Module):
    """
    This network uses a "late fusion" architecture, merging the pitcher, batter, and pitch embeddings
    after a series of convolutional layers, along with the strike/ball count.

    It outputs the distribution of swing outcomes (strike, foul, hit, out).
    """

    def __init__(self):
        super(SwingOutcomeLateFusion, self).__init__()

        # Stateless modules
        self.padding = nn.ZeroPad2d((2, 1, 2, 1))  # To replicate padding='same' for the max pool layers
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # Pitcher layers
        self.p_conv_1 = nn.Conv2d(2 * len(PitchType), 32, 4, padding='same')
        self.p_conv_2 = nn.Conv2d(32, 32, 3, padding='same')
        self.p_conv_3 = nn.Conv2d(32, 32, 2, padding='same')
        self.p_conv_4 = nn.Conv2d(32, 64, (2, 5), padding='same')
        self.p_conv_5 = nn.Conv2d(64, 64, (2, 4), padding='same')
        self.p_conv_6 = nn.Conv2d(64, 64, (2, 3), padding='same')
        self.p_conv_7 = nn.Conv2d(64, 64, (2, 2), padding='same')

        # Batter layers
        self.b_conv_1 = nn.Conv2d(2 * len(PitchType), 32, 4, padding='same')
        self.b_conv_2 = nn.Conv2d(32, 32, 3, padding='same')
        self.b_conv_3 = nn.Conv2d(32, 32, 2, padding='same')
        self.b_conv_4 = nn.Conv2d(32, 64, (2, 3), padding='same')
        self.b_conv_5 = nn.Conv2d(64, 64, 3, padding='same')
        self.b_conv_6 = nn.Conv2d(64, 64, (2, 3), padding='same')
        self.b_conv_7 = nn.Conv2d(64, 64, 2, padding='same')

        # Pitch layers
        self.pp_conv_1 = nn.Conv2d(len(PitchType), 32, 3, padding='same')

        # After flattening and merging pitcher, batter, pitch, and strike/ball count
        output_size = 642
        self.linear_1 = nn.Linear(output_size, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, len(SwingResult))

    def infer(self, pitch: Pitch) -> dict[SwingResult, float]:
        output = self.forward(pitch.at_bat.pitcher.data, pitch.at_bat.batter.data,
                              pitch.get_one_hot_encoding(),
                              torch.tensor(pitch.at_bat_state.strikes, dtype=torch.float32),
                              torch.tensor(pitch.at_bat_state.balls, dtype=torch.float32))

        return {SwingResult(i): output[i].item() for i in range(len(SwingResult))}

    def forward(self, pitcher: Tensor, batter: Tensor, pitch: Tensor, strikes: Tensor, balls: Tensor) -> Tensor:
        pitcher = self.relu(self.p_conv_1(pitcher))
        pitcher = self.relu(self.p_conv_2(pitcher))
        pitcher = self.relu(self.p_conv_3(pitcher))
        pitcher = self.padding(self.pool(pitcher))
        pitcher = self.relu(self.p_conv_4(pitcher))
        pitcher = self.padding(self.pool(pitcher))
        pitcher = self.relu(self.p_conv_5(pitcher))
        pitcher = self.padding(self.pool(pitcher))
        pitcher = self.relu(self.p_conv_6(pitcher))
        pitcher = self.padding(self.pool(pitcher))
        pitcher = self.relu(self.p_conv_7(pitcher))
        pitcher = self.pool(pitcher)
        pitcher = pitcher.flatten(1)

        batter = self.relu(self.b_conv_1(batter))
        batter = self.relu(self.b_conv_2(batter))
        batter = self.relu(self.b_conv_3(batter))
        batter = self.padding(self.pool(batter))
        batter = self.relu(self.b_conv_4(batter))
        batter = self.padding(self.pool(batter))
        batter = self.relu(self.b_conv_5(batter))
        batter = self.padding(self.pool(batter))
        batter = self.relu(self.b_conv_6(batter))
        batter = self.padding(self.pool(batter))
        batter = self.relu(self.b_conv_7(batter))
        batter = self.pool(batter)
        batter = batter.flatten(1)

        pitch = self.relu(self.pp_conv_1(pitch))
        pitch = self.pool(pitch)
        pitch = pitch.flatten(1)

        strikes = strikes.unsqueeze(1)
        balls = balls.unsqueeze(1)

        output = torch.cat((pitcher, batter, pitch, strikes, balls), dim=1)
        output = self.sigmoid(self.linear_1(output))
        output = self.sigmoid(self.linear_2(output))
        output = self.sigmoid(self.linear_3(output))
        output = self.output(output)
        output = self.softmax(output)
        return output


class SwingOutcome(nn.Module):
    def __init__(self):
        super(SwingOutcome, self).__init__()
        self.relu = nn.ReLU()

        output_size = 128
        post_size = ZONES_DIMENSION - 1
        self.p_conv = nn.Conv2d(2 * len(PitchType), 32, 2)
        self.p_dropout = nn.Dropout(0.25)
        self.p_linear = nn.Linear(post_size * post_size * 32, output_size)

        self.b_conv = nn.Conv2d(2 * len(PitchType), 32, 2)
        self.b_dropout = nn.Dropout(0.25)
        self.b_linear = nn.Linear(post_size * post_size * 32, output_size)

        self.pitch_conv = nn.Conv2d(len(PitchType), 16, 2)
        self.pitch_linear = nn.Linear(post_size * post_size * 16, output_size)

        self.linear_1 = nn.Linear(3 * output_size + 2, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, len(SwingResult))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pitcher: Tensor, batter: Tensor, pitch: Tensor, strikes: Tensor, balls: Tensor) -> Tensor:
        pitcher = self.relu(self.p_conv(pitcher))
        pitcher = pitcher.flatten(1)
        pitcher = self.p_dropout(pitcher)
        pitcher = self.relu(self.p_linear(pitcher))

        batter = self.relu(self.b_conv(batter))
        batter = batter.flatten(1)
        batter = self.b_dropout(batter)
        batter = self.relu(self.b_linear(batter))

        pitch = self.relu(self.pitch_conv(pitch))
        pitch = pitch.flatten(1)
        pitch = self.relu(self.pitch_linear(pitch))

        strikes = strikes.unsqueeze(1)
        balls = balls.unsqueeze(1)

        output = torch.cat((pitcher, batter, pitch, strikes, balls), dim=1)
        output = self.relu(self.linear_1(output))
        output = self.relu(self.linear_2(output))
        output = self.relu(self.linear_3(output))
        output = self.softmax(self.output(output))
        return output


def train(epochs: int = 30, batch_size: int = 64, learning_rate: float = 0.001,
          path: str | None = '../../model_weights/swing_outcome.pth'):
    data = BaseballData.load_with_cache()
    pitch_dataset = PitchSwingDataset(data)
    training_split = 0.8
    training_set, testing_set = random_split(pitch_dataset, [int(training_split * len(pitch_dataset)),
                                                             len(pitch_dataset) - int(training_split * len(pitch_dataset))])

    training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    testing_dataloader = DataLoader(testing_set, batch_size=batch_size, shuffle=True)

    # compiled_model = torch.compile(SwingOutcome())
    # model = compiled_model.to(device)
    model = SwingOutcome().to(device)

    print(f'Using device: {device}')

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f'Running epoch {epoch + 1}\n')

        train_total = 0
        train_correct = 0
        model.train()
        for i, ((pitcher, batter, pitch, strikes, balls), result) in enumerate(training_dataloader):
            pitcher, batter, pitch, strikes, balls = (pitcher.to(device), batter.to(device), pitch.to(device),
                                                      strikes.to(device), balls.to(device))
            result: Tensor = result.to(device)

            optimizer.zero_grad()
            output = model.forward(pitcher, batter, pitch, strikes, balls)
            loss: Tensor = criterion(output, result)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            train_total += result.size(0)
            train_correct += (predicted == result.argmax(dim=1)).sum().item()

        torch.save(model.state_dict(), path if path else 'model.pth')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for i, ((pitcher, batter, pitch, strikes, balls), result) in enumerate(testing_dataloader):
                pitcher, batter, pitch, strikes, balls = (pitcher.to(device), batter.to(device), pitch.to(device),
                                                          strikes.to(device), balls.to(device))
                result: Tensor = result.to(device)

                output = model(pitcher, batter, pitch, strikes, balls)
                total_loss += criterion(output, result)
                _, predicted = torch.max(output.data, 1)
                total += result.size(0)
                correct += (predicted == result.argmax(dim=1)).sum().item()

            print(f'Epoch {epoch + 1}, '
                  f'average loss: {total_loss / (len(testing_dataloader) * batch_size)}, '
                  f'training accuracy: {100 * train_correct / train_total}%, '
                  f'testing accuracy: {100 * correct / total}%')

        scheduler.step()


if __name__ == '__main__':
    train()
