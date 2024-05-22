import os

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_loading import BaseballData
from src.data.datasets import PitchSwingDataset, SwingResult
from src.model.pitch import PitchType
from src.model.zones import ZONES_DIMENSION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SwingOutcome(nn.Module):
    """
    This network learns the distribution of a swing based on the pitcher, batter, pitch, and current count.
    """

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
    training_set, testing_set = PitchSwingDataset.get_random_split(data, 0.2, seed=0)

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

    loader_length = len(training_dataloader)

    for epoch in range(epochs):
        print(f'Running epoch {epoch + 1}\n')

        train_total = 0
        train_correct = 0
        model.train()
        for i, ((pitcher, batter, pitch, strikes, balls), result) in tqdm(enumerate(training_dataloader), leave=True, total=loader_length):
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


def analyze_distribution(model_path: str = '../../model_weights/swing_outcome.pth'):
    data = BaseballData.load_with_cache()
    training_set, testing_set = PitchSwingDataset.get_random_split(data, 0.2, seed=0)

    # Note that the dataset puts strikes before balls
    pitches_30 = [pitch for pitch in training_set if pitch[0][3].item() == 0 and pitch[0][4].item() == 3]
    pitches_02 = [pitch for pitch in training_set if pitch[0][3].item() == 2 and pitch[0][4].item() == 0]

    actual_dist_30 = 100 * sum([pitch[1] for pitch in pitches_30]) / len(pitches_30)
    actual_dist_02 = 100 * sum([pitch[1] for pitch in pitches_02]) / len(pitches_02)

    model = SwingOutcome().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predicted_dist_30 = torch.zeros(1, len(SwingResult)).to(device)
    predicted_dist_02 = torch.zeros(1, len(SwingResult)).to(device)

    with torch.no_grad():
        for pitch in tqdm(pitches_30):
            pitcher, batter, pitch, strikes, balls = (pitch[0][0].unsqueeze(0).to(device), pitch[0][1].unsqueeze(0).to(device),
                                                      pitch[0][2].unsqueeze(0).to(device), pitch[0][3].unsqueeze(0).to(device),
                                                      pitch[0][4].unsqueeze(0).to(device))
            output = model(pitcher, batter, pitch, strikes, balls)
            predicted_dist_30 += output
        predicted_dist_30 /= len(pitches_30)

        for pitch in tqdm(pitches_02):
            pitcher, batter, pitch, strikes, balls = (pitch[0][0].unsqueeze(0).to(device), pitch[0][1].unsqueeze(0).to(device),
                                                      pitch[0][2].unsqueeze(0).to(device), pitch[0][3].unsqueeze(0).to(device),
                                                      pitch[0][4].unsqueeze(0).to(device))
            output = model(pitcher, batter, pitch, strikes, balls)
            predicted_dist_02 += output
        predicted_dist_02 /= len(pitches_02)

    print('Actual distribution for 3-0 count:', actual_dist_30)
    print('Predicted distribution for 3-0 count:', predicted_dist_30[0] * 100)
    print('Actual distribution for 0-2 count:', actual_dist_02)
    print('Predicted distribution for 0-2 count:', predicted_dist_02[0] * 100)


if __name__ == '__main__':
    analyze_distribution()
