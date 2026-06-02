import torch
import torch.nn as nn
import torch.nn.functional as F

class PokerEVNet(nn.Module):
    def __init__(self, input_size=109, output_size=5):
        super(PokerEVNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.head = nn.Linear(64, output_size)

    def forward(self, x):
        # We use ReLU activation to allow the network to learn non-linear patterns
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # The output represents the "Q-Value" for each action
        return self.head(x)
    