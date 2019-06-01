import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F








class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool(2, 2)
        self.conv1 = nn.Conv2d(4, 6, 3, stride=2)
        self.conv2 = nn.Conv2d(6, 6, 4, stride=2)
        self.conv3 = nn.Conv2d(6, 6, 6, stride=2)

        self.fc1 = nn.Linear(6*70*70, 180)
        self.fc2 = nn.Linear(180, 140)
        self.fc3 = nn.Linear(140, 40)
        self.fc4 = nn.Linear(40, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(-1, 6*70*70)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return x
        




