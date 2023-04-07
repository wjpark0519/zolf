import numpy as np
import torch
import torch.nn as nn

class ZolfModel(nn.Module):
    def __init__(self):
        super(ZolfModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 1000),
            # nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            # nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            # nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            # nn.BatchNorm1d(5000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            # nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 2),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x
        return x