""""Landslide Detection Mapping Employing CNN, ResNet, and DenseNet in the Three Gorges Reservoir, China"""
""""Fig. 4. Structure of CNN."""

import torch.nn as nn
import config
import torch

config = config.config

class AlexNet_LSM(nn.Module):
    def __init__(self, in_chanel):
        super(AlexNet_LSM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chanel, 64, 3, 1, 1),  
            nn.ReLU(),
            nn.BatchNorm2d(64)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 1, 1)
        )
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(21 * 21 * 256, 2)  
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = self.fc(self.dropout(torch.flatten(x, start_dim=1)))
        out = self.softmax(x)
        return out