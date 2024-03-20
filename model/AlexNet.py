import torch.nn as nn
import config
import torch

config = config.config

# class AlexNet_LSM(nn.Module):
#     def __init__(self, in_chanel):
#         super(AlexNet_LSM, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_chanel, 32, 3, 1, 1),  
#             nn.BatchNorm2d(32), 
#             nn.ReLU()
#             )
#         self.dropout1 = nn.Dropout(0.4)
#         self.subsample1 = nn.MaxPool2d(2)

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, 1, 1),  
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.dropout2 = nn.Dropout(0.3)
#         self.subsample2 = nn.MaxPool2d(2)
#         self.fc = nn.Linear(4 * 4 * 64, 2)  
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.subsample1(self.dropout1(self.conv1(x)))
#         x = self.subsample2(self.dropout2(self.conv2(x)))
#         x = self.fc(torch.flatten(x, start_dim=1))
#         out = self.softmax(x)
#         return out

class AlexNet_LSM(nn.Module):
    def __init__(self, in_chanel):
        super(AlexNet_LSM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chanel, 32, 3, 1, 1),  
            nn.BatchNorm2d(32), 
            nn.ReLU()
            )
        self.dropout1 = nn.Dropout(0.4)
        self.subsample1 = nn.MaxPool2d(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),  
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dropout2 = nn.Dropout(0.3)
        self.subsample2 = nn.MaxPool2d(2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),  
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dropout3 = nn.Dropout(0.3)
        self.subsample3 = nn.MaxPool2d(2)

        self.fc = nn.Linear(2 * 2 * 128, 2)   
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.subsample1(self.dropout1(self.conv1(x)))
        x = self.subsample2(self.dropout2(self.conv2(x)))
        x = self.subsample3(self.dropout3(self.conv3(x)))
        x = self.fc(torch.flatten(x, start_dim=1))
        out = self.softmax(x)
        return out
