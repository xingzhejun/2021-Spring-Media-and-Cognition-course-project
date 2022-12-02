"""
自己修改后的模型
修改处为kernal size和dropout
"""

import torch.nn as nn


class base_model(nn.Module):
    def __init__(self, class_num=35):
        super(base_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.GAP = nn.AdaptiveMaxPool2d((1, 1))
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, class_num)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pooling(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.max_pooling(x)
        x = self.conv3(x)
        x = self.GAP(x).squeeze(dim=3).squeeze(dim=2)
        # you can see this x as the feature, and use it to visualize something

        x = self.fc1(self.relu(self.bn3(x)))
        x = self.fc2(self.relu(self.bn4(x)))
        return x
