import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Convolution Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Convolution Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Convolution Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Convolution Block 5 (new deeper layer)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Max pooling halves width and height
        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling reduces each feature map to one number
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier
        # Fully connected classifier (slightly increased capacity)
        self.fc1 = nn.Linear(512, 768)
        self.fc2 = nn.Linear(768, 384)
        self.fc3 = nn.Linear(384, 37)

        self.dropout = nn.Dropout(0.4)


    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Block 5
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        # Global average pooling
        x = self.gap(x)

        # Flatten from [batch, 512, 1, 1] to [batch, 512]
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x