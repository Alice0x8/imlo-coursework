import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolution Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # batch normalisation to help it train faster by keeping it at a consistent scale of 0-1
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

        # Convolution Block 5 
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Max pooling halves width and height
        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling to reduce number of parameters to connect layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 37)

        # helps with overfitting, i found 0.3 to be the best value
        self.dropout = nn.Dropout(0.3)

    # passes input through the model
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
        
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x