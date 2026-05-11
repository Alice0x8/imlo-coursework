import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #Convolution Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        #Convolution Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #Convolution Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        #Pooling layer (reduces image size)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 37)  # 37 pet classes

        #Dropout (helps prevent overfitting)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        #Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        #Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        #Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        #Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x