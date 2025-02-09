import torch
import torch.nn as nn
import torch.nn.functional as F

# CANDIDATE: You must define the model yourself. So not use any off-the-shelf models here
# Level 1: Add more layers to rach at least 85% accuracy on the validation dataset
# Level 2: Investigate batch normalization and dropout to improve the model's performance
# Level 3: Implement attention mechanisms to improve the model's performance

class CNNModelBase(nn.Module):
    def __init__(self, no_classes):
        super(CNNModelBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 256, 3)

        self.fc1 = nn.Linear(256, no_classes)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.fc1(x)
        return x
