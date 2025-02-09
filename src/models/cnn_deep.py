import torch
import torch.nn as nn
import torch.nn.functional as F

# CANDIDATE: You must define the model yourself. So not use any off-the-shelf models here
# Level 1: Add more layers to rach at least 85% accuracy on the validation dataset
# Level 2: Investigate batch normalization and dropout to improve the model's performance
# Level 3: Implement attention mechanisms to improve the model's performance

class CNNModelDeep(nn.Module):
    """
    Key Improvements:
	1. More Layers: Increased convolutional layers to enhance feature extraction.
	2. Batch Normalization: Added after each convolution to stabilize learning and improve generalization.
	3. Dropout: Applied after the first fully connected layer to reduce overfitting.
    4. Increased Fully Connected Layer Depth: Added an intermediate layer to learn complex features.
    """
    def __init__(self, no_classes):
        super(CNNModelDeep, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, no_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
