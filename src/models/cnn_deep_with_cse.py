import torch
import torch.nn as nn
import torch.nn.functional as F

# CANDIDATE: You must define the model yourself. So not use any off-the-shelf models here
# Level 1: Add more layers to rach at least 85% accuracy on the validation dataset
# Level 2: Investigate batch normalization and dropout to improve the model's performance
# Level 3: Implement attention mechanisms to improve the model's performance

class CBAMChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        bs, c, _, _ = x.shape
        avg_out = self.fc2(F.relu(self.fc1(self.global_avg_pool(x).view(bs, c))))
        max_out = self.fc2(F.relu(self.fc1(self.global_max_pool(x).view(bs, c))))
        y = torch.sigmoid(avg_out + max_out).view(bs, c, 1, 1)
        return x * y

class CBAMSpatialAttention(nn.Module):
    """
    Replaced the SE blocks with Convolutional Block Attention Module (CBAM), 
    which includes both channel and spatial attention mechanisms. This should 
    enhance feature extraction and improve classification accuracy.
    """
    def __init__(self, kernel_size=7):
        super(CBAMSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * y

class CNNModelWithConvSEBlock(nn.Module):
    def __init__(self, no_classes):
        super(CNNModelWithConvSEBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.ca1 = CBAMChannelAttention(32)
        self.sa1 = CBAMSpatialAttention()
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.ca2 = CBAMChannelAttention(64)
        self.sa2 = CBAMSpatialAttention()
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.ca3 = CBAMChannelAttention(128)
        self.sa3 = CBAMSpatialAttention()
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.ca4 = CBAMChannelAttention(256)
        self.sa4 = CBAMSpatialAttention()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, no_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.sa1(self.ca1(self.bn1(self.conv1(x))))))
        x = self.pool(F.relu(self.sa2(self.ca2(self.bn2(self.conv2(x))))))
        x = self.pool(F.relu(self.sa3(self.ca3(self.bn3(self.conv3(x))))))
        x = self.pool(F.relu(self.sa4(self.ca4(self.bn4(self.conv4(x))))))
        
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
