import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return self.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class ResNet(nn.Module):
    def __init__(self, type = 'train'):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 125), stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.MaxPool1 = nn.MaxPool2d(kernel_size=(1, 8), stride=2)
        self.resnet_block1 = resnet_block(64, 64, 2, first_block=True)
        self.resnet_block2 = resnet_block(64, 128, 2)
        self.resnet_block3 = resnet_block(128, 256, 2)
        self.resnet_block4 = resnet_block(256, 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 2)
        # if type == 'train' or type == 'valid': 
        #     self.softmax = None
        # elif type == 'pred':
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.MaxPool1(x)
        x = self.resnet_block1(x)
        x = self.dropout(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.dropout(x)
        x = self.resnet_block4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    
if __name__=='__main__':
    model = ResNet("train")

    x = torch.randn(32,1,6,18633)
    print(model._get_name())
    
    for layer in model.children():
        x = layer(x)
        print(layer._get_name(), x.shape)
    