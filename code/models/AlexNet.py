import torch.nn as nn
import torch
import numpy as np

class AlexNet(nn.Module):
    '''
    BatchNorm2d output shape:        torch.Size([1, 1, 64, 1600])
    Conv2d output shape:             torch.Size([1, 96, 15, 399])
    ReLU output shape:               torch.Size([1, 96, 15, 399])
    MaxPool2d output shape:          torch.Size([1, 96, 7, 199])
    Conv2d output shape:             torch.Size([1, 256, 7, 199])
    ReLU output shape:               torch.Size([1, 256, 7, 199])
    MaxPool2d output shape:          torch.Size([1, 256, 3, 99])
    Conv2d output shape:             torch.Size([1, 384, 3, 99])
    ReLU output shape:               torch.Size([1, 384, 3, 99])
    Conv2d output shape:             torch.Size([1, 384, 3, 99])
    ReLU output shape:               torch.Size([1, 384, 3, 99])
    Conv2d output shape:             torch.Size([1, 256, 3, 99])
    ReLU output shape:               torch.Size([1, 256, 3, 99])
    MaxPool2d output shape:          torch.Size([1, 256, 1, 49])
    Flatten output shape:            torch.Size([1, 12544])
    Linear output shape:             torch.Size([1, 4096])
    ReLU output shape:               torch.Size([1, 4096])
    Dropout output shape:            torch.Size([1, 4096])
    Linear output shape:             torch.Size([1, 4096])
    ReLU output shape:               torch.Size([1, 4096])
    Dropout output shape:            torch.Size([1, 4096])
    Linear output shape:             torch.Size([1, 7])
    Softmax output shape:            torch.Size([1, 7])
    '''
    
    def __init__(self):
        super(AlexNet, self).__init__()
        self.bn = nn.BatchNorm2d(1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=1)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 16, kernel_size=3, padding = 'valid')
        self.MaxPool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Flatten = nn.Flatten()
        self.fc1 = nn.Linear(2512, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.MaxPool1(self.relu(self.conv1(x)))
        x = self.MaxPool2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        # x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.MaxPool3(self.relu(self.conv6(x)))
        x = self.Flatten(x)
        x = self.relu(self.fc1(x))
        # x = self.softmax(x)
        return x
    
if __name__ == "__main__":
    # model = AlexNet()

    net = nn.Sequential(
    nn.BatchNorm2d(1, affine=True, track_running_stats=True),
    nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(256, 16, kernel_size=3, padding = 'valid'), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(2512, 2), nn.ReLU(),
    nn.Softmax(dim=1))

    # print(net)
    X = torch.randn(64,1,60,2560)
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)