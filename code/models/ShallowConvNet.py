import torch.nn as nn
import torch

class ShallowConvNet(nn.Module):
    def __init__(self,dropout_rate = 0.5, C = 6, type = 'train'):
        super(ShallowConvNet, self).__init__()
        self.Conv1 = nn.Conv2d(1, 40, (1, 25), padding='same')
        self.elu1 = nn.ELU()
        self.Conv2 = nn.Conv2d(40, 40, (C, 1), padding='valid')
        self.bn1 = nn.BatchNorm2d(40, False)
        self.avgPool = nn.AvgPool2d(kernel_size = (1, 75), stride = (1, 25))
        self.Conv3 = nn.Conv2d(40, 6, (1, 13), padding='same')
        self.elu2 = nn.ELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(4458, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.elu1(x)
        x = self.Conv2(x)
        x = self.bn1(x)
        x = self.avgPool(x)
        x = self.Conv3(x)
        x = self.elu2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x

if __name__=='__main__':
    model = ShallowConvNet()

    x = torch.randn(64,1,6,18633)
    
    for layer in model.children():
        x = layer(x)
        print(layer._get_name(), x.shape)
    