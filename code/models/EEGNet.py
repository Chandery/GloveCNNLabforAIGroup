import torch.nn as nn
import torch

class EEGNet(nn.Module):
    # kernel_size = (1, half of the time rate) 250HZ
    def __init__(self, F1 = 8, D = 2, F2 = 16, kernel_size = (1, 125), dropout_rate = 0.25, pool_size = (1, 4), norm_rate = 0.25, dropout_type = 'Dropout', C = 6, T = 18633, type = 'train'):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size, padding='same', bias = False)
        self.bn1 = nn.BatchNorm2d(F1, False)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, padding='valid', bias = False)
        self.bn2 = nn.BatchNorm2d(F1 * D, False)
        self.elu = nn.ELU()
        self.avgPool = nn.AvgPool2d(pool_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias = False)
        self.bn3 = nn.BatchNorm2d(F2, False)
        self.avgPool2 = nn.AvgPool2d((1, 8))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(16*582, 2)
        # if type == 'pred':
        #     self.softmax = nn.Softmax(dim=1)
        # elif type == 'train':
        #     self.softmax = None
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = self.avgPool(self.elu(x))
        x = self.dropout(x)
        x = self.separableConv(x)
        x = self.bn3(x)
        x = self.avgPool2(self.elu(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        
        return x

if __name__=='__main__':
    model = EEGNet()

    x = torch.randn(32,1,6,18633)
    print(model._get_name())
    
    for layer in model.children():
        x = layer(x)
        print(layer._get_name(), x.shape)
    