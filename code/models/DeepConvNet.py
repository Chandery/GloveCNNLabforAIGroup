import torch.nn as nn
import torch

class DeepConvNet(nn.Module):
    def __init__(self,dropout_rate = 0.5, C = 6, type = 'train'):
        super(DeepConvNet, self).__init__()
        # *block1
        self.conv1 = nn.Conv2d(1, 25, (1, 5))
        self.conv2 = nn.Conv2d(25, 25, (C, 1))
        self.bn = nn.BatchNorm2d(25 , momentum = 0.9, eps = 1e-5)
        self.elu = nn.ELU()
        self.MaxPool = nn.MaxPool2d((1, 2), stride = (1, 2))
        self.dropout = nn.Dropout(dropout_rate)

        # *block2
        self.conv3 = nn.Conv2d(25, 50, (1, 5))
        self.bn2 = nn.BatchNorm2d(50 , momentum = 0.9, eps = 1e-5)
        self.elu2 = nn.ELU()
        self.MaxPool2 = nn.MaxPool2d((1, 2), stride = (1, 2))
        self.dropout2 = nn.Dropout(dropout_rate)

        # *block3
        self.conv4 = nn.Conv2d(50, 100, (1, 5))
        self.bn3 = nn.BatchNorm2d(100 , momentum = 0.9, eps = 1e-5)
        self.elu3 = nn.ELU()
        self.MaxPool3 = nn.MaxPool2d((1, 2), stride = (1, 2))
        self.dropout3 = nn.Dropout(dropout_rate)

        # *block4
        self.conv5 = nn.Conv2d(100, 200, (1, 5))
        self.bn4 = nn.BatchNorm2d(200 , momentum = 0.9, eps = 1e-5)
        self.elu4 = nn.ELU()
        self.MaxPool4 = nn.MaxPool2d((1, 7), stride = (1, 8))
        self.dropout4 = nn.Dropout(dropout_rate)

        self.conv6 = nn.Conv2d(200, 25, (1, 1))
        self.elu4 = nn.ELU()

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(7250, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # *block1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.MaxPool(self.bn(self.elu(x)))
        x = self.dropout(x)

        # *block2
        x = self.conv3(x)
        x = self.MaxPool(self.bn2(self.elu(x)))
        x = self.dropout(x)

        # *block3
        x = self.conv4(x)
        x = self.MaxPool(self.bn3(self.elu(x)))
        x = self.dropout(x)

        # *block4
        x = self.conv5(x)
        x = self.MaxPool4(self.bn4(self.elu(x)))
        x = self.dropout(x)

        x = self.conv6(x)
        x = self.elu(x)
        
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)

        
        return x

if __name__=='__main__':
    model = DeepConvNet()

    x = torch.randn(64,1,6,18633)
    
    for layer in model.children():
        x = layer(x)
        print(layer._get_name(), x.shape)
    