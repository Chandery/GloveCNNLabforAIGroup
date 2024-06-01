import torch.nn as nn
import torch
from torch.nn import Transformer

class EEGtrans(nn.Module):
    # kernel_size = (1, half of the time rate) 250HZ
    def __init__(self, F1 = 8, D = 2, F2 = 16, kernel_size = (1, 125), dropout_rate = 0.25, pool_size = (1, 4), norm_rate = 0.25, dropout_type = 'Dropout', C = 6, T = 18633, type = 'train', alpha = 0.05):
        super(EEGtrans, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, kernel_size, padding='same', bias = False)
        self.bn1 = nn.BatchNorm2d(F1, False)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, padding='valid', bias = False)
        self.bn2 = nn.BatchNorm2d(F1 * D, False)
        self.elu = nn.ELU()
        self.avgPool = nn.AvgPool2d(pool_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias = False)
        self.bn3 = nn.BatchNorm2d(F2, False)
        self.avgPool2 = nn.AvgPool2d((1, 5))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(16*931, 2)
        self.softmax = nn.Softmax(dim=1)
        # self.embedding = nn.Embedding(6*931, 520)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=520, nhead=8), 6)
        self.fc = nn.Linear(520, 931)
        self.embedding = nn.Linear(6*931, 520)
        self.feat_conv = nn.Conv2d(20, 20, kernel_size = (3, 125), padding='valid', bias=False)
        self.feat_bn = nn.BatchNorm2d(20, False)
        self.feat_avgPool = nn.AvgPool2d(kernel_size = (1, 7))
        self.conv11 = nn.Conv2d(20, 16, kernel_size = 1, bias=False)
        self.alpha = alpha

    def forward(self, x):
        # * EEGNet_output = (Bs, 16, 1, 931)
        EEGNet_output = self.EEGBlock()(x)
        
        # * split  (Bs, 1, 6, 18633)
        max_len = x.size(-1) // 20 * 20
        x_cut = torch.narrow(x, -1, 0, max_len)
        chunks = torch.chunk(x_cut, 20, dim=-1)  
        tokens = torch.stack(chunks, dim=1) 
        tokens = tokens.squeeze(2) 

        # * tokens.shape = (Bs, 20, 64, 80) 相当于一个token是一个(64, 80)的特征矩阵
        # * (Bs, 20, 6, 931) -> (Bs, 20, 4, 73) -> (Bs, 20, 460)
        # tokens = self.CatchFeature()(tokens)

        # * (Bs, 20, 6, 931) -> (Bs, 20, 6*931)
        tokens = tokens.view(tokens.size(0), tokens.size(1), -1)

        # * Embedding (Bs, 20, 6*931) -> (Bs, 20, 520)
        Trans_Input = self.embedding(tokens)
        # Trans_Input = tokens

        # * Transformer ##############################################
        Trans_output = self.transformer(Trans_Input)

        # * (Bs, 20, 520) -> (Bs, 20, 931)
        Trans_output = self.fc(Trans_output)
        # * (Bs, 20, 931) -> (Bs, 20, 1, 931)
        Trans_output = Trans_output.unsqueeze(2)
        # * (Bs, 20, 1, 931) -> (Bs, 16, 1, 931)
        Trans_output = self.conv11(Trans_output)

        ##############################################################

        # * (Bs, 16, 1, 931) + alpha*(Bs, 16, 1, 931) -> (Bs, 16, 1, 931)
        # output_cat = torch.cat((EEGNet_output, Trans_output), dim = 1)
        output_cat = EEGNet_output + Trans_output * self.alpha

        output_cat = self.flatten(output_cat)
        output = self.dense(output_cat)
        output = self.softmax(output)

        return output
    
    def CatchFeature(self):
        Catch_Feature = nn.Sequential(
            self.feat_conv,
            self.feat_bn,
            self.elu,
            self.feat_avgPool
        )
        return Catch_Feature

    def EEGBlock(self):
        EEG_Block = nn.Sequential(
            self.conv1,
            self.bn1,
            self.depthwiseConv,
            self.bn2,
            self.avgPool,self.elu,
            self.dropout,
            self.separableConv,
            self.bn3,
            self.avgPool2,self.elu,
            self.dropout
        )
        return EEG_Block

if __name__=='__main__':

    # x1 = torch.randn(2, 16, 1, 80)
    # x2 = torch.randn(2, 20, 1, 80)
    # y = torch.cat((x1, x2), dim = 1)

    # x = torch.randn(2, 1, 6, 18633)
    # max_len = x.size(-1) // 20 * 20
    # x_cut = torch.narrow(x, -1, 0, max_len)
    # chunks = torch.chunk(x_cut, 20, dim=-1)  
    # tokens = torch.stack(chunks, dim=1) 
    # tokens = tokens.squeeze(2) 
    # print(tokens.shape)

    # print(y.shape)


    # x = torch.randn(2, 20, 6, 931)

    # Catch_Feature = nn.Sequential(
    #         nn.Conv2d(20, 20, kernel_size = (3, 125), padding='valid', bias=False),
    #         nn.BatchNorm2d(20, False),
    #         nn.ELU(),
    #         nn.AvgPool2d(kernel_size = (1, 7))
    #     )
    
    # x = Catch_Feature(x)
    # print(x.shape)

    num_embeddings = 6*931
    x = torch.randn(2, 20, 6*931)
    x = nn.Linear(num_embeddings, 520)(x)
    print(x.shape)

    # F1 = 8
    # D = 2
    # F2 = 16
    # kernel_size = (1, 125)
    # dropout_rate = 0.25  
    # pool_size = (1, 4)
    # norm_rate = 0.25
    # C = 6
    # EEG_Block = nn.Sequential(
    #         nn.Conv2d(1, F1, kernel_size, padding='same', bias = False),
    #         nn.BatchNorm2d(F1, False),
    #         nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, padding='valid', bias = False),
    #         nn.BatchNorm2d(F1 * D, False),
    #         nn.AvgPool2d(pool_size),nn.ELU(),
    #         nn.Dropout(dropout_rate),
    #         nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias = False),
    #         nn.BatchNorm2d(F2, False),
    #         nn.AvgPool2d((1, 5)),nn.ELU(),
    #         nn.Dropout(dropout_rate)
    #     )
    # x = torch.rand(2, 1, 6, 18633)
    # for layers in EEG_Block:
    #     x = layers(x)
    #     print(x.shape)
    