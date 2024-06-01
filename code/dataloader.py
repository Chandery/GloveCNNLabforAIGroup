import scipy.io as sio
from torch.utils.data import Dataset,DataLoader
import numpy as np
import random
import pandas as pd

class BrainWaveDataset(Dataset):
    def __init__(self, features_path, label_path, valid_len_path, num_class, split='train'):
    # def __init__(self, data_path , num_class,split='train'):
        self.num_class = num_class

        self.data_list = None
        self.label_list = None
        
        print(f"------------start loading {split}set-------------")
        # for index,i in enumerate(data_path):
        #     mat = sio.loadmat(i)

        #     data = mat['data'].transpose(2,0,1)
        #     label = mat['label'].squeeze()

        #     # for idx in range(len(label)):
        #     #     label[idx] = 2 if label[idx] == 2 else 1

        #     label_one_hot = self.one_hot(label)

        #     self.data_list = data if index==0 else np.concatenate((self.data_list,data),axis=0)
        #     if split == 'predict':
        #         self.label_list = label-1 if index==0 else np.concatenate((self.label_list,label-1),axis=0)
        #     else:
        #         self.label_list= label_one_hot if index==0 else np.concatenate((self.label_list,label_one_hot),axis=0)
            
        #     print(f"loading {i} finished")

        self.data_list = np.load(features_path)
        
        label_list = np.load(label_path).astype(np.int32)

        self.valid_len = np.load(valid_len_path)

        # self.label_list = label_list if split == 'predict' else self.one_hot(label_list)
        self.label_list = label_list-1 if split == 'predict' else self.one_hot(label_list)
        # self.label_list = self.one_hot(label_list)

        # print(np.isnan(self.data_list).any())
        # print(np.isnan(self.label_list).any())

        
        print(f"------------finish loading {split}set-------------")

        if split=='train':
            start = 0
            end = int(0.7*len(self.data_list))
            # end = 10
        elif split=='valid':
            start = int(0.7*len(self.data_list))
            end = int(0.9*len(self.data_list))
        elif split=='predict':
            start = int(0.9*len(self.data_list))
            end = int(len(self.data_list))

        self.data_list = self.data_list[start:end,::]
        
        if split == 'predict': 
            self.label_list = self.label_list[start:end]
        else:
            self.label_list = self.label_list[start:end,::]
        
        self.valid_len = self.valid_len[start:end]

        # self.label_list = self.label_list[start:end,::]

        print(self.__len__())
        
        

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = data[None,::]

        # min-max normalization
        valid_len = self.valid_len[idx]
        max_val = np.max(data[0 : valid_len, :])
        min_val = np.min(data[0 : valid_len, :])
        if max_val == min_val:
            norm = data
        else:
            norm = data.copy()  
            norm[0 : valid_len, :] = (data[0 : valid_len, :] - min_val) / (max_val - min_val) 

        label = self.label_list[idx]
        return norm, label
    
    def __len__(self):
        return len(self.label_list) 


    def one_hot(self,label):
        label_onehot = np.zeros((len(label), self.num_class))
        for i in range(len(label)):
            label_onehot[i][label[i]-1] = 1  # *label starts from 1
        return label_onehot



if __name__ == '__main__':
    datapath = ['/home/cdy/GloveCNN/dataset/data1/cl.mat']

    # features_path = "/home/cdy/GloveCNN/dataset3a/Features.npy"
    # label_path = "/home/cdy/GloveCNN/dataset3a/Label.npy"

    data = BrainWaveDataset(data_path=datapath, num_class=2,split='train')
    # data = BrainWaveDataset(features_path= features_path, label_path= label_path, num_class=2,split='train')

    dataloader = DataLoader(dataset=data,batch_size=2,shuffle=True)
    for data,label in dataloader:
        print(data.shape)
        print(label.shape)
        break
       
