import dataloader
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
import matplotlib.pyplot as plt
from dataloader import BrainWaveDataset
from models.EEGNet import EEGNet
from models.DeepConvNet import DeepConvNet
from models.ShallowConvNet import ShallowConvNet
from models.EEGtrans import EEGtrans
from models.ResNet import ResNet
from sklearn.metrics import confusion_matrix
import os


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """

    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='all')
    # cm = confusion_matrix(y_true=label_true, y_pred=label_pred)
    # print(type(cm))
    plt.figure(figsize=(12, 6))

    # plt.subplot(121)

    plt.grid(False)
    plt.imshow(cm, cmap='YlGnBu')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            # color = (1, 1, 1) if i==j else (0, 0, 0) 
            color = (0, 0, 0)
            value = '{:.2%}'.format(cm[j, i])  
            # value = cm[j, i]
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    if pdf_save_path is not None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
    plt.show()


def predict():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model = EEGNet(type = "pred")
    # model = DeepConvNet(type = "pred")
    # model = EEGtrans()
    # model = ShallowConvNet(type = "pred")
    # model = ResNet(type = "pred")
    model = model.to(device)

    # *load model
    ckpt = torch.load('/home/cdy/GloveCNN/code/checkpoint/splitbatch/EEGNetcheckpoint_50.pt') 
    model.load_state_dict(ckpt['model'])
    model.eval()

    # data_path = ['/home/cdy/GloveCNN/dataset/data1/cl.mat',
    #              '/home/cdy/GloveCNN/dataset/data2/wcf.mat']

    # features_path = "/home/cdy/GloveCNN/dataset3a/Features.npy"
    # label_path = "/home/cdy/GloveCNN/dataset3a/Label.npy"

    features_path = "/home/cdy/GloveCNN/dataset2b/Features.npy"
    label_path = "/home/cdy/GloveCNN/dataset2b/Labels.npy"
    valid_len_path = "/home/cdy/GloveCNN/dataset2b/valid_len.npy"

    # train_dataset = BrainWaveDataset(features_path=features_path, label_path=label_path, num_class=4,split='train')
    # train_dataloader  = DataLoader(train_dataset,  batch_size = batch_size, shuffle = False)

    # valid_dataset = BrainWaveDataset(features_path=features_path, label_path=label_path, num_class=4,split='valid')
    # valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle = False)

    batch_size = 64

    # test_dataset = BrainWaveDataset(features_path=features_path, label_path=label_path, num_class=2,split='predict')
    # predict_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

    # predict_dataset = BrainWaveDataset(data_path=data_path,num_class=7,split='predict')
    # predict_dataloader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = BrainWaveDataset(features_path=features_path, label_path=label_path, valid_len_path = valid_len_path, num_class=2,split='predict')
    predict_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)


    pred = None
    gt = None
    total = 0
    total_correct = 0

    for idx,(feature, gt_y) in enumerate(predict_dataloader):
        feature = feature.type(torch.float32).to(device)
        # print(feature.shape)
        
        # with torch.no_grad():
        output = model(feature) # *(batch_size, num_class)
        output_pred = output.argmax(dim=1) # *(batch_size)
        

        # for i in range(0, len(gt_y)):
        #     for j in range(2):
        #         if gt_y[i][j]==1:
        #             if output[i].argmax()==j:
        #                 total_correct+=1
        #             total+=1
        #             break

        pred = output_pred if idx == 0 else torch.cat((pred, output_pred))
        gt = gt_y if idx == 0 else torch.cat((gt, gt_y))

        # print(output_pred.shape)
    
    # eval
    for i in range(len(gt)):
        print(gt[i])

    gt = gt.type(type(pred)).to('cpu')
    pred = pred.to('cpu')
    True_list = (pred == gt)
    total_correct = True_list.sum().item()
    total = len(gt)

    draw_confusion_matrix(gt, pred, ['0','1'], title="Confusion Matrix", pdf_save_path="confusion_matrix.png", dpi=100)

    cm = confusion_matrix(y_true=gt, y_pred=pred)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1 = 2 * Precision * Recall / (Precision + Recall)

    print(f"Accuracy: {total_correct/total}")
    print(f"Recall: {Recall}")
    print(f"Precision: {Precision}")
    print(f"F1: {F1}")
    print(gt.sum().item())
    print(len(gt))

    
if __name__ == '__main__':
    predict()
