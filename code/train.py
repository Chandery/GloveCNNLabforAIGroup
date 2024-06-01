import dataloader
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from accelerate import Accelerator
import matplotlib.pyplot as plt
from dataloader import BrainWaveDataset
from timm.utils import AverageMeter
from torch.optim.lr_scheduler import StepLR
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--model", type=str, help="Model to train")
    parser.add_argument("--init_lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="epochs")
    return parser.parse_args()

def train(args):
    batch_size = 64
    # data_path = ['/home/cdy/GloveCNN/dataset/data1/cl.mat']
    # data_path_train = ['/home/cdy/GloveCNN/dataset/data1/cl.mat',
    #                 '/home/cdy/GloveCNN/dataset/data1/cyy.mat',
    #                 '/home/cdy/GloveCNN/dataset/data1/kyf.mat',
    #                 '/home/cdy/GloveCNN/dataset/data1/lnn.mat',
    #                 '/home/cdy/GloveCNN/dataset/data2/ls.mat',
    #                 '/home/cdy/GloveCNN/dataset/data2/ry.mat']
    
    # data_path_valid = ['/home/cdy/GloveCNN/dataset/data2/wcf.mat',
    #                 '/home/cdy/GloveCNN/dataset/data3/wx.mat']
    
    # train_dataset = BrainWaveDataset(data_path=data_path_train, num_class=7,split='train')
    # train_dataloader  = DataLoader(train_dataset,  batch_size = batch_size, shuffle = False)
    # valid_dataset = BrainWaveDataset(data_path=data_path_valid, num_class=7,split='valid')
    # valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle = False)

    # features_path = "/home/cdy/GloveCNN/dataset3a/Features.npy"
    # label_path = "/home/cdy/GloveCNN/dataset3a/Label.npy"

    features_path = "/home/cdy/GloveCNN/dataset2b/Features.npy"
    label_path = "/home/cdy/GloveCNN/dataset2b/Labels.npy"
    valid_len_path = "/home/cdy/GloveCNN/dataset2b/valid_len.npy"

    train_dataset = BrainWaveDataset(features_path=features_path, label_path=label_path, valid_len_path = valid_len_path, num_class=2,split='train')
    train_dataloader  = DataLoader(train_dataset,  batch_size = batch_size, shuffle = False)

    valid_dataset = BrainWaveDataset(features_path=features_path, label_path=label_path, valid_len_path = valid_len_path, num_class=2,split='valid')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle = False)

    test_dataset = BrainWaveDataset(features_path=features_path, label_path=label_path, valid_len_path = valid_len_path, num_class=2,split='predict')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)



    
    # if(args.model == "AlexNet"):
    # from models.DeepConvNet import DeepConvNet
    # model = DeepConvNet(type = "train")
    # from models.EEGNet import EEGNet
    # model = EEGNet(type = "train")
    # from models.ResNet import ResNet
    # model = ResNet(type = "train")
    # from models.AlexNet import AlexNet
    # model = AlexNet()
    # from models.ShallowConvNet import ShallowConvNet
    # model = ShallowConvNet()
    from models.EEGtrans import EEGtrans
    model = EEGtrans()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # *initialize the weights
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.init_lr)

    scheduler = StepLR(optimizer, step_size = 50, gamma = 0.8)

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    train_epochs_acc = []
    valid_epochs_loss = []
    valid_epochs_acc = []

    # train_loss = AverageMeter()
    model.train()

    for epoch in range(args.epochs):
        print("#"*20+f"epoch{epoch}/{args.epochs}"+"#"*20)

        train_epoch_loss = []
        total = 0
        total_correct = 0
        total0 = 0
        for idx,(data_x,data_y) in enumerate(train_dataloader,0):

            data_x = data_x.type(torch.float32).to(device)
            data_y = data_y.type(torch.float32).to(device)

            # print(data_x.shape, data_y.shape)

            outputs = model(data_x)
            # print(outputs.shape)
            optimizer.zero_grad()
            loss = criterion(outputs,data_y)
            # print(data_y, outputs)
            loss.backward()
            optimizer.step()
            # train_loss.update(val=loss.item(),n=batch_size)
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())

            for i in range(0, len(data_y)):
                for j in range(2):
                    if data_y[i][j]==1:
                        if outputs[i].argmax()==j:
                            total_correct+=1
                            if j==0: total0 += 1
                        total+=1
                        break

            # if idx%(len(train_dataloader)//2)==0:
                # print(f"epoch={epoch}/{args.epochs},{idx}/{len(train_dataloader)} of train,loss={loss.item}")
            
            # print(f"{idx}:", loss)
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_epochs_acc.append(total_correct/total)
        print(f"train_accuracy={total_correct/total}")
        print(f"train_accuracy0={total0/total}")
        
        #=====================valid============================
        with torch.no_grad():

            total_correct = 0
            total = 0

            valid_epoch_loss = []
            for idx,(data_x,data_y) in enumerate(valid_dataloader,0):
                data_x = data_x.to(torch.float32).to(device)
                data_y = data_y.to(torch.float32).to(device)
                outputs = model(data_x)
                loss = criterion(outputs,data_y)
                valid_epoch_loss.append(loss.item())
                valid_loss.append(loss.item())

                for i in range(0, len(data_y)):
                    for j in range(2):
                        if data_y[i][j]==1:
                            if outputs[i].argmax()==j:
                                total_correct+=1
                            total+=1
                            break

        valid_epochs_loss.append(np.average(valid_epoch_loss))
        valid_epochs_acc.append(total_correct/total)

        print(f"train_loss={train_epochs_loss[-1]}, valid_loss={valid_epochs_loss[-1]}")
        print(f"valid_accuracy={total_correct/total}")
        print(f"Learning rate={scheduler.get_last_lr()[0]}")


        scheduler.step()

        # *Save checkpoint
        if (epoch+1) % 50 == 0:
            torch.save(
                {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                },
                os.path.join('/home/cdy/GloveCNN/code/checkpoint/splitbatch', f'{model._get_name()}checkpoint_{epoch+1}.pt'),
            )

        # *Save last checkpoint
        torch.save(
            {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            },
            os.path.join('/home/cdy/GloveCNN/code/checkpoint/splitbatch', f'{model._get_name()}_last_checkpoint.pt'),
        )

    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(train_loss[:])
    plt.title("train_loss")
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:],'-o',label="train_loss")
    plt.plot(valid_epochs_loss[1:],'-o',label="valid_loss")
    plt.plot(valid_epochs_acc[1: ],'-o',label="valid_acc")
    plt.plot(train_epochs_acc[1: ],'-o',label="train_acc")
    plt.title("epochs_loss")
    plt.legend()
    plt.savefig("loss.png")
    plt.show()


    # model = EEGNet(type = 'pred')
    # model.to(device)
    # model.eval()
    # pred = None
    # gt = None

    # for idx,(feature, gt_y) in enumerate(test_dataloader):
    #     feature = feature.type(torch.float32).to(device)
    #     # print(feature.shape)
        
    #     # with torch.no_grad():
    #     output = model(feature) # *(batch_size, num_class)
    #     output_pred = output.argmax(dim=1) # *(batch_size)
        

    #     pred = output_pred if idx == 0 else torch.cat((pred, output_pred))
    #     gt = gt_y if idx == 0 else torch.cat((gt, gt_y))

    # # print(output_pred.shape)

    # # eval
    # gt = gt.type(type(pred)).to('cpu')
    # pred = pred.to('cpu')
    # True_list = (pred == gt)
    # total_correct = True_list.sum().item()
    # total = len(gt)

    # print("test acc=",total_correct/total)





if __name__ == '__main__':
    args = parse_args()
    train(args)