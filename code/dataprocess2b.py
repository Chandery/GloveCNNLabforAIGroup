import scipy.io as sio
import pandas as pd
import numpy as np

path = ['BCI_IV2b_mat/B01T.mat',
        'BCI_IV2b_mat/B02T.mat',
        'BCI_IV2b_mat/B03T.mat',
        'BCI_IV2b_mat/B04T.mat',
        'BCI_IV2b_mat/B05T.mat',
        'BCI_IV2b_mat/B06T.mat',
        'BCI_IV2b_mat/B07T.mat',
        'BCI_IV2b_mat/B08T.mat',
        'BCI_IV2b_mat/B09T.mat',
        'BCI_IV2b_mat/B01E.mat',
        'BCI_IV2b_mat/B02E.mat',
        'BCI_IV2b_mat/B03E.mat',
        'BCI_IV2b_mat/B04E.mat',
        'BCI_IV2b_mat/B05E.mat',
        'BCI_IV2b_mat/B06E.mat',
        'BCI_IV2b_mat/B07E.mat',
        'BCI_IV2b_mat/B08E.mat',
        'BCI_IV2b_mat/B09E.mat']

pathtest = ["BCI_IV2b_mat/B01T.mat"]

F = None
L = None

Maxlength = 0
Sum = 0
tot = 0

Features_all = None
Labels_all = None
valid_len = []

# *load data
for p in path:
    matfile = sio.loadmat(p)
    if "E" in p:
        items = 2
    else:
        items = 3
    # print(type(matfile["data"][0,0]["gender"]))

    for idx in range(items):
        data = matfile["data"][0,idx]
        # print(matfile["data"][0,idx]["X"][0,0].shape)
        X = data["X"][0,0]
        trials = data["trial"][0,0]
        y = data["y"][0,0]
        Features = []

        for i in range(len(trials)-1):
            feature = X[trials[i][0]:trials[i+1][0]].copy() #* 加上.copy()是因为让feature称为一个新的对象，而不是X的一个视图
            valid_len.append(feature.shape[0])
            feature.resize(18633, 6)
            feature = feature.transpose(1, 0)
            Features.append(feature)
        
        feature = X[trials[-1][0]:].copy()
        valid_len.append(feature.shape[0])
        feature.resize(18633, 6)
        feature = feature.transpose(1, 0)
        Features.append(feature)

        Features = np.stack(Features)

        Label = data["y"][0,0]

        Features_all = Features if Features_all is None else np.concatenate((Features_all, Features), axis=0)
        Labels_all = Label if Labels_all is None else np.concatenate((Labels_all, Label), axis=0)

        # print(Features.shape)
        # print(Label.shape)

    print("finished in ", p)

valid_len = np.array(valid_len)
        
print(Features_all.shape)
print(Labels_all.shape)
print(valid_len.shape)

# np.save("dataset2b/Features.npy", Features_all)
# np.save("dataset2b/Labels.npy", Labels_all)
np.save("dataset2b/valid_len.npy", valid_len)

    



