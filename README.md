# GloveCNN Lab for AIGroup (806) in Pytorch

**This project can be the first lab for AIGroup to learn how to build a CNN project.**

The main contrbutor to the project is [Chandery(myself)](https://github.com/Chandery)

Here is the introduction to this project used for lab.

## Dataset

The dataset we use in the lab is the **dataset2b** from [BCI Competition IV](https://www.bbci.de/competition/iv/). There is already official introduction to the dataset in the project file.

### How to get the dataset?

Because the dataset is too big to push up to the reporsitory, we have uploaded it to the Release Version. Readers can download it with the whole project files in the **Releases**.

We have two directories in the dataset package, Raw and Processed.The Raw one is pre-processed by official from `.gdf` to `.mat`. Readers can open it by matlab to see what is it like and process it with python mathod *`scipy.io`.* It is worth noting that Using this method to import `.mat` files, especially when encountering struct type data within `.mat`, can lead to some peculiar issues that require special solutions. For more details, Readers can refer to the reference code provided in this project.

### DataScale

There are about 5700 trials in total in the dataset. The Features shapes like (channels_num, Trial_Length). Among them, the number of channels is 6; and the average of the trial length is 3229 as well as the max one is 18633. The reference method for processing the data is to pad all trials with 0 to a length of 18633, which is similar to masking the data.

## Models

Models used in this project include ResNet, EEGNet, ShallowConvNet, DeepConvNet, EEGtrans, etc. Among them the original report of the EEGNet have been provided in the project file. We recommend that readers construct the EEGNet themselves by reading the paper, rather than directly referring to the code provided in this project. It is  worth to pay attention that *EEGtrans* is a model that is pieced together by the contributor. Considering that this is a basic exercise of the constructing code, Readers can ignore it.

We provide the rank of the models symbolize the extent to which it is worth for Readers to implement.

1. EEGNet
2. ShallowConvNet
3. ResNet
4. EEGtrans
5. DeepConvNet

## Good Wishes

It is hoped that through this experiment, Readers can build their first deep learning project code, learn how to handle data, and enhance their proficiency in using Python.
