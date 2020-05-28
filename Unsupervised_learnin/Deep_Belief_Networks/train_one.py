#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/5/15 21:11
# @Author  : Cathy 
# @FileName: train_one.py

from DBN import DBN
import torch
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader

import matplotlib
import matplotlib.pyplot as plt

import math
import numpy as np


DOWNLOAD = False
#Loading MNIST dataset
train_dataset = datasets.MNIST(root='../../data/',
                               train=True,
                               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                               download=DOWNLOAD)
# Need to convert th data into binary variables
train_dataset.data = (train_dataset.train_data.type(torch.FloatTensor)/255).bernoulli()

batch_size = 10

# 加载数据
number = 5 #A number between 0 and 10.

particular_mnist = []

limit = len(train_dataset)
# limit = 60000
for i in range(limit):
    if(train_dataset.targets[i] == number):
        particular_mnist.append(train_dataset.data[i].numpy())
print(len(particular_mnist))  # 5421

train_data = torch.stack([torch.Tensor(i) for i in particular_mnist])
train_label = torch.stack([torch.Tensor(number) for i in range(len(particular_mnist))])

dbn_mnist = DBN(visible_units=28*28,
                hidden_units=[23*23, 18*18],
                k=5,
                learning_rate=0.01,
                learning_rate_decay=True,
                xavier_init=True,
                increase_to_cd_k=False,
                use_gpu=False)


Epoch = [10, 15, 20, 25, 30, 35]
for epoch in Epoch:
    print('-----------------------{}------------'.format(epoch))
    dbn_mnist.train_static(train_data, train_label, epoch, batch_size)

    idx = 3
    img = train_dataset.train_data[idx]
    reconstructed_img = img.view(1, -1).type(torch.FloatTensor)

    _, reconstructed_img = dbn_mnist.reconstruct(reconstructed_img)

    reconstructed_img = reconstructed_img.view((28,28))
    print("The original number: {}".format(train_dataset.train_labels[idx]))
    plt.imshow(img, cmap='gray')
    plt.show()
    print("The reconstructed image")
    plt.title(epoch)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.show()


