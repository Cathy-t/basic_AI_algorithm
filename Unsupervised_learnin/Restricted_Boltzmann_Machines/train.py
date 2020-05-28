#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/4/19 18:41
# @Author  : Cathy # If we train on the whole set we expect it to learn to detect edges.
# @FileName: train.py

from RBM import RBM
import torch
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader

import matplotlib
import matplotlib.pyplot as plt

import math
import numpy as np

DOWNLOAD = False
batch_size = 32

visible_units = 28*28
hidden_units = 500
k = 3
learning_rate = 0.01
learning_rate_decay = True
xavier_init = True
increase_to_cd_k = False
use_gpu = False

#Loading MNIST dataset
train_dataset = datasets.MNIST(root='../../data/',
                               train=True,
                               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                               download=DOWNLOAD)

# Need to convert th data into binary variables
train_dataset.data = (train_dataset.train_data.type(torch.FloatTensor)/255).bernoulli()

tensor_x = train_dataset.train_data.type(torch.FloatTensor) # transform to torch tensors
tensor_y = train_dataset.train_labels.type(torch.FloatTensor)
_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
train_loader = torch.utils.data.DataLoader(_dataset,
                    batch_size=batch_size, shuffle=True,drop_last = True)

# 加载数据
# train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

# make model
rbm_mnist = RBM(visible_units, hidden_units, k, learning_rate, learning_rate_decay, xavier_init,
                increase_to_cd_k, use_gpu)

epochs = 50

rbm_mnist.train(train_loader, epochs, batch_size)

learned_weights = rbm_mnist.W.transpose(0, 1).numpy()
plt.show()
fig = plt.figure(3, figsize=(10, 10))
for i in range(25):
    sub = fig.add_subplot(5, 5, i+1)
    sub.imshow(learned_weights[i, :].reshape((28, 28)), cmap=plt.cm.gray)
plt.show()

torch.save(rbm_mnist, 'RBM.pth')


