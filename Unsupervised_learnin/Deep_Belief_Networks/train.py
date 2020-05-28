#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/4/19 19:06
# @Author  : Cathy 
# @FileName: train.py

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

train_dataset.data = (train_dataset.train_data.type(torch.FloatTensor)/255).bernoulli()

#Lets us visualize a number from the data set
idx = 5
img = train_dataset.train_data[idx]
print("The number shown is the number: {}".format(train_dataset.train_labels[idx]))
plt.imshow(img, cmap='gray')
plt.show()

# I have have set these hyper parameters although you can experiment with them to find better hyperparameters.
dbn_mnist = DBN(visible_units=28*28,
                hidden_units=[23*23, 18*18],
                k=5,
                learning_rate=0.01,
                learning_rate_decay=True,
                xavier_init=True,
                increase_to_cd_k=False,
                use_gpu=False)

num_epochs = 1
batch_size = 10

dbn_mnist.train_static(train_dataset.train_data,train_dataset.train_labels, num_epochs, batch_size)

# visualising layer 1
learned_weights = dbn_mnist.rbm_layers[0].W.transpose(0, 1).numpy()
plt.show()
fig = plt.figure(3, figsize=(10,10))
for i in range(25):
    sub = fig.add_subplot(5, 5, i+1)
    sub.imshow(learned_weights[i,:].reshape((28,28)), cmap=plt.cm.gray)
plt.title('visualising layer 1')
plt.show()

# visualising layer 2
learned_weights = dbn_mnist.rbm_layers[1].W.transpose(0,1).numpy()
plt.show()
fig = plt.figure(3, figsize=(10,10))
for i in range(25):
    sub = fig.add_subplot(5, 5, i+1)
    sub.imshow(learned_weights[i,:].reshape((23,23)), cmap=plt.cm.gray)
plt.title('visualising layer 2')
plt.show()
