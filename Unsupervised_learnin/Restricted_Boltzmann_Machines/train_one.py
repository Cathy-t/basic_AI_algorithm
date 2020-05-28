#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/5/15 19:59
# @Author  : Cathy 
# @FileName: train_one.py

from RBM import RBM
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt

import math
import numpy as np

#This is an unsupervised learning algorithm. So let us try training on one particular number.But first
# we need to seperate the data.

DOWNLOAD = False
#Loading MNIST dataset
train_dataset = datasets.MNIST(root='../../data/',
                               train=True,
                               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                               download=DOWNLOAD)
# Need to convert th data into binary variables
train_dataset.data = (train_dataset.train_data.type(torch.FloatTensor)/255).bernoulli()

batch_size = 32

# 加载数据
number = 5 #A number between 0 and 10.

particular_mnist = []

limit = len(train_dataset)
# limit = 60000
for i in range(limit):
    if(train_dataset.targets[i] == number):
        particular_mnist.append(train_dataset.data[i].numpy())
print(len(particular_mnist))  # 5421

tensor_x = torch.stack([torch.Tensor(i) for i in particular_mnist]).type(torch.FloatTensor)
tensor_y = torch.stack([torch.Tensor(number) for i in range(len(particular_mnist))]).type(torch.FloatTensor)
mnist_particular_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
mnist_particular_dataloader = torch.utils.data.DataLoader(mnist_particular_dataset, batch_size=batch_size, drop_last=True, num_workers=0)

visible_units = 28*28
hidden_units = 500
k = 3
learning_rate = 0.01
learning_rate_decay = False
xavier_init = True
increase_to_cd_k = False
use_gpu = False

rbm_mnist = RBM(visible_units, hidden_units, k, learning_rate, learning_rate_decay, xavier_init,
                increase_to_cd_k, use_gpu)

# Epochs = [10, 15, 20, 25, 30, 35]
Epochs = [30]

for epochs in Epochs:
    print('-----------------epoch:{}----------------'.format(epochs))

    rbm_mnist.train(mnist_particular_dataloader, epochs)

    # This shows the weights for each of the 64 hidden neurons and give an idea how each neuron is activated.

    learned_weights = rbm_mnist.W.transpose(0,1).numpy()
    plt.show()
    fig = plt.figure(3, figsize=(10, 10))
    for i in range(25):
        sub = fig.add_subplot(5, 5, i+1)
        sub.imshow(learned_weights[i, :].reshape((28, 28)), cmap=plt.cm.gray)
    plt.title(epochs)
    plt.show()

    #Lets try reconstructing a random number from this model which has learned 5
    idx = 7
    img = train_dataset.train_data[idx]
    reconstructed_img = img.view(-1).type(torch.FloatTensor)

    # _ , reconstructed_img1 = rbm_mnist.to_hidden(reconstructed_img)
    # _ , reconstructed_img2 = rbm_mnist.to_visible(reconstructed_img)

    _, reconstructed_img3 = rbm_mnist.reconstruct(reconstructed_img, 1)

    reconstructed_img = reconstructed_img3.view((28, 28))
    print("The original number: {}".format(train_dataset.train_labels[idx]))
    plt.imshow(img, cmap='gray')
    plt.show()
    print("The reconstructed image")
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title(str(epochs) + 'img' + str(i + 1))
    plt.show()

torch.save(rbm_mnist, 'RBM_one.pth')