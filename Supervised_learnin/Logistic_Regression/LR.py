#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/4/15 17:01
# @Author  : Cathy 
# @FileName: LR.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# 超参数设置 Hyper-parameters
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.005

# 未下载数据，使用Ｔｒｕｅ表示下载数据
DOWNLOAD = False

"""MINIST数据加载"""
# train (bool, optional): If True, creates dataset from ``training.pt``,otherwise from ``test.pt``
train_dataset = datasets.MNIST(root='../../data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=DOWNLOAD)

test_dataset = datasets.MNIST(root='../../data/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=DOWNLOAD)

# 加载数据
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


"""构建模型"""


# 定义逻辑回归模型
class LR(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=True)

    def forward(self, x):
        x = F.sigmoid(self.linear(x))
        return x


LR_model = LR(input_size, num_classes)
print(LR_model)

# 定义逻辑回归的损失函数，采用nn.CrossEntropyLoss(),nn.CrossEntropyLoss()内部集成了softmax函数
criterion = nn.CrossEntropyLoss(reduction='mean')

# 定义optimizer
optimizer = torch.optim.SGD(LR_model.parameters(), lr=learning_rate)

# 训练模型
losses = []
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将图像序列转换至大小为 (batch_size, input_size),应为（100，,784）
        images = images.reshape(-1, 28 * 28)

        # forward
        y_pred = LR_model(images)
        # print(y_pred.size())
        # print(labels.size())
        # exit()
        loss = criterion(y_pred, labels)
        losses.append(loss)

        # backward()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % 100 == 0):
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                     loss.item()))

"""模型测试"""
# 在测试阶段，为了运行内存效率，就不需要计算梯度了
# PyTorch 默认每一次前向传播都会计算梯度
with torch.no_grad():
    correct = 0
    total = 0
    confusion_matrix = torch.zeros((10, 10))
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = LR_model(images)
        # torch.max的输出：out (tuple, optional) – the result tuple of two output tensors (max, max_indices)
        _, predicted = torch.max(outputs.data, 1)
        # print(max.data)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        for i in range(len(predicted)):
            p = predicted[i]
            l = labels[i]
            confusion_matrix[l][p] += 1

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


# 保存模型
torch.save(LR_model.state_dict(), './LR_sigmoid_model.pth')

if True:
    # 画图
    plt.plot(losses)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # 画混淆矩阵
    import numpy as np
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    confusion_matrix = confusion_matrix.numpy()
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2.
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(10)] for i in range(10)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(confusion_matrix[i, j]))  # 显示对应的数字

    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()






