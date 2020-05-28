#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/4/15 17:10
# @Author  : Cathy 
# @FileName: MLP.py

import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 超参数设置 Hyper-parameters
batch_size = 100
# input_size（输入层大小）、hidden_size（隐藏层大小）、output_size（输出层大小）
input_size, hidden_size, output_size = 784, 100, 10
# 设置超参数
learning_rate = 0.5
num_epochs = 5

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

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        在构建模型的时候，在能够使用nn.Sequential的地方尽量使用它，因为这样可以让结构更加的清晰
        """
        super(TwoLayerNet, self).__init__()

        self.twolayernet = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        y_pred = self.twolayernet(x)
        return y_pred


MLP_model = TwoLayerNet(input_size, hidden_size, output_size)

# 定义损失函数
criterion = nn.CrossEntropyLoss(reduction='mean')

# 进行优化（进行梯度下降和更新参数）
optimizer = torch.optim.SGD(MLP_model.parameters(), lr=learning_rate)

# 开始训练
losses = []
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将图像序列转换至大小为 (batch_size, input_size),应为（100，,784）
        images = images.reshape(-1, 28 * 28)

        # forward
        y_pred = MLP_model(images)
        # print(y_pred.size())
        # print(labels.size())
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
        outputs = MLP_model(images)
        # torch.max的输出：out (tuple, optional) – the result tuple of two output tensors (max, max_indices)
        _, predicted = torch.max(outputs.data, 1)
        # print(max.data)
        # print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        for i in range(len(predicted)):
            p = predicted[i]
            l = labels[i]
            confusion_matrix[l][p] += 1

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


# 保存模型
torch.save(MLP_model.state_dict(), './MLP_model.pth')

if True:
    import matplotlib.pyplot as plt
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


