#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/4/15 17:11
# @Author  : Cathy 
# @FileName: LeNet5.py

import torch
import torch.nn as nn
import torch.nn.functional as F

"""使用torchvision 来加载数据集"""

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

# hyper-parameters
# 【0.0001\0.001\0.005\0.01\0.1\0.5\1】
learning_rate = 0.001
num_epoches = 5
use_gpu = torch.cuda.is_available()
# mini-batch
batch_size = 100
in_dim = 1
n_class = 10


# 未下载数据，使用Ｔｒｕｅ表示下载数据
DOWNLOAD = False


# 获得数据
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


# 构建模型
class LeNet5(nn.Module):
    def __init__(self, in_dim, n_class):
        super(LeNet5, self).__init__()

        # 第一层：卷积层的输入是 1 channel，　输出是 6 channel，kenel_size=(5, 5)
        # 当 kenel_size=(5, 5)　时，要使padding满足'same', 则 padding = (kenel_size - 1) / 2
        self.conv1 = nn.Conv2d(in_dim,  6, (5, 5), 1, padding=2)

        # 第二层：卷积层，　输入　6 channel ，输出　１6 channel，　kennel_size=(5, 5)
        # padding=0,默认，此时进行 valid 操作
        self.conv2 = nn.Conv2d(6, 16, (5, 5))

        # 第三层：　全连接层（线性表示）
        # 此时的全连接里面有４００(5*5*16)个节点,其中每个节点中有１２０个神经元
        self.fc1 = nn.Linear(5*5*16, 120)

        # 第四层：全连接层
        self.fc2 = nn.Linear(120, 84)

        # 第五层：　输出层
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        # subsampling 1 process
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)   # input & kennel_size

        # subsampling 2 process
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)

        # -1 的话，意味着最后的相乘维数  [使４００维的 变成　行　自动补齐]
        # 使用num_flat_features函数计算张量x的总特征量（把每个数字都看出是一个特征，即特征总量），比如x是4*2*2的张量，那么它的特征总量就是16。
        x = x.view(-1, self.num_flat_features(x))

        # full connect1
        x = F.relu(self.fc1(x))

        # full connect2
        x = F.relu(self.fc2(x))

        # full connect3
        x = self.fc3(x)

        return x

    # 16 channel 卷积层，转全连接层的处理
    # num_flat_features函数是把经过两次池化后的16x5x5的矩阵组降维成二维，便于view函数处理，而其中用乘法也是为了不丢失每一层相关的特性
    def num_flat_features(self, x):
        # 得到　channel * iW * iH 的值
        # # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
        size = x.size()[1:]
        # print(size)

        num_features = 1

        for s in size:
            num_features *= s

        return num_features


leNet = LeNet5(in_dim, n_class)
print(leNet)   # 打印出了模型每一层的信息

"""有了模型和数据集后，开始进行训练测试"""
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimazer = optim.Adam(leNet.parameters(), lr=learning_rate)
tt = 0

# 开始训练
losses = []
accs = []
for epoch in range(num_epoches):

    print('epoch {}'.format(epoch + 1))
    print('*' * 10)

    running_loss = 0.0
    running_acc = 0.0

    for i, data in enumerate(train_loader, 1):
        tt += 1
        img, label = data

        # img = Variable(img).squeeze()
        # label = Variable(label).squeeze()

        # 前向传播
        out = leNet(img)

        loss = criterion(out, label)

        running_loss += loss.item() * label.size(0)   # label.size(0) = 100
        losses.append(loss)

        _, pred = torch.max(out, 1)     # torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        # 或写为：　pred = torch.max(out, 1)[1]     https://blog.csdn.net/Z_lbj/article/details/79766690

        num_correct = (pred == label).float().mean()

        running_acc += num_correct.item()
        accs.append(running_acc)

        # 反向传播
        optimazer.zero_grad()
        loss.backward()
        optimazer.step()

        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc:{:.6f}'.format(epoch+1, num_epoches,
                                                            running_loss/(batch_size*i), running_acc/(batch_size*i)))


"""模型测试"""
# 在测试阶段，为了运行内存效率，就不需要计算梯度了
# PyTorch 默认每一次前向传播都会计算梯度
with torch.no_grad():
    correct = 0
    total = 0
    confusion_matrix = torch.zeros((10, 10))
    for images, labels in test_loader:
        outputs = leNet(images)
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
torch.save(leNet.state_dict(), './leNet_model.pth')

if True:
    import matplotlib.pyplot as plt
    # 画图
    plt.plot(losses)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # plt.plot(accs)
    # plt.ylabel('Accuracy')
    # plt.xlabel('iterations')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()

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

print('Done!')
