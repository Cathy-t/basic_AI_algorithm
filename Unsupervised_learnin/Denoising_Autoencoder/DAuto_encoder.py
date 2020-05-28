#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/4/15 17:40
# @Author  : Cathy 
# @FileName: DAuto_encoder.py

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from utils import masking_noise

if not os.path.exists('./img'):
    os.mkdir('./img')

DOWNLOAD = False


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3
corrupt = 0.3

img_transform = transforms.Compose([
     transforms.ToTensor(),
     # transforms.Lambda(lambda x: x.repeat(3,1,1)),
     # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
 ])   # 修改的位置

dataset = MNIST('../../data/', transform=img_transform, download=DOWNLOAD)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# model = autoencoder().cuda()
model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

if False:
    losses = []
    for epoch in range(num_epochs):
        # for data in dataloader:
        for i, data in enumerate(dataloader, 1):
            img, _ = data    # torch.Size([128, 1, 28, 28])
            # img = Variable(img).cuda()
            # img = Variable(img)
            inputs_corr = masking_noise(img, corrupt)
            # ===================forward=====================
            output = model(inputs_corr)
            loss = criterion(output, img)
            losses.append(loss)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, loss.item()))
        if epoch % 10 == 0:
            # pic = to_img(output.cpu().data)
            pic = to_img(output.data)
            save_image(pic, './img/image_{}.png'.format(epoch))

    torch.save(model.state_dict(), './conv_Dautoencoder.pth')

    if True:
        import matplotlib.pyplot as plt
        # 画图
        plt.plot(losses)
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
else:
    model.load_state_dict(torch.load('./conv_Dautoencoder.pth'))

    test_dataset = MNIST('../../data/', train=False, transform=img_transform, download=DOWNLOAD)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, data in enumerate(test_dataloader, 1):
        img, _ = data  # torch.Size([128, 1, 28, 28])
        # img = Variable(img).cuda()
        # img = Variable(img)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================log========================
    print('loss:{:.4f}'.format(loss.item()))

    # pic = to_img(output.data)
    # save_image(pic, './img/test_image.png')