import os
import time

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image

from model import StackedDAutoEncoder

if not os.path.exists('./imgs'):
    os.mkdir('./imgs')

def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 1000
batch_size = 128

img_transform = transforms.Compose([
    #transforms.RandomRotation(360),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    transforms.ToTensor(),
])

# dataset = CIFAR10('../../data/cifar10/', transform=img_transform, download=False)
dataset = MNIST('../../data/', transform=img_transform, download=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = StackedDAutoEncoder()
criterion = nn.CrossEntropyLoss()

if False:
    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            # Test the quality of our features with a randomly initialzed linear classifier.
            classifier = nn.Linear(1024, 10)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

        model.train()
        total_time = time.time()
        correct = 0
        for i, data in enumerate(dataloader):
            img, target = data
            # target = Variable(target)  # batch_size
            # img = Variable(img)
            features = model(img).detach()  # torch.Size([128, 1, 28, 28])
            a = features.view(features.size(0), -1)
            prediction = classifier(features.view(features.size(0), -1))
            loss = criterion(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = prediction.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        total_time = time.time() - total_time

        model.eval()
        img, _ = data
        img = Variable(img)
        features, x_reconstructed = model(img)
        reconstruction_loss = torch.mean((x_reconstructed.data - img.data)**2)

        if epoch % 10 == 0:
            print("Saving epoch {}".format(epoch))
            orig = to_img(img.cpu().data)
            save_image(orig, './imgs/orig_{}.png'.format(epoch))
            pic = to_img(x_reconstructed.cpu().data)
            save_image(pic, './imgs/reconstruction_{}.png'.format(epoch))

        print("Epoch {} complete\tTime: {:.4f}s\t\tLoss: {:.4f}".format(epoch, total_time, reconstruction_loss))
        print("Feature Statistics\tMean: {:.4f}\t\tMax: {:.4f}\t\tSparsity: {:.4f}%".format(
            torch.mean(features.data), torch.max(features.data), torch.sum(features.data == 0.0)*100 / features.data.numel())
        )
        print("Linear classifier performance: {}/{} = {:.2f}%".format(correct, len(dataloader)*batch_size, 100*float(correct) / (len(dataloader)*batch_size)))
        print("="*80)

    torch.save(model.state_dict(), './CDAE.pth')

else:
    model.load_state_dict(torch.load('./mnist_CDAE.pth'))

    test_dataset = MNIST('../../data/', train=False, transform=img_transform, download=False)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, data in enumerate(test_dataloader):
        model.eval()
        img, _ = data
        img = Variable(img)
        features, x_reconstructed = model(img)
        reconstruction_loss = torch.mean((x_reconstructed.data - img.data) ** 2)

    print(reconstruction_loss)
    # orig = to_img(img.cpu().data)
    # pic = to_img(x_reconstructed.cpu().data)
    # save_image(pic, './imgs/reconstruction_test.png')
