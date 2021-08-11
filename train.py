from torch import optim
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import res50_cnn
import torch.nn as nn
from torchnet.meter import AverageValueMeter
import torchvision


class Config(object):
    class_num = 10
    save_every = 5
    epochs = 50


def train(opt):
    train_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    error_net = AverageValueMeter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainset = torchvision.datasets.CIFAR10(root='./image', train=True,
                                            download=True, transform=train_transform)
    # dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=False)
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                             shuffle=True, num_workers=0)
    net = res50_cnn(opt).to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    epochs = tqdm(range(opt.epochs))
    for epoch in epochs:
        for ii, data in enumerate(dataloader):
            x, label = data
            x.to(device)
            label.to(device)
            optimizer.zero_grad()
            output = net(x)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            error_net.add(loss.data.item())

        if (epoch + 1) % opt.save_every != 0:
            torch.save(net.state_dict(), 'checkpoints/weights.pth')

        epochs.set_postfix('Average Loss: %s' % error_net.mean)
        error_net.reset()


opt = Config()
train(opt)
