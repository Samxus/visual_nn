import torch
from torchvision.models import resnet50
import torchvision.transforms
import ssl
import torch.nn as nn

ssl._create_default_https_context = ssl._create_unverified_context


class res50_cnn(nn.Module):
    def __init__(self, opt):
        super(res50_cnn, self).__init__()
        self.opt = opt
        resnet = resnet50(pretrained=True)
        layers = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
                  resnet.layer4, resnet.avgpool]
        self.cnn = nn.Sequential(*layers)
        self.fc = resnet.fc
        self.fc1 = nn.Linear(in_features=1000, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=self.opt.class_num)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 2048)
        x = self.fc(x)
        x = self.fc1(x)
        return self.fc2(x)
