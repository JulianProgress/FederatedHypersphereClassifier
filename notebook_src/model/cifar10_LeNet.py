import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseNet


class CIFAR10_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        # self.conv1 = nn.Conv2d(3, 16, 5, bias=False, padding=2)
        # self.bn1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        # self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        # self.bn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(32 * 7 * 7, self.rep_dim, bias=False)

        self.conv1 = nn.Conv2d(3, 32, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class CIFAR10_FE(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        # self.conv1 = nn.Conv2d(3, 16, 5, bias=False, padding=2)
        # self.bn1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        # self.conv2 = nn.Conv2d(16, 32, 5, bias=False, padding=2)
        # self.bn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(32 * 7 * 7, self.rep_dim, bias=False)

        self.conv1 = nn.Conv2d(3, 96, 5, bias=False, padding=2)
        # self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(96, 80, 5, bias=False, padding=2)
        # self.bn2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(80, 96, 5, bias=False, padding=2)
        # self.bn3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.conv4 = nn.Conv2d(96, 64, 5, bias=False, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(x))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(x))
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

