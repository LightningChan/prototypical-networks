"""
Backbone of ResNet 101
"""
import torch.nn as nn


def build_ResNet101():
    return ResNet(layers=[3, 4, 23, 3])


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.block = Bottleneck

        self.C1 == nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64, eps=0.001),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.C2 = self._make_layer(self.block, 64, layers[0])
        self.C3 = self._make_layer(self.block, 128, layers[1], stride=2)
        self.C4 = self._make_layer(self.block, 256, layers[2], stride=2)
        self.C5 = self._make_layer(self.block, 512, layers[3], stride=2)

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * self.block.expansion, eps=0.001)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]
