from torch import nn
import torch


class block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        resid = x
        out = self.seq(x)
        if self.downsample is not None:
            resid = self.downsample(x)
        out += resid
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        self.in_channels = 64
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.relu = nn.ReLU()
        self.layer1 = self.create_layers(64, 3)
        self.layer2 = self.create_layers(128, 4, stride=2)
        self.layer3 = self.create_layers(256, 6, stride=2)
        self.layer4 = self.create_layers(512, 3, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.fc = nn.Linear(4608, 3)

    def create_layers(self, out_channels, layers_count, stride=1):
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            downsample = None

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        layers.extend([block(self.in_channels, out_channels) for _ in range(layers_count - 1)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.seq1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x