from torch import nn


class skipConnection(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.identity = nn.Identity()
    
    def forward(self, x):
        return self.net(x) + self.identity(x)


class VGG19(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg_layers = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            skipConnection(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            skipConnection(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            skipConnection(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                skipConnection(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(True)
                ),
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            skipConnection(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                skipConnection(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True)
                ),
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            skipConnection(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                skipConnection(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True)
                ),
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.classifier = nn.Sequential(
            # nn.Linear(32768, 8192),
            # nn.ReLU(),
            # nn.Dropout(),
            # skipConnection(
            #     nn.Linear(8192, 8192),
            #     nn.ReLU(),
            #     nn.Dropout()
            # ),
            nn.Linear(8192, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 3),
        )
    
    def forward(self, x):
        x = self.vgg_layers(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

