import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            skipConnection(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, stride=2),
            
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            skipConnection(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(),
            ),
            nn.MaxPool2d(2, stride=2),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            skipConnection(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, stride=2),

            # conv4
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            skipConnection(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, stride=2),

            # conv5
            nn.BatchNorm2d(512),
            skipConnection(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(),
                skipConnection(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
            ),
            nn.MaxPool2d(2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32768, 4096),
            nn.ReLU(),
            nn.Dropout(),
            skipConnection(
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(),
            ),
            nn.Linear(4096, 3)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    

class skipConnection(nn.Module):
    def __init__(self, *args, downsample=None):
        """
        block.__init__(self, net, downsample)

        Аргументы:
        - net - та часть, сети, которая между skip-connection
        - downsample - преобразования к X, чтобы он совпал.
        """

        super().__init__()

        self.net = nn.Sequential(*args)
        if downsample is None:
            downsample = nn.Identity()

        self.downsample = downsample

    def forward(self, X):
        identity = self.downsample(X)

        # F(x) + x
        return self.net(X) + identity

