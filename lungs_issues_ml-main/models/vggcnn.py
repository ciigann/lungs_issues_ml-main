from torch import nn


class block(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.net = nn.Sequential(*layers)
        self.identity = nn.Identity()
    
    def forward(self, x):
        return self.net(x) + self.identity(x)


class VGGCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg_layers = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            block(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            block(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(True)
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            block(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                block(
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
            block(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                block(
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
            block(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                block(
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(True)
                ),
            ),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(32, 3)
        )
        self.softmax = nn.LogSoftmax()
    
    def forward(self, x):
        x = self.vgg_layers(x)
        x = self.cnn_layers(x)
        x = self.dense_layers(x)
        return self.softmax(x)