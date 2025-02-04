from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=16, kernel_size=7, stride=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.seq3 = nn.Sequential(
            nn.Conv2d(32, 64, 6, 1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.seq4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.seq5 = nn.Sequential(
            nn.Linear(15488, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 3),
        )
    
    def forward(self, x):

        x =  self.seq4(
                self.seq3(
                    self.seq2(
                        self.seq1(x)
                    )
                )
            )
        x = x.view(x.size(0), -1)
        x = self.seq5(x)
        return x
