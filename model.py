from torch import nn


class DogvsCatModel(nn.Module):
    def __init__(
        self,
    ):
        super(DogvsCatModel, self).__init__()
        self.mConv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=(2, 2),
                stride=1,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.mConv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(2, 2),
                stride=1,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.mConv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(2, 2),
                stride=1,
                padding=(1, 1),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )
        self.mFlat = nn.Flatten()
        self.mFC = nn.Sequential(
            nn.Linear(32768, 512),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.mConv0(x)
        x = self.mConv1(x)
        x = self.mConv2(x)
        x = self.mFlat(x)

        x = self.mFC(x)
        return x
