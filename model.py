import torch
import torch.nn.functional as F
from torch import nn


class DogvsCatModel(nn.Module):
    def __init__(
        self,
    ):
        super(DogvsCatModel, self).__init__()
        self.mFC0 = nn.Linear(in_features=2560, out_features=512)
        self.mBN0 = nn.BatchNorm1d(512)
        self.mDropout0 = nn.Dropout(0.2)
        self.mFC1 = nn.Linear(in_features=512, out_features=128)
        self.mBN1 = nn.BatchNorm1d(128)
        self.mDropout1 = nn.Dropout(0.2)
        self.mFC2 = nn.Linear(in_features=128, out_features=1)
        self.mSigmoid = nn.Sigmoid()

    def forward(self, x, eval=False):
        x = self.mFC0(x)
        x = self.mBN0(x)
        x = F.relu(x)
        x = self.mDropout0(x)
        x = self.mFC1(x)
        x = self.mBN1(x)
        x = F.relu(x)
        x = self.mDropout1(x)
        x = self.mFC2(x)
        x = self.mSigmoid(x)
        if eval == True:
            x = torch.round(x)
        return x
