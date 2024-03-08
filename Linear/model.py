from layer import LinearLayer
import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LinearLayer(4, 1)

    def forward(self, x):
        return self.linear(x)