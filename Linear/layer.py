import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class LinearLayer(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = Parameter(torch.empty(input_size, output_size))
        self.bias = Parameter(torch.empty(output_size))
        
    def forward(self,input):
        return torch.mm(input, self.weight) + self.bias