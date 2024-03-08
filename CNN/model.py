import torch.nn as nn
from layer import conv_pool_layer, fc_layer

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_pool_layer1 = conv_pool_layer(in_channels=1, 
                                                out_channels=32, 
                                                kernel_size=3, 
                                                stride=1, 
                                                padding=1,
                                                pool_k_size=2)
        
        self.conv_pool_layer2 = conv_pool_layer(in_channels=32,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                pool_k_size=2)
        
        self.fc_layer = fc_layer(64*7*7, 10)

    def forward(self, input):

        output = self.conv_pool_layer1(input)
        output = self.conv_pool_layer2(output)
        output = self.fc_layer(output)

        return output