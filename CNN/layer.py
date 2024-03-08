import torch.nn as nn

class conv_pool_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool_k_size=1):
        super().__init__()

        self.in_c = in_channels
        self.out_c = out_channels
        self.k_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_k_size = pool_k_size

        self.conv = nn.Conv2d(in_channels=self.in_c, 
                              out_channels=self.out_c, 
                              kernel_size=self.k_size, 
                              stride=self.stride, 
                              padding=self.padding)
        self.batch_norm = nn.BatchNorm2d(num_features=self.out_c)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(self.pool_k_size)

    def forward(self, input):

        output = self.conv(input)
        output = self.batch_norm(output)
        output = self.relu(output)
        output = self.maxpool(output)

        return output

class fc_layer(nn.Module):
    def __init__(self, feature_map_size, num_class):
        super().__init__()

        self.feature_map_size = feature_map_size
        self.num_class = num_class

        self.fc = nn.Linear(self.feature_map_size, self.num_class)
        
    def forward(self, input):

        output = input.view(input.size(0), -1)
        output = self.fc(output)

        return output