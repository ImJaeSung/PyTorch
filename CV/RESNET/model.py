import torch
import torch.nn as nn

class Baseblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Baseblock, self).__init__()
        self.residual_function = self._make_residual_function(in_channels, out_channels, stride)
        self.shortcut = self._make_shortcut(in_channels, out_channels, stride)

    def _make_residual_function(self, in_channels, out_channels, stride):
        # expansion = 1
        residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False), # 64
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False), # 64
            nn.BatchNorm2d(out_channels)
        )
        return residual_function

    def _make_shortcut(self, in_channels, out_channels, stride):
        # expansion = 1
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False), # 1x1 convolution
              nn.BatchNorm2d(out_channels)
            )
        else:
            shortcut = nn.Sequential()
        
        return shortcut    
    
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x) # residual network
        x = nn.ReLU(inplace = True)(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Bottleneck, self).__init__()
        self.residual_function = self._make_residual_function(in_channels, out_channels, stride)
        self.shortcut = self._make_shortcut(in_channels, out_channels, stride)

    def _make_residual_function(self, in_channels, out_channels, stride): # f(x) = h(x) - x
        expansion = 4
        
        residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = False), # 64
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False), # 64
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(out_channels, out_channels*expansion, kernel_size = 1, bias = False), # 256
            nn.BatchNorm2d(out_channels*expansion)
        )
        
        return residual_function

    def _make_shortcut(self, in_channels, out_channels, stride): # x
        expansion = 4

        if stride != 1 or in_channels != out_channels*expansion:
            shortcut = nn.Sequential(
              nn.Conv2d(in_channels, out_channels*expansion, kernel_size = 1, stride = stride, bias = False), # 1x1 convolution
              nn.BatchNorm2d(out_channels*expansion) 
            )
        
        else:
            shortcut = nn.Sequential()
        
        return shortcut
    
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x) # residual network
        x = nn.ReLU(inplace = True)(x)
        return x


class ResNet(nn.Module):
    def __init__(self, n_layers, n_classes = 10):
        super(ResNet, self).__init__() 
        self.in_channels = 64
        self.building_block = self._n_layers2buildingblock(n_layers)
        self.layers = self._n_layers2layers(n_layers)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        ) # 64, 32, 32

        self.conv2_x = self._make_block(self.building_block, self.layers[0], 64, 1) 
        self.conv3_x = self._make_block(self.building_block, self.layers[1], 128, 2)
        self.conv4_x = self._make_block(self.building_block, self.layers[2], 256, 2)
        self.conv5_x = self._make_block(self.building_block, self.layers[3], 512, 2)
        
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        if self.building_block == Bottleneck:
            expansion = 4
        else:
            expansion = 1
        
        self.fc = nn.Linear(512*expansion, n_classes)

    def _make_block(self, building_block, n_layers, out_channels, stride):
        strides = [stride] + [1]*(n_layers - 1) # [stride, 1, ..., 1]
        blocks = []

        if building_block == Bottleneck:
            expansion = 4
        else:
            expansion = 1
        
        for stride in strides:
            blocks.append(building_block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels*expansion
        
        return nn.Sequential(*blocks)
    
    def _n_layers2buildingblock(self, n_layers):
        if n_layers >= 50:
            return Bottleneck
        else:
            return Baseblock


    def _n_layers2layers(self, n_layers):
        dictionary = {18 : [2, 2, 2, 2], 
                    34 : [3, 4, 6, 3],
                    50 : [3, 4, 6, 3],
                    101 : [3, 4, 23, 3],
                    152 : [3, 8, 36, 3]}
        return dictionary[n_layers]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pooling(x)
        x = self.fc(x.view(x.size(0), -1))

        return x


model = ResNet(50)
model.building_block == Bottleneck