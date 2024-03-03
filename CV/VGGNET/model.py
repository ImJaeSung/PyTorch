import torch.nn as nn
import math

class VGGnet(nn.Module):
    def __init__(self, n_layers, n_classes = 10, batch_norm = False):
        super(VGGnet, self).__init__()
        self.n_layers = n_layers
        self.layers = self.make_layers(self.channels(), batch_norm = batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5), #  dropout regularization for the first two FC layer
            nn.Linear(512, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace = True),
            nn.Linear(512, n_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean = 0, std = math.sqrt(0.01))
                nn.init.constant_(m.bias, 0)

    def channels(self):
        channels = {
            11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            }
        return channels[self.n_layers]

    def make_layers(self, channels, batch_norm = False):
        layers = []
        in_channels = 3
        for channel in channels:
            if channel == 'M':
                layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)] # make halves
            else:
                conv2d = nn.Conv2d(in_channels, channel, kernel_size = 3, padding = 1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(channel), nn.ReLU(inplace = True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace = True)]
                in_channels = channel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x