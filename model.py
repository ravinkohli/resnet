from torch import nn
import torch
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_depth, depth, downsample=None, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.depth = depth
        self.conv1 = nn.Conv2d(in_depth, self.depth, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.depth)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.depth, self.depth, kernel_size=3, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(self.depth)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
    
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        
class ResNet(nn.Module):
    def __init__(self, block, layers, n_channels=3, initial_depth=64, n_classes=10):
        super(ResNet, self).__init__()
        self.depth = initial_depth
        self.initial_depth = initial_depth
        print(self.depth)
        self.dilation = 1

        self.conv1 = nn.Conv2d(n_channels, self.depth, kernel_size=7, stride= 2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.depth)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.initial_depth, layers[0])
        self.layer2 = self._make_layer(block, 2*self.initial_depth, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4*self.initial_depth, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, n_classes)
        
    def _make_layer(self, block, depth, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.depth != depth * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.depth, depth * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(depth * block.expansion),
            )

        layers = []
        layers.append(block(self.depth, depth, downsample, stride))
        self.depth = depth * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.depth, depth))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

