from torch import nn
import torch
from torch_backend import *
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
import skeleton
from collections import OrderedDict
torch.autograd.set_detect_anomaly(True)

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_depth, depth, batch_norm, downsample=None, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.depth = depth
        self.conv1 = nn.Conv2d(in_depth, self.depth, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = batch_norm(self.depth)  
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.depth, self.depth, kernel_size=3, padding=padding, bias=False)
        self.bn2 = batch_norm(self.depth) 
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
    def __init__(self, block, layers, batch_norm, n_channels=3, initial_depth=64, n_classes=10):
        super(ResNet, self).__init__()
        self.depth = initial_depth
        self.initial_depth = initial_depth
        self.dilation = 1
        self.batch_norm = batch_norm
        self.conv1 = nn.Conv2d(n_channels, self.depth, kernel_size=7, stride= 2, padding=3, bias=False)
        self.bn1 = batch_norm(self.depth)  #BN(self.depth)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.initial_depth, layers[0])
        self.layer2 = self._make_layer(block, 2*self.initial_depth, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4*self.initial_depth, layers[2], stride=2)
        last_depth = 4*self.initial_depth
        self.layer4 = False
        if len(layers) == 4:
            self.layer4 = self._make_layer(block, 8*self.initial_depth, layers[3], stride=2)
            last_depth = 8*self.initial_depth
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_depth * block.expansion, n_classes)
        
    def _make_layer(self, block, depth, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.depth != depth * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.depth, depth * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.batch_norm(depth * block.expansion),
            )

        layers = []
        layers.append(block(self.depth, depth, self.batch_norm, downsample, stride))
        self.depth = depth * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.depth, depth, self.batch_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if self.layer4:
            x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def conv_bn_skeleton(channels_in, channels_out, batch_norm, kernel_size=3, stride=1, padding=1, groups=1, bn=True, activation=True):
    op = [
            torch.nn.Conv2d(channels_in, channels_out,
                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
    ]
    if bn:
        op.append(torch.nn.BatchNorm2d(channels_out))
    if activation:
        op.append(torch.nn.ReLU(inplace=True))
    return torch.nn.Sequential(*op)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def build_network(num_class=10):
    return torch.nn.Sequential(
        conv_bn_skeleton(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn_skeleton(64, 128, kernel_size=5, stride=2, padding=2),
        # torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn_skeleton(128, 128),
            conv_bn_skeleton(128, 128),
        )),

        conv_bn_skeleton(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),

        Residual(torch.nn.Sequential(
            conv_bn_skeleton(256, 256),
            conv_bn_skeleton(256, 256),
        )),

        conv_bn_skeleton(256, 128, kernel_size=3, stride=1, padding=0),

        torch.nn.AdaptiveMaxPool2d((1, 1)),
        skeleton.nn.Flatten(),
        torch.nn.Linear(128, num_class, bias=False),
        skeleton.nn.Mul(0.2)
    )



# ### Network definition


# def conv_bn_dpage(c_in, c_out):
#     return {
#         'conv': nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False), 
#         'bn': BatchNorm(c_out), 
#         'relu': nn.ReLU(True)
#     }

# def residual(c):
#     return {
#         'in': Identity(),
#         'res1': conv_bn_dpage(c, c),
#         'res2': conv_bn_dpage(c, c),
#         'add': (Add(), ['in', 'res2/relu']),
#     }

# def net(channels=None, weight=0.125, pool=nn.MaxPool2d(2), extra_layers=(), res_layers=('layer1', 'layer3')):
#     channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
#     n = {
#         'input': (None, []),
#         'prep': conv_bn_dpage(3, channels['prep']),
#         'layer1': dict(conv_bn_dpage(channels['prep'], channels['layer1']), pool=pool),
#         'layer2': dict(conv_bn_dpage(channels['layer1'], channels['layer2']), pool=pool),
#         'layer3': dict(conv_bn_dpage(channels['layer2'], channels['layer3']), pool=pool),
#         'pool': nn.MaxPool2d(4),
#         'flatten': Flatten(),
#         'linear': nn.Linear(channels['layer3'], 10, bias=False),
#         'logits': Mul(weight),
#     }
#     for layer in res_layers:
#         n[layer]['residual'] = residual(channels[layer])
#     for layer in extra_layers:
#         n[layer]['extra'] = conv_bn_dpage(channels[layer], channels[layer])       
#     return n

class conv_bn_self(nn.Module):
    def __init__(self, c_in, c_out, batch_norm, kernel_size=3, activation=True):
        super(conv_bn_self, self).__init__()
        
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn = batch_norm(c_out) #nn.BatchNorm2d(self.depth)
        self.activation=activation
        if self.activation:
            self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, c, batch_norm, downsample=None):
        super(Residual, self).__init__()
        self.conv = conv_bn_self(c, c, batch_norm=batch_norm)
        self.conv2 = conv_bn_self(c, c, batch_norm=batch_norm)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.conv2(out)
    
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity

        return out 

class Network(nn.Module):
    def __init__(self, batch_norm, weight=0.125, pool=nn.MaxPool2d(2)):
        super(Network, self).__init__()
        channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
        self.prep = conv_bn_self(3, channels['prep'], batch_norm=batch_norm)
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv_bn', conv_bn_self(channels['prep'], channels['layer1'], batch_norm=batch_norm)),
            ('pool', pool),
            ('residual', Residual(channels['layer1'], batch_norm=batch_norm))
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv_bn', conv_bn_self(channels['layer1'], channels['layer2'], batch_norm=batch_norm)),
            ('pool', pool)
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
            ('conv_bn', conv_bn_self(channels['layer2'], channels['layer3'], batch_norm=batch_norm)),
            ('pool', pool),
            ('residual', Residual(channels['layer3'], batch_norm=batch_norm))
        ]))
        self.pool = nn.MaxPool2d(4)
        self.flatten = Flatten()
        self.linear =  nn.Linear(channels['layer3'], 10, bias=False)
        self.mul = Mul(weight)

    def forward(self, x):
        x = self.prep(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.mul(x)

        return x
