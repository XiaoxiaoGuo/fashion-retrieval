import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ResNetBatch(nn.Module):
    def __init__(self, resnet):
        super(ResNetBatch, self).__init__()
        self.resnet = resnet

    def forward(self, x, att_size=14):
        # size of x: nimages x nChannel x dim x dim

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2)
        # att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).permute(0, 2, 3, 1)

        return fc, att

