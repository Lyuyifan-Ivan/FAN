import torch.nn as nn
import torch


class SpatialAttention(nn.Module):
    def __init__(self, kernelSize=7):
        '''空间注意力'''
        super(SpatialAttention, self).__init__()
        padding = 3
        self.conv1 = nn.Conv2d(2, 1, kernelSize, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):   # x 的输入格式是：[batch_size, C, H, W]
        avgOut = torch.mean(x, dim=1, keepdim=True)
        maxOut, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgOut, maxOut], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):   # x 的输入格式是：[batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
