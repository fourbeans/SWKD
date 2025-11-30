import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.activation_based.auto_cuda import neuron_kernel as ac_neuron_kernel
# from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime

from torch.utils.data.dataloader import default_collate
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)



# ==============modified ANN===============
class CIFAR10Net_ANN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                conv.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(nn.BatchNorm2d(channels))
                conv.append(nn.ReLU())

            conv.append(nn.AvgPool2d(2, 2))

        self.conv = nn.Sequential(*conv)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, channels * 8 * 8 // 4),
            nn.ReLU(),
            nn.Linear(channels * 8 * 8 // 4, 10),
        )

    def forward(self, x: torch.Tensor):
        features = []
        idx = 0
        for layer in self.conv:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x.clone())
        for layer in self.fc:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x.clone())
        # return self.fc(self.conv(x))
        return x, features

#===========LIF==================
class MergeTemporalDim(nn.Module):
    def __init__(self, T, dt=1):
        super().__init__()
        self.T = T
        self.dt = dt

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()


class ExpandTemporalDim(nn.Module):
    def __init__(self, T, dt=1):
        super().__init__()
        self.T = T
        self.dt = dt

    def forward(self, x_seq: torch.Tensor):
        T = int(self.T / self.dt)
        y_shape = [T, int(x_seq.shape[0] / T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


def lif_forward(model, x):
    x = model.expand(x)
    mem = 0
    spikes = []
    for t in range(model.T):
        mem = mem * model.tau + x[t, ...]
        spike = model.act(mem - model.thresh, model.gama)
        mem = (1 - spike) * mem
        spikes.append(spike)
    x = torch.stack(spikes, dim=0)
    x = model.merge(x)
    return x


class LIFSpike(nn.Module):
    def __init__(self, T, thresh=1.0, tau=0.99, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.T = T
        self._forward = lif_forward
        print('use LIFSpike')

    def forward(self, x):
        return self._forward(self, x)

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional

class CIFAR10Net_LIFSpike(nn.Module):
    def __init__(self, channels, T: int, thresh=1.0, tau=0.99, gama=1.0):
        super().__init__()
        self.T = T
        self.temporal_merge_input = MergeTemporalDim(T)
        self.temporal_expand_output = ExpandTemporalDim(T)
        # self.transform1 = transform(channels)
        self.transform2 = Feature_Converter(channels)

        conv = []
        for i in range(2):
            for j in range(3):
                in_channels = 3 if len(conv) == 0 else channels
                conv.append(layer.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm2d(channels))
                conv.append(LIFSpike(T=T, thresh=thresh, tau=tau, gama=gama))
            conv.append(layer.AvgPool2d(2, 2))

        self.conv = nn.Sequential(*conv)

        self.flatten = layer.Flatten()
        self.spike = LIFSpike(T=T, thresh=thresh, tau=tau, gama=gama)

        # 已知输入经过 conv 后变为 [C, 8, 8]，写死输入维度
        self.fc1 = layer.Linear(channels * 8 * 8, (channels * 8 * 8) // 4)
        self.fc2 = layer.Linear((channels * 8 * 8) // 4, 10)

        # self.fc1 = None  # Lazy init
        # self.fc2 = None

    def forward(self, x_seq: torch.Tensor):  # [T, N, C, H, W]
        x = x_seq.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        features = []
        x = self.temporal_merge_input(x)  # [T*N, C, H, W]
        for layer in self.conv:
            x = layer(x) # [T*N, C, H', W']
            if isinstance(layer, LIFSpike):
                # print('layer',x)
                # # 不含transform
                # feat = x.clone().view(self.T, -1, *x.shape[1:])  # [T, N, ...]
                # features.append(feat.mean(0))  # [N, ...]
                # 含
                feat = x.clone().view(self.T, -1, *x.shape[1:])  # [T, N, ...]
                #===新
                total_elements_per_t = feat.shape[1] * feat.shape[2] * feat.shape[3] * feat.shape[4]
                spike_rates_per_t = []  # 存储每个时间步的发放率
                for t in range(self.T):
                    # 取第t个时间步的脉冲特征：[N, C, H, W]
                    feat_t = feat[t]
                    # 统计该时间步中值为1的元素个数（发放数）
                    spike_count_t = torch.sum(feat_t == 1)
                    # 计算发放率（转为float避免整数除法）
                    spike_rate_t = (spike_count_t / total_elements_per_t).item()
                    spike_rates_per_t.append(spike_rate_t)

                # 3. 可选：打印结果（直观查看每个时间步的发放率）
                print(f"各时间步发放率（共{self.T}个时间步）：")
                for t_idx, rate in enumerate(spike_rates_per_t):
                    print(f"时间步{t_idx + 1}/{self.T}：发放率 = {rate:.4f}（{rate * 100:.2f}%）")

                feat1 = self.transform2(feat.mean(0))
                features.append(feat1)  # [N, ...]
                # features.append(x.clone())

        x = self.flatten(x)                   # [T*N, features]

        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = layer.Linear(in_features, in_features // 4).to(x.device)
            self.fc2 = layer.Linear(in_features // 4, 10).to(x.device)

        x = self.fc1(x)
        x = self.spike(x)
        print('fc',x)
        #fc本来就不需要transform
        feat = x.clone().view(self.T, -1, *x.shape[1:])  # [T, N, ...]
        features.append(feat.mean(0))  # [N, ...]
        # features.append(x.clone())  # after fc1 spike feature

        x = self.fc2(x)
        x = self.temporal_expand_output(x)    # [T, N, 10]
        # return x.mean(0)                      # [N, 10]
        return x, features

class Feature_Converter(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel),  # Depthwise
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)