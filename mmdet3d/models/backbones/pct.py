import torch
from torch import nn as nn
from torch.nn import functional as F
from mmcv.runner import BaseModule, ModuleList, auto_fp16
from mmcv.cnn import ConvModule

from mmdet3d.ops import PointFPModule, build_sa_module
from mmdet.models import BACKBONES


class SA_Layer(nn.Module):

    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


@BACKBONES.register_module()
class Point_Transformer(BaseModule):
    """PointNet2 with Single-scale grouping.

    Args:

    """

    def __init__(self,
                 in_channels,
                 channels=128,
                 num_stages=4,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(Point_Transformer, self).__init__(init_cfg=init_cfg)

        self.num_stages = num_stages

        self.point_embedding = ConvModule(
            in_channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv = ConvModule(
            channels,
            channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.sa = ModuleList()
        for _ in range(self.num_stages):
            self.sa.append(SA_Layer(channels))

    @auto_fp16()
    def forward(self, points):
        # b, n, c --> b, c, n
        x = points.permute(0, 2, 1)
        x = self.point_embedding(x)
        x = self.conv(x)

        outs = []
        for i in range(self.num_stages):
            x = self.sa[i](x)
            outs.append(x)
        return outs
