import numpy as np
import torch
import torch.nn.parallel
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule,ModuleList
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from mmdet.models import BACKBONES


class TNet(nn.Module):
    def __init__(self,
                 trans_dim,
                 in_channels,
                 fc_in_channels=1024,
                 channels=[64, 128, 1024],
                 fc_channels=[512, 256, 4096],
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLu'),
                 init_cfg=None,):
        super(TNet, self).__init__(init_cfg=init_cfg)

        self.convs = nn.ModuleList()
        for i, channel in enumerate(channels):
            self.convs.append = ConvModule(
                in_channels=in_channels if i == 0 else channels[i-1],
                out_channels=channels[i],
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,

            )

        self.fcs = nn.ModuleList()
        for j, channel in enumerate(fc_channels):
            self.fcs.append = nn.Linear(in_channels=fc_in_channels if j == 0 else fc_channels[j-1],
                                        out_channels=fc_channels[j],)

        self.relu = nn.ReLU

        self.trans_dim = trans_dim
        self.num_layers = len(channels)
        self.num_fclayers = len(fc_channels)

    def forward(self, x):

        B = x.shape[0]  # x的维度
        for i in range(self.num_layers):
            x = self.convs[i](x)
        # 此时x是三维数据[B,D,N],L里面的维度数据包含了N个点在该维度下的值，实现maxpool
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # view相当于numpy里的reshape， reshape为1024列

        for j in range(self.num_fclayers):
            x = self.fcs[j](x)

        # repeat 沿着指定的维度重复tensor，此时就有batchsize个一维数组->初始化为对角单位阵

        unit_matrix = Variable(torch.from_numpy(np.eye(self.trans_dim).flatten().astype(
            np.float32))).view(1, self.trans_dim*self.trans_dim).repeat(B, 1)

        unit_matrix = unit_matrix.to(device=x.device)
        x = x+unit_matrix
        x = x.view(-1, self.trans_dim, self.trans_dim)
        return x


@BACKBONES.register_module()
class PointNet(BaseModule):

    def __init__(self,
                 in_channels=3,
                 channels=[64, 128, 1024],
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 global_feat=False,
                 feature_transform=True,
                 init_cfg=None,
                 ):

        super(PointNet, self).__init__(init_cfg=init_cfg)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.stn = TNet(trans_dim=3, in_channels=3)
        self.fstn = TNet(trans_dim=64, in_channels=64)
        self.num_layers = len(channels)
        self.conv = nn.ModuleList()
        for i, channel in enumerate(channels):
            self.conv.append = ConvModule(
                in_channels=in_channels if i == 0 else channels[i-1],
                out_channels=channels[i],
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg if i == 2 else None,
            )

    def forward(self, x):  # should return a tuple
        N = x.shape[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.conv[0](x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x  # pointfeat局部特征
        x = self.conv[1](x)
        x = self.conv[2](x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024, 1).repeat(1, 1, N)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
