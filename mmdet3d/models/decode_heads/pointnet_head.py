from mmcv.cnn.bricks import ConvModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet.models import HEADS
from .decode_head import Base3DDecodeHead

@HEADS.register_module()
class PointNetHead(Base3DDecodeHead):
    def __init__(self, 
                 in_channels=1088,
                 channels = [512,256,128],
                 **kwargs):
        super(PointNetHead,self).__init__(**kwargs)
        self.num_layers = len(channels)
        self.conv = nn.ModuleList()
        for i,channel in enumerate(channels): 
            self.conv.append = ConvModule(
            in_channels = in_channels if i == 0 else channels[i-1],
            out_channels = channels[i],
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
        ) 



    def forward(self,x):
        B,_,N = x.shape
     #   x, trans, trans_feat = self.feat(x)
        for i in range(self.num_layers):
            x = self.conv[i](x)
        x = x.transpose(2,1).contiguous()
        x = x.view(B, N, self.num_classes)
        x = self.cls_seg(x)
        return x
