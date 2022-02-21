# Copyright (c) OpenMMLab. All rights reserved.
from .paconv_head import PAConvHead
from .pointnet2_head import PointNet2Head
from .pct_head import Point_TransformerHead

__all__ = ['PointNet2Head', 'PAConvHead','Point_TransformerHead']
