import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from mmseg.core import add_prefix
from mmseg.models import SEGMENTORS
from .. import builder
from .encoder_decoder import EncoderDecoder3D


@SEGMENTORS.register_module()
class MultivewEncoderDecoder(EncoderDecoder3D):
    r"""

    """

    def __init__(self, img_backbone, fusion_layer, img_neck=None, **kwargs):
        super(MultivewEncoderDecoder, self).__init__(**kwargs)

        # image branch
        self.img_backbone = builder.build_seg_backbone(img_backbone)
        self.fusion_layer = builder.build_fusion_layer(fusion_layer)
        if img_neck:
            self.img_neck = builder.build_neck(img_neck)

    def extract_img_feat(self, img):
        """Directly extract features from the img backbone+neck."""
        x = self.img_backbone(img)
        if self.with_img_neck:
            x = self.img_neck(x)
        return x

    def extract_fused_feat(self, img, pts, pts_feature, img_metas):
        img_feature = self.extract_img_feat(img)
        fused_feature = self.fusion_layer(img_feature, pts, pts_feature,
                                          img_metas)
        return fused_feature

    def encode_decode(self, points, img, img_metas):
        """Encode points with backbone and decode into a semantic segmentation
        map of the same size as input.

        Args:
            points (torch.Tensor): Input points of shape [B, N, 3+C].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            torch.Tensor: Segmentation logits of shape [B, num_classes, N].
        """
        pts_feature = self.extract_feat(points)
        x = self.extract_fused_feat(img, points, pts_feature, img_metas)
        out = self._decode_head_forward_test(x, img_metas)
        return out

    def forward_dummy(self, points, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(points, img, None)

        return seg_logit

    def forward_train(self, points, img, img_metas, pts_semantic_mask):
        """Forward function for training.

        Args:
            points (list[torch.Tensor]): List of points of shape [N, C].
            img_metas (list): Image metas.
            pts_semantic_mask (list[torch.Tensor]): List of point-wise semantic
                labels of shape [N].

        Returns:
            dict[str, Tensor]: Losses.
        """
        points_cat = torch.stack(points)
        pts_semantic_mask_cat = torch.stack(pts_semantic_mask)

        # extract features using backbone
        pts_feature = self.extract_feat(points_cat)
        x = self.extract_fused_feat(img, points_cat, pts_feature, img_metas)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      pts_semantic_mask_cat)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, pts_semantic_mask_cat)
            losses.update(loss_aux)

        if self.with_regularization_loss:
            loss_regularize = self._loss_regularization_forward_train()
            losses.update(loss_regularize)

        return losses
