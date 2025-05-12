import math
import torch
from mmcv.runner import force_fp32, load_checkpoint
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.cuda.amp.autocast_mode import autocast

from mmdet.models import DETECTORS
from .. import builder
from mmdet3d.models import backbones, build_model
from .bevdet import BEVDepth
from mmcv.cnn import initialize
from mmdet3d.models.utils import clip_sigmoid_keep_value as sigmoid
from mmdet3d.core import bbox3d2result
import cv2
import os


@DETECTORS.register_module()
class BEVHeight(BEVDepth):

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""

        img_feats, depth = self.extract_img_feat(img, img_metas)
        return (img_feats, None, depth)

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0])

        x, depth = self.img_view_transformer([x] + img[1:])

        x = self.bev_encoder(x)

        return [x], depth

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(points, img=img, img_metas=img_metas)

        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            if self.visual:
                result_dict['img_metas'] = img_metas
        
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        losses = dict()

        img_feats, pts_feats, depth = self.extract_feat(
        points, img=img_inputs, img_metas=img_metas)

        assert self.with_pts_bbox

        depth_gt = img_inputs[-3]
        
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)

        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)

        losses.update(losses_pts)
        return losses
