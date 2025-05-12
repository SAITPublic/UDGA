# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

# from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet.models.backbones.resnet import ResNet

from .bevdet import BEVDet
from memory_profiler import profile 

@DETECTORS.register_module()
class BEVDetUDA(BEVDet):       
    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list
