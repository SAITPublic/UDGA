# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

# from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet.models.backbones.resnet import ResNet

from torch.cuda.amp.autocast_mode import autocast
from .bevdet import BEVDet
from memory_profiler import profile 

@DETECTORS.register_module()
class BEVDetDA(BEVDet):    
    def __init__(self, vit_weight=10, **kwargs):
        self.vit_weight = vit_weight
        super(BEVDetDA, self).__init__(**kwargs)
        self.loss_bev = torch.nn.CosineEmbeddingLoss()
    
    def get_bev_feature_loss(self, bev_feat, src_feat):
        B = bev_feat.shape[0]
        with autocast(enabled=False):
            y = torch.autograd.Variable(torch.Tensor(B).to(bev_feat.device).fill_(1.0))
            vit_loss = self.loss_bev(bev_feat.view(B, -1), src_feat.view(B, -1), y)            
        return self.vit_weight * vit_loss

    def extract_feat(self, img):
        x = self.image_encoder(img[0])  
        x = self.img_view_transformer([x] + img[1:]) 
        if isinstance(x, tuple):
            out = self.bev_encoder(x[0]) 
        else : 
            out = self.bev_encoder(x) 
        return [out], x
        
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
        assert self.with_pts_bbox
        losses = dict()        
        img_feats, bev_feats = self.extract_feat(img=img_inputs) 
        if isinstance(bev_feats, tuple):
            losses.update({'bev_csm_loss':self.get_bev_feature_loss(bev_feats[0], bev_feats[1])})
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
    
    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _ = self.extract_feat(img=img_inputs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return bbox_list
