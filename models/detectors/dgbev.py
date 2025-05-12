# Copyright (c) Phigent Robotics. All rights reserved.

import math
import torch
from mmcv.runner import force_fp32, load_checkpoint
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.cuda.amp.autocast_mode import autocast

from mmdet.models import DETECTORS
from .centerpoint import CenterPoint
from .. import builder
from mmdet3d.models import backbones, build_model
from .mvx_two_stage import MVXTwoStageDetector
from .bevdet import BEVDepth
from mmcv.cnn import initialize
from mmdet3d.models.utils import clip_sigmoid_keep_value as sigmoid
from mmdet3d.core import bbox3d2result
import cv2
import os


@DETECTORS.register_module()
class DGBEV(BEVDepth):
    def __init__(self, img_view_transformer, 
                    img_bev_encoder_backbone=None, 
                    img_bev_encoder_neck=None, 
                    img_bev_da_module=None,
                    pts_backbone=None, 
                    pts_neck=None, 
                    bev_loss_cfg=None, 
                    teacher_model_cfg=None, 
                    res_reg_loss_cfg=None,
                    scale_invariant=False,
                    without=False,
                    visual=False,
                    get_bev=False,
                    pixel_const=500,
                    **kwargs):
        super(DGBEV, self).__init__(**kwargs)
        #self.now_epoch = 0

        if img_view_transformer: self.img_view_transformer         = builder.build_neck(img_view_transformer)
        if img_bev_encoder_backbone: self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        if img_bev_encoder_neck: self.img_bev_encoder_neck         = builder.build_neck(img_bev_encoder_neck)
        if img_bev_da_module: self.img_bev_da_module               = builder.build_neck(img_bev_da_module)
        if teacher_model_cfg: self.teacher_model                   = builder.build_model(teacher_model_cfg)
        else: self.teacher_model = None

        self.loss_res_reg_loss_fun = None

        self.scale_invariant    = scale_invariant
        self.without            = without
        self.visual             = visual
        self.get_bev            = get_bev

        self.count = 0

        self.x_sample_num  = 24
        self.y_sample_num  = 24
        self.enlarge_width = 1.6

        self.pixel_const = pixel_const

    def init_pretrained(self):
        load_checkpoint(self.teacher_model, self.teacher_model.guided_weights.checkpoint, map_location='cpu')
        self.teacher_model.fp16_enabled=False
        for m in self.teacher_model.modules(): m.fp16_enabled=False
        for p in self.teacher_model.parameters(): p.requires_grad = False
        self.teacher_model.eval()            

    def get_gt_sample_grid(self,corner_points2d):
        dH_x, dH_y = corner_points2d[0] - corner_points2d[1] 
        dW_x, dW_y = corner_points2d[0] - corner_points2d[2]
        raw_grid_x = torch.linspace(corner_points2d[0][0], corner_points2d[1][0], self.x_sample_num).view(1,-1).repeat(self.y_sample_num,1)
        raw_grid_y = torch.linspace(corner_points2d[0][1],corner_points2d[2][1], self.y_sample_num).view(-1,1).repeat(1,self.x_sample_num)
        raw_grid = torch.cat((raw_grid_x.unsqueeze(2),raw_grid_y.unsqueeze(2)), dim=2)
        raw_grid_x_offset = torch.linspace(0,-dW_x,self.x_sample_num).view(-1,1).repeat(1,self.y_sample_num)
        raw_grid_y_offset = torch.linspace(0,-dH_y,self.y_sample_num).view(1,-1).repeat(self.x_sample_num,1)
        raw_grid_offset = torch.cat((raw_grid_x_offset.unsqueeze(2),raw_grid_y_offset.unsqueeze(2)),dim=2)
        grid = raw_grid + raw_grid_offset #X_sample,Y_sample,2
        grid[:,:,0] = torch.clip(((grid[:,:,0] - (self.img_view_transformer.bx[0].to(grid.device) - self.img_view_transformer.dx[0].to(grid.device) / 2.)
                       ) / self.img_view_transformer.dx[0].to(grid.device) / (self.img_view_transformer.nx[0].to(grid.device)-1))*2.0 - 1.0 ,min=-1.0,max=1.0)
        grid[:,:,1] = torch.clip(((grid[:,:,1] - (self.img_view_transformer.bx[1].to(grid.device) - self.img_view_transformer.dx[1].to(grid.device) / 2.)
                       ) / self.img_view_transformer.dx[1].to(grid.device) / (self.img_view_transformer.nx[1].to(grid.device)-1))*2.0 - 1.0 ,min=-1.0,max=1.0)
        
        return grid.unsqueeze(0)  

    def get_inner_feat(self,gt_bboxes_3d,img_feats,pts_feats):
        """Use grid to sample features of key points"""
        device = img_feats.device
        dtype = img_feats[0].dtype

        img_feats_sampled_list = []
        pts_feats_sampled_list = []
        
        for sample_ind in torch.arange(len(gt_bboxes_3d)):
            img_feat = img_feats[sample_ind].unsqueeze(0)   #1,C,H,W
            pts_feat = pts_feats[sample_ind].unsqueeze(0)   #1,C,H,W
            
            bbox_num, corner_num, point_num = gt_bboxes_3d[sample_ind].corners.shape
            
            for bbox_ind in torch.arange(bbox_num):
                if self.enlarge_width>0:
                    gt_sample_grid = self.get_gt_sample_grid(gt_bboxes_3d[sample_ind].enlarged_box(self.enlarge_width).corners[bbox_ind][[0,2,4,6],:-1]).to(device)
                else:
                    gt_sample_grid = self.get_gt_sample_grid(gt_bboxes_3d[sample_ind].corners[bbox_ind][[0,2,4,6],:-1]).to(device)  #1,sample_y,sample_x,2
                
                img_feats_sampled_list.append(F.grid_sample(img_feat, grid=gt_sample_grid, align_corners=False, mode='bilinear'))#'bilinear')) #all_bbox_num,C,y_sample,x_sample
                pts_feats_sampled_list.append(F.grid_sample(pts_feat, grid=gt_sample_grid, align_corners=False, mode='bilinear'))#'bilinear')) #all_bbox_num,C,y_sample,x_sample
                
        return torch.cat(img_feats_sampled_list,dim=0), torch.cat(pts_feats_sampled_list,dim=0)

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
        losses = dict()
        
        img_feats, pts_feats, depth = self.extract_feat(
        points, img=img_inputs, img_metas=img_metas)

        depth_gt = img_inputs[-1]        

        c = self.pixel_const
        intrinsics = img_inputs[3]
        post_rotat = img_inputs[4]
        new_intrin = torch.matmul(post_rotat, intrinsics)
        scale_depth_gt = depth_gt / (c*torch.sqrt(1/torch.square(new_intrin[...,0,0])+1/torch.square(new_intrin[...,1,1]))[...,None,None])
        scale_loss_depth = self.get_depth_loss(scale_depth_gt, depth)
        losses.update(dict(scale_loss_depth=scale_loss_depth))
        # loss_depth = self.get_depth_loss(depth_gt, depth)
        # losses.update(dict(loss_depth=loss_depth))

        # rpn loss
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)     
        losses.update(losses_pts)

        return losses

    def bev_encoder(self, x):
        if getattr(self,'img_bev_encoder_backbone',None):
            x = self.img_bev_encoder_backbone(x)
        if getattr(self,'img_bev_encoder_neck',None):
            x = self.img_bev_encoder_neck(x)
        if getattr(self,'img_bev_da_module',None):
            x = self.img_bev_da_module(x)
        return x

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
  
        img_feats,_ , _ = self.extract_feat(points, img=img, img_metas=img_metas)

        if self.get_bev:
            if self.count % 4 == 0:
                for bat_idx, batch_img_feats in enumerate(img_feats[0]):
                    os.makedirs(f"/home/sr5/rsu/users/kuaicv/cvpr/outputs_bev/{self.count}/", exist_ok=True)
                    for img_feat_idx in range(len(batch_img_feats)):
                        if img_feat_idx % 32 != 0: continue
                        data = batch_img_feats[img_feat_idx].clone().detach().cpu().numpy()
                        data = np.clip(data, 0, data.max())
                        data = ((data - data.min())/(data.max() - data.min()) * 255).astype(np.uint8)
                        data = cv2.applyColorMap(data, cv2.COLORMAP_JET)
                        cv2.imwrite(f"/home/sr5/rsu/users/kuaicv/cvpr/outputs_bev/{self.count}/{bat_idx}_{img_feat_idx:04d}.jpg", data)
                    mean_data = torch.mean(batch_img_feats, dim=0).clone().detach().cpu().numpy()
                    mean_data = np.clip(mean_data, 0, mean_data.max())
                    mean_data = ((mean_data - mean_data.min())/(mean_data.max() - mean_data.min()) * 255).astype(np.uint8)
                    mean_data = cv2.applyColorMap(mean_data, cv2.COLORMAP_JET)
                    cv2.imwrite(f"/home/sr5/rsu/users/kuaicv/cvpr/outputs_bev/{self.count}/{bat_idx}_mean.jpg", mean_data)

                if self.teacher_model:
                    with torch.no_grad():
                        if not self.loss_res_reg_loss_fun: bev_lidar = self.teacher_model.forward_teacher(points)
                        else: bev_lidar, logits_lidar = self.teacher_model.forward_teacher(points, res=True)

                    for bat_idx, batch_img_feats in enumerate(bev_lidar):
                        os.makedirs(f"/home/sr5/rsu/users/kuaicv/cvpr/outputs_bev/{self.count}/", exist_ok=True)
                        for img_feat_idx in range(len(batch_img_feats)):
                            if img_feat_idx % 32 != 0: continue
                            data = batch_img_feats[img_feat_idx].clone().detach().cpu().numpy()
                            data = np.clip(data, 0, data.max())
                            data = ((data - data.min())/(data.max() - data.min()) * 255).astype(np.uint8)
                            hist, bins = np.histogram(data.flatten(), 256, [0, 256])
                            cdf = hist.cumsum()
                            cdf_m = np.ma.masked_equal(cdf,0)
                            cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
                            cdf = np.ma.filled(cdf_m,0).astype('uint8')
                            data = cdf[data]
                            data = cv2.applyColorMap(data, cv2.COLORMAP_JET)
                            cv2.imwrite(f"/home/sr5/rsu/users/kuaicv/cvpr/outputs_bev/{self.count}/{bat_idx}_{img_feat_idx:04d}_bev.jpg", data)
                        mean_data = torch.mean(batch_img_feats, dim=0).clone().detach().cpu().numpy()
                        mean_data = np.clip(mean_data, 0, mean_data.max())
                        mean_data = ((mean_data - mean_data.min())/(mean_data.max() - mean_data.min()) * 255)
                        mean_data *= 1.5
                        mean_data = np.clip(mean_data, 0, 255).astype(np.uint8)
                        mean_data = cv2.applyColorMap(mean_data, cv2.COLORMAP_JET)
                        cv2.imwrite(f"/home/sr5/rsu/users/kuaicv/cvpr/outputs_bev/{self.count}/{bat_idx}_mean_bev.jpg", mean_data)

            self.count += 1


        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            if self.visual:
                result_dict['img_metas'] = img_metas
                #rots, trans, intrins, post_rots, post_trans, depth_gt = img[1:]
                #result_dict['rotations'] = rots[0]
                #result_dict['translations'] = trans[0]
                #result_dict['intrinsics'] = intrins[0]
                #result_dict['post_rotations'] = post_rots[0]
                #result_dict['post_trainslations'] = post_trans[0]
        
        return bbox_list