import json
import yaml
import random
import math
import torch
import numpy as np
from mmcv.runner import force_fp32, load_checkpoint
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.cuda.amp.autocast_mode import autocast

from mmdet.models import DETECTORS

from .bevdet import BEVDepth
from mmdet3d.models.utils import Adapter

import PIL
from PIL import Image
from torchvision import datasets, transforms

import open3d as o3d



@DETECTORS.register_module()
class BEVSSL3(BEVDepth):
    def __init__(self, pixel_const=500, **kwargs):
        super(BEVSSL3, self).__init__(**kwargs)
        self.data_config = kwargs['img_view_transformer']['data_config']
        self.grid_config = kwargs['img_view_transformer']['grid_config']
        
        if self.data_config['Ncams'] == 6:
            self.cam_pair_idx = [[3,1],  # Front_left
                                 [0,2],  # Front
                                 [1,5],  # Front_right
                                 [0,4],  # Back_left
                                 [3,5],  # Back
                                 [4,2],] # Back_right
        else:
            self.cam_pair_idx = [[1,2],  # Front_left - Front
                                 [3,0],  # Front - Front_right
                                 [0,4],  # Front_right - Side_right
                                 [1],
                                 [2],] # Back_left - Front_left
        self.pixel_const = pixel_const

        self.compute_ssim_loss = SSIM().to('cuda')

# torch.Size([48, 89, 16, 44]) torch.Size([8, 6, 3, 3]) torch.Size([8, 6, 3]) torch.Size([8, 6, 3, 3]) torch.Size([8, 6, 3, 3]) torch.Size([8, 6, 3])
    def reprojection(self, depth_pred, rots, trans, intrins, post_rots, post_trans, downsample=16):
        BN, D, height, width = depth_pred.shape
        device = depth_pred.device
        # depth_feat = self.conv_bottle(depth_pred)
        B = BN//len(self.cam_pair_idx)
        N = len(self.cam_pair_idx)

        # coor_w = torch.linspace(0, width-1, width).expand(BN, D, height, width).permute(0,1,3,2).to(depth_pred.device)
        # coor_h = torch.linspace(0, height-1, height).expand(BN, D, width, height).to(depth_pred.device)
        # coor_d = torch.linspace(1, D, D).expand(BN, width, height, D).permute(0,3,1,2).to(depth_pred.device)

        coor_w = torch.linspace(0, width-1, width).to(device)
        coor_h = torch.linspace(0, height-1, height).to(device)
        coor_d = torch.linspace(1, D, D).to(device)
        coor_w, coor_h, coor_d = torch.meshgrid(coor_w,coor_h,coor_d)
        whd = torch.stack([coor_w*downsample, 
                           coor_h*downsample, 
                           coor_d], -1).permute(2,0,1,3).expand(B,N,D,width,height,3).reshape(B,N,D*width*height,3)

        # whd = torch.stack([coor_w*downsample, \
        #                    coor_h*downsample, \
        #                    coor_d], dim=-1).view(B, N, width*height*D, 3) # ([8, 6, 59, 44, 16, 3])
        
        depth_feats = depth_pred.view(B, N, D*width*height)
        
        if self.data_config['Ncams'] != 6:
            N = N-1

        stereo_depth = []
        for b in range(B):
            for cam, pair in enumerate(self.cam_pair_idx):
                for pid in pair:
                    depth_img = whd[b,cam]
                    depth_feat = depth_feats[b,cam]
                    # Depth to Point
                    post_r = torch.inverse(post_rots[b, cam])
                    post_t = post_trans[b, cam]
                    depth_img = (depth_img - post_t).matmul(post_r.T)
                    depth_pts = torch.cat([depth_img[:,:2]*depth_img[:,2:3], depth_img[:,2:3]], 1)
                    rot = rots[b,cam]
                    intrin = intrins[b,cam]
                    tran = trans[b,cam]
                    combine = rot.matmul(torch.inverse(intrin))
                    depth_pts = depth_pts.matmul(combine.T) + tran
                    
                    # Point to adjacent cam Depth
                    rot = rots[b,pid]
                    intrin = intrins[b,pid]
                    tran = trans[b,pid]
                    combine_inv = torch.inverse(rot.matmul(torch.inverse(intrin)))
                    adj_depth = (depth_pts[:,:3] - tran).matmul(combine_inv.T)
                    adj_depth = torch.cat([adj_depth[:,:2]/adj_depth[:,2:3],
                                   adj_depth[:,2:3]], 1)
                    post_r = post_rots[b,pid]
                    post_t = post_trans[b,pid:pid+1,:]
                    adj_depth = adj_depth.matmul(post_r.T)+post_t
                    adj_depth[:,:2] = torch.round(adj_depth[:,:2]/downsample)
                    
                    kept1 = (adj_depth[:, 0] >= 0) & (adj_depth[:, 0] < width)  \
                          & (adj_depth[:, 1] >= 0) & (adj_depth[:, 1] < height) \
                          & (adj_depth[:, 2] < self.grid_config['dbound'][1])   \
                          & (adj_depth[:, 2] >= self.grid_config['dbound'][0])

                    depth_feat = depth_feat[kept1]
                    coor = adj_depth[kept1].to(device)
                    ranks = coor[:, 0] + coor[:, 1] * width + coor[:, 2]/100.
                    sort = (ranks).argsort()
                    coor = coor[sort]
                    depth_feat = depth_feat[sort]

                    kept2 = torch.ones(coor.shape[0], dtype=torch.bool)
                    kept2[1:] = (ranks[1:] != ranks[:-1])
                    depth_feat = depth_feat[kept2] # ([309, 3])
                    coor = coor[kept2].to(dtype=torch.long) # ([309, 3])
                    adj_depth_map = torch.zeros((D, height, width), device=device, dtype=torch.float32)
                    adj_depth_map[coor[:,2]-1, coor[:,1], coor[:,0]] = depth_feat
                    stereo_depth.append(adj_depth_map)
        
        return torch.stack(stereo_depth, dim=0).view(B,N*2,D,height,width)

    @force_fp32()
    def get_stereo_depth_loss(self, depth, stereo_depth, weight=30.0):
        BN, D, H, W = depth.shape
        B = BN // self.data_config['Ncams']
        N = self.data_config['Ncams']
        depth = depth.view(B, N, D, H, W)
        if self.data_config['Ncams'] != 6:
            N = N-1
        # make stereo depth gt
        stereo_depth_gt = []
        for b in range(B):
            stereo_pair = []
            for pair in self.cam_pair_idx:
                for pid in pair:
                    stereo_pair.append(depth[b,pid])
            stereo_depth_gt.append(torch.stack(stereo_pair, dim=0))
        stereo_depth_gt = torch.stack(stereo_depth_gt, dim=0) 
        
        loss_weight = (~(torch.sum(stereo_depth_gt, dim=2) == 0)) & (~(torch.sum(stereo_depth, dim=2) == 0))
        loss_weight = loss_weight.reshape(B, N*2, 1, H, W).expand(B, N*2,
                                                                    self.img_view_transformer.D,
                                                                    H, W)
        # stereo_depth_gt = (stereo_depth_gt - self.img_view_transformer.grid_config['dbound'][0])\
        #            /self.img_view_transformer.grid_config['dbound'][2]
        # stereo_depth_gt = torch.clip(torch.floor(stereo_depth_gt), 0,
        #                       self.img_view_transformer.D).to(torch.long)
        # stereo_depth_gt_logit = F.one_hot(stereo_depth_gt.reshape(-1),
        #                            num_classes=self.img_view_transformer.D)
        # stereo_depth_gt_logit = stereo_depth_gt_logit.reshape(B, N*2, H, W,
        #                                         self.img_view_transformer.D).permute(
        #     0, 1, 4, 2, 3).to(torch.float32)
        stereo_depth_gt = stereo_depth_gt.sigmoid().view(B, N*2, self.img_view_transformer.D, H, W)
        stereo_depth = stereo_depth.sigmoid().view(B, N*2, self.img_view_transformer.D, H, W)

        loss_stereo_depth = F.binary_cross_entropy(stereo_depth, stereo_depth_gt.detach(),
                                            weight=loss_weight)
        loss_stereo_depth = weight * loss_stereo_depth
        return loss_stereo_depth

    def photo_reprojection(self, depth_pred, img, rots, trans, intrins, post_rots, post_trans, downsample=16):
        B,N,_,H,W = img.shape
        BN, D, height, width = depth_pred.shape
        device = depth_pred.device
        depth = torch.argmax(depth_pred.sigmoid(), dim=1)

        coor_w = torch.linspace(0, width-1, width)
        coor_h = torch.linspace(0, height-1, height)
        coor_w, coor_h = torch.meshgrid(coor_w, coor_h)
        wh = torch.stack([coor_w*downsample, \
                          coor_h*downsample], dim=-1).expand(B, N, width, height, 2).view(B, N, width*height, 2)
        whd = torch.cat([wh.to(device), depth.view(B, N, width*height, 1)], dim=-1)
        # ([8, 6, 59, 44, 16, 3])

        if self.data_config['Ncams'] != 6:
            N = N-1

        src_img, tar_img = [], []        
        for b in range(B):
            for cam, pair in enumerate(self.cam_pair_idx):
                for pid in pair:
                    img_org = whd[b,cam]
                    img_s = img[b,cam].permute(1,2,0)
                    img_t = img[b,pid].permute(1,2,0)
                    # Depth to Point
                    depth_img = (img_org - post_trans[b,cam]).matmul(torch.inverse(post_rots[b,cam]).T)
                    depth_pts = torch.cat([depth_img[:,:2]*depth_img[:,2:3], depth_img[:,2:3]], 1)
                    combine = rots[b,cam].matmul(torch.inverse(intrins[b,cam]))
                    depth_pts = depth_pts.matmul(combine.T) + trans[b,cam]
                    
                    # Point to adjacent cam Depth
                    combine_inv = torch.inverse(rots[b,pid].matmul(torch.inverse(intrins[b,pid])))
                    adj_depth = (depth_pts[:,:3] - trans[b,pid:pid+1,:]).matmul(combine_inv.T)
                    adj_depth = torch.cat([adj_depth[:,:2]/adj_depth[:,2:3],
                                   adj_depth[:,2:3]], 1)
                    adj_depth = adj_depth.matmul(post_rots[b,pid].T)+post_trans[b,pid:pid+1,:] 
                    adj_depth[:,:2] = torch.round(adj_depth[:,:2])
                
                    kept1 = (adj_depth[:, 0] >= 0) & (adj_depth[:, 0] < W)     \
                          & (adj_depth[:, 1] >= 0) & (adj_depth[:, 1] < H)     \
                          & (adj_depth[:, 2] < self.grid_config['dbound'][1])  \
                          & (adj_depth[:, 2] >= self.grid_config['dbound'][0])
                    
                    coor_s = torch.round(img_org[kept1]).to(torch.long)
                    img_s = img_s[coor_s[:,1], coor_s[:,0], :]
                    coor_t = torch.round(adj_depth[kept1]).to(torch.long) 
                    img_t = img_t[coor_t[:,1], coor_t[:,0], :]
                    src_img.append(img_s)
                    tar_img.append(img_t)
        num_img = B*N
        src_img = torch.cat(src_img, dim=0)
        tar_img = torch.cat(tar_img, dim=0)
        
        return src_img.view(1,1,-1,3), tar_img.view(1,1,-1,3)

    def get_rgb_loss(self, rgb1, rgb2, weight=2.0):
        diff_img = (rgb1 - rgb2).abs()
        diff_img = diff_img.clamp(0, 1)
        
        ssim_map = self.compute_ssim_loss(rgb1, rgb2)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        b,h,w,c = diff_img.shape
        loss = weight * diff_img.sum() / (b*h*w*c)
        return loss

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      ov_points=None,
                      ):
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
        
        # Image Encoder
        imgs = img_inputs[0]
        B, N, C, imH, imW = imgs.shape        
        imgs = imgs.contiguous().view(B * N, C, imH, imW)

        # with torch.no_grad():
        x = self.img_backbone(imgs)    
        if self.with_img_neck:
            x = self.img_neck(x)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        depth_gt = img_inputs[-1]
        
        x, depth = self.img_view_transformer([x] + img_inputs[1:]) # x shape: torch.Size([8, 64, 128, 128])

        c = self.pixel_const
        intrinsics = img_inputs[3]
        post_rotat = img_inputs[4]
        new_intrin = torch.matmul(post_rotat, intrinsics)
        scale_depth_gt = depth_gt / (c*torch.sqrt(1/torch.square(new_intrin[...,0,0])+1/torch.square(new_intrin[...,1,1]))[...,None,None])
        valid = (scale_depth_gt<0) | (scale_depth_gt >= self.grid_config['dbound'][1])
        scale_depth_gt[valid] = 0
        scale_loss_depth = self.get_depth_loss(scale_depth_gt, depth)
        losses.update(dict(loss_depth=scale_loss_depth))

        # Reprojection Error
        stereo_depth = self.reprojection(depth, *img_inputs[1:6])
        loss_stereo_depth = self.get_stereo_depth_loss(depth, stereo_depth)
        losses.update(dict(loss_stereo_depth=loss_stereo_depth))

        # Photometric Reprojection Error
        rgb_s, rgb_t = self.photo_reprojection(depth, *img_inputs[:6])
        loss_rgb = self.get_rgb_loss(rgb_s, rgb_t)
        losses.update(dict(loss_rgb=loss_rgb))


        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)

        img_feats = [x] # x shape: BEV Encoder output: torch.Size([8, 256, 128, 128])

        assert self.with_pts_bbox
        
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                        gt_labels_3d, img_metas,
                                        gt_bboxes_ignore)

        losses.update(losses_pts)

        return losses



@DETECTORS.register_module()
class BEVSSL2(BEVDepth):
    def __init__(self, pixel_const=500, **kwargs):
        super(BEVSSL2, self).__init__(**kwargs)
        self.data_config = kwargs['img_view_transformer']['data_config']
        self.grid_config = kwargs['img_view_transformer']['grid_config']
        
        if self.data_config['Ncams'] == 6:
            self.cam_pair_idx = [[3,1],  # Front_left
                                 [0,2],  # Front
                                 [1,5],  # Front_right
                                 [0,4],  # Back_left
                                 [3,5],  # Back
                                 [4,2],] # Back_right
        else:
            self.cam_pair_idx = [[1,2],  # Front_left - Front
                                 [3,0],  # Front - Front_right
                                 [0,4],  # Front_right - Side_right
                                 [1],
                                 [2],] # Back_left - Front_left
        self.pixel_const = pixel_const

        self.compute_ssim_loss = SSIM().to('cuda')

# torch.Size([48, 89, 16, 44]) torch.Size([8, 6, 3, 3]) torch.Size([8, 6, 3]) torch.Size([8, 6, 3, 3]) torch.Size([8, 6, 3, 3]) torch.Size([8, 6, 3])
    def reprojection(self, depth_pred, rots, trans, intrins, post_rots, post_trans, downsample=16):
        BN, D, height, width = depth_pred.shape
        device = depth_pred.device
        # depth_feat = self.conv_bottle(depth_pred)
        B = BN//len(self.cam_pair_idx)
        N = len(self.cam_pair_idx)

        # coor_w = torch.linspace(0, width-1, width).expand(BN, D, height, width).permute(0,1,3,2).to(depth_pred.device)
        # coor_h = torch.linspace(0, height-1, height).expand(BN, D, width, height).to(depth_pred.device)
        # coor_d = torch.linspace(1, D, D).expand(BN, width, height, D).permute(0,3,1,2).to(depth_pred.device)

        coor_w = torch.linspace(0, width-1, width).to(device)
        coor_h = torch.linspace(0, height-1, height).to(device)
        coor_d = torch.linspace(1, D, D).to(device)
        coor_w, coor_h, coor_d = torch.meshgrid(coor_w,coor_h,coor_d)
        whd = torch.stack([coor_w*downsample, 
                           coor_h*downsample, 
                           coor_d], -1).permute(2,0,1,3).expand(B,N,D,width,height,3).reshape(B,N,D*width*height,3)

        # whd = torch.stack([coor_w*downsample, \
        #                    coor_h*downsample, \
        #                    coor_d], dim=-1).view(B, N, width*height*D, 3) # ([8, 6, 59, 44, 16, 3])
        
        depth_feats = depth_pred.view(B, N, D*width*height)
        
        if self.data_config['Ncams'] != 6:
            N = N-1

        stereo_depth = []
        for b in range(B):
            for cam, pair in enumerate(self.cam_pair_idx):
                for pid in pair:
                    depth_img = whd[b,cam]
                    depth_feat = depth_feats[b,cam]
                    # Depth to Point
                    post_r = torch.inverse(post_rots[b, cam])
                    post_t = post_trans[b, cam]
                    depth_img = (depth_img - post_t).matmul(post_r.T)
                    depth_pts = torch.cat([depth_img[:,:2]*depth_img[:,2:3], depth_img[:,2:3]], 1)
                    rot = rots[b,cam]
                    intrin = intrins[b,cam]
                    tran = trans[b,cam]
                    combine = rot.matmul(torch.inverse(intrin))
                    depth_pts = depth_pts.matmul(combine.T) + tran
                    
                    # Point to adjacent cam Depth
                    rot = rots[b,pid]
                    intrin = intrins[b,pid]
                    tran = trans[b,pid]
                    combine_inv = torch.inverse(rot.matmul(torch.inverse(intrin)))
                    adj_depth = (depth_pts[:,:3] - tran).matmul(combine_inv.T)
                    adj_depth = torch.cat([adj_depth[:,:2]/adj_depth[:,2:3],
                                   adj_depth[:,2:3]], 1)
                    post_r = post_rots[b,pid]
                    post_t = post_trans[b,pid:pid+1,:]
                    adj_depth = adj_depth.matmul(post_r.T)+post_t
                    adj_depth[:,:2] = torch.round(adj_depth[:,:2]/downsample)
                    
                    kept1 = (adj_depth[:, 0] >= 0) & (adj_depth[:, 0] < width)  \
                          & (adj_depth[:, 1] >= 0) & (adj_depth[:, 1] < height) \
                          & (adj_depth[:, 2] < self.grid_config['dbound'][1])   \
                          & (adj_depth[:, 2] >= self.grid_config['dbound'][0])

                    depth_feat = depth_feat[kept1]
                    coor = adj_depth[kept1].to(device)
                    ranks = coor[:, 0] + coor[:, 1] * width + coor[:, 2]/100.
                    sort = (ranks).argsort()
                    coor = coor[sort]
                    depth_feat = depth_feat[sort]

                    kept2 = torch.ones(coor.shape[0], dtype=torch.bool)
                    kept2[1:] = (ranks[1:] != ranks[:-1])
                    depth_feat = depth_feat[kept2] # ([309, 3])
                    coor = coor[kept2].to(dtype=torch.long) # ([309, 3])
                    
                    adj_depth_map = torch.zeros((D, height, width), device=device, dtype=torch.float32)
                    adj_depth_map[coor[:,2]-1, coor[:,1], coor[:,0]] = depth_feat
                    stereo_depth.append(adj_depth_map)
        
        return torch.stack(stereo_depth, dim=0).view(B,N*2,D,height,width)

    @force_fp32()
    def get_stereo_depth_loss(self, depth_gt, stereo_depth, weight=30.0):
        B, N, H, W = depth_gt.shape
        if self.data_config['Ncams'] != 6:
            N = N-1
        # make stereo depth gt
        stereo_depth_gt = []
        for b in range(B):
            stereo_pair = []
            for pair in self.cam_pair_idx:
                for pid in pair:
                    stereo_pair.append(depth_gt[b,pid])
            stereo_depth_gt.append(torch.stack(stereo_pair, dim=0))
        stereo_depth_gt = torch.stack(stereo_depth_gt, dim=0) 
        
        loss_weight = (~(stereo_depth_gt == 0)) & (~(torch.sum(stereo_depth, dim=2) == 0))
        loss_weight = loss_weight.reshape(B, N*2, 1, H, W).expand(B, N*2,
                                                                    self.img_view_transformer.D,
                                                                    H, W)
        stereo_depth_gt = (stereo_depth_gt - self.img_view_transformer.grid_config['dbound'][0])\
                   /self.img_view_transformer.grid_config['dbound'][2]
        stereo_depth_gt = torch.clip(torch.floor(stereo_depth_gt), 0,
                              self.img_view_transformer.D).to(torch.long)
        stereo_depth_gt_logit = F.one_hot(stereo_depth_gt.reshape(-1),
                                   num_classes=self.img_view_transformer.D)
        stereo_depth_gt_logit = stereo_depth_gt_logit.reshape(B, N*2, H, W,
                                                self.img_view_transformer.D).permute(
            0, 1, 4, 2, 3).to(torch.float32)
        stereo_depth = stereo_depth.sigmoid().view(B, N*2, self.img_view_transformer.D, H, W)

        loss_stereo_depth = F.binary_cross_entropy(stereo_depth, stereo_depth_gt_logit,
                                            weight=loss_weight)
        loss_stereo_depth = weight * loss_stereo_depth
        return loss_stereo_depth

    def photo_reprojection(self, depth_pred, img, rots, trans, intrins, post_rots, post_trans, downsample=16):
        B,N,_,H,W = img.shape
        BN, D, height, width = depth_pred.shape
        device = depth_pred.device
        depth = torch.argmax(depth_pred.sigmoid(), dim=1)

        coor_w = torch.linspace(0, width-1, width)
        coor_h = torch.linspace(0, height-1, height)
        coor_w, coor_h = torch.meshgrid(coor_w, coor_h)
        wh = torch.stack([coor_w*downsample, \
                          coor_h*downsample], dim=-1).expand(B, N, width, height, 2).view(B, N, width*height, 2)
        whd = torch.cat([wh.to(device), depth.view(B, N, width*height, 1)], dim=-1)
        # ([8, 6, 59, 44, 16, 3])

        if self.data_config['Ncams'] != 6:
            N = N-1

        src_img, tar_img = [], []        
        for b in range(B):
            for cam, pair in enumerate(self.cam_pair_idx):
                for pid in pair:
                    img_org = whd[b,cam]
                    img_s = img[b,cam].permute(1,2,0)
                    img_t = img[b,pid].permute(1,2,0)
                    # Depth to Point
                    depth_img = (img_org - post_trans[b,cam]).matmul(torch.inverse(post_rots[b,cam]).T)
                    depth_pts = torch.cat([depth_img[:,:2]*depth_img[:,2:3], depth_img[:,2:3]], 1)
                    combine = rots[b,cam].matmul(torch.inverse(intrins[b,cam]))
                    depth_pts = depth_pts.matmul(combine.T) + trans[b,cam]
                    
                    # Point to adjacent cam Depth
                    combine_inv = torch.inverse(rots[b,pid].matmul(torch.inverse(intrins[b,pid])))
                    adj_depth = (depth_pts[:,:3] - trans[b,pid:pid+1,:]).matmul(combine_inv.T)
                    adj_depth = torch.cat([adj_depth[:,:2]/adj_depth[:,2:3],
                                   adj_depth[:,2:3]], 1)
                    adj_depth = adj_depth.matmul(post_rots[b,pid].T)+post_trans[b,pid:pid+1,:] 
                    adj_depth[:,:2] = torch.round(adj_depth[:,:2])
                
                    kept1 = (adj_depth[:, 0] >= 0) & (adj_depth[:, 0] < W)     \
                          & (adj_depth[:, 1] >= 0) & (adj_depth[:, 1] < H)     \
                          & (adj_depth[:, 2] < self.grid_config['dbound'][1])  \
                          & (adj_depth[:, 2] >= self.grid_config['dbound'][0])
                    
                    coor_s = torch.round(img_org[kept1]).to(torch.long)
                    img_s = img_s[coor_s[:,1], coor_s[:,0], :]
                    coor_t = torch.round(adj_depth[kept1]).to(torch.long) 
                    img_t = img_t[coor_t[:,1], coor_t[:,0], :]
                    src_img.append(img_s)
                    tar_img.append(img_t)
        num_img = B*N
        src_img = torch.cat(src_img, dim=0)
        tar_img = torch.cat(tar_img, dim=0)
        
        return src_img.view(1,1,-1,3), tar_img.view(1,1,-1,3)

    def get_rgb_loss(self, rgb1, rgb2, weight=2.0):
        diff_img = (rgb1 - rgb2).abs()
        diff_img = diff_img.clamp(0, 1)
        
        ssim_map = self.compute_ssim_loss(rgb1, rgb2)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)
        b,h,w,c = diff_img.shape
        loss = weight * diff_img.sum() / (b*h*w*c)
        return loss

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      ov_points=None,
                      ):
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
        
        # Image Encoder
        imgs = img_inputs[0]
        B, N, C, imH, imW = imgs.shape        
        imgs = imgs.contiguous().view(B * N, C, imH, imW)

        # with torch.no_grad():
        x = self.img_backbone(imgs)    
        if self.with_img_neck:
            x = self.img_neck(x)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        depth_gt = img_inputs[-1]
        
        x, depth = self.img_view_transformer([x] + img_inputs[1:]) # x shape: torch.Size([8, 64, 128, 128])

        c = self.pixel_const
        intrinsics = img_inputs[3]
        post_rotat = img_inputs[4]
        new_intrin = torch.matmul(post_rotat, intrinsics)
        scale_depth_gt = depth_gt / (c*torch.sqrt(1/torch.square(new_intrin[...,0,0])+1/torch.square(new_intrin[...,1,1]))[...,None,None])
        valid = (scale_depth_gt<0) | (scale_depth_gt >= self.grid_config['dbound'][1])
        scale_depth_gt[valid] = 0
        scale_loss_depth = self.get_depth_loss(scale_depth_gt, depth)
        losses.update(dict(loss_depth=scale_loss_depth))

        # Reprojection Error
        stereo_depth = self.reprojection(depth, *img_inputs[1:6])
        loss_stereo_depth = self.get_stereo_depth_loss(depth_gt, stereo_depth)
        losses.update(dict(loss_stereo_depth=loss_stereo_depth))

        # Photometric Reprojection Error
        rgb_s, rgb_t = self.photo_reprojection(depth, *img_inputs[:6])
        loss_rgb = self.get_rgb_loss(rgb_s, rgb_t)
        losses.update(dict(loss_rgb=loss_rgb))


        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)

        img_feats = [x] # x shape: BEV Encoder output: torch.Size([8, 256, 128, 128])

        assert self.with_pts_bbox

        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                        gt_labels_3d, img_metas,
                                        gt_bboxes_ignore)

        losses.update(losses_pts)

        return losses



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)



@DETECTORS.register_module()
class SSLAdapter(BEVSSL2):
    def __init__(self, **kwargs):
        super(SSLAdapter, self).__init__(**kwargs)


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      ov_points=None,
                      ):
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
        
        # Image Encoder
        imgs = img_inputs[0]
        B, N, C, imH, imW = imgs.shape        
        imgs = imgs.contiguous().view(B * N, C, imH, imW)

        with torch.no_grad():
            x = self.img_backbone(imgs)    
            if self.with_img_neck:
                x = self.img_neck(x)
            _, output_dim, ouput_H, output_W = x.shape
            x = x.view(B, N, output_dim, ouput_H, output_W)

        depth_gt = img_inputs[-1]
        
        x, depth = self.img_view_transformer([x] + img_inputs[1:]) # x shape: torch.Size([8, 64, 128, 128])

        c = self.pixel_const
        intrinsics = img_inputs[3]
        post_rotat = img_inputs[4]
        new_intrin = torch.matmul(post_rotat, intrinsics)
        scale_depth_gt = depth_gt / (c*torch.sqrt(1/torch.square(new_intrin[...,0,0])+1/torch.square(new_intrin[...,1,1]))[...,None,None])
        scale_loss_depth = self.get_depth_loss(scale_depth_gt, depth)
        losses.update(dict(loss_depth=scale_loss_depth))

        # Reprojection Error
        stereo_depth = self.reprojection(depth, *img_inputs[1:6])
        loss_stereo_depth = self.get_stereo_depth_loss(depth_gt, stereo_depth)
        losses.update(dict(loss_stereo_depth=loss_stereo_depth))

        # Photometric Reprojection Error
        rgb_s, rgb_t = self.photo_reprojection(depth, *img_inputs[:6])
        loss_rgb = self.get_rgb_loss(rgb_s, rgb_t)
        losses.update(dict(loss_rgb=loss_rgb))


        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)

        img_feats = [x] # x shape: BEV Encoder output: torch.Size([8, 256, 128, 128])

        assert self.with_pts_bbox
        
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                        gt_labels_3d, img_metas,
                                        gt_bboxes_ignore)

        losses.update(losses_pts)

        return losses


    
@DETECTORS.register_module()
class SSLAdapter2(BEVSSL3):
    def __init__(self, **kwargs):
        super(SSLAdapter2, self).__init__(**kwargs)


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      ov_points=None,
                      ):
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
        
        # Image Encoder
        imgs = img_inputs[0]
        B, N, C, imH, imW = imgs.shape        
        imgs = imgs.contiguous().view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)    
        if self.with_img_neck:
            x = self.img_neck(x)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        depth_gt = img_inputs[-1]
        
        x, depth = self.img_view_transformer([x] + img_inputs[1:]) # x shape: torch.Size([8, 64, 128, 128])

        c = self.pixel_const
        intrinsics = img_inputs[3]
        post_rotat = img_inputs[4]
        new_intrin = torch.matmul(post_rotat, intrinsics)
        scale_depth_gt = depth_gt / (c*torch.sqrt(1/torch.square(new_intrin[...,0,0])+1/torch.square(new_intrin[...,1,1]))[...,None,None])
        valid = (scale_depth_gt<0) | (scale_depth_gt >= self.grid_config['dbound'][1])
        scale_depth_gt[valid] = 0
        scale_loss_depth = self.get_depth_loss(scale_depth_gt, depth)
        losses.update(dict(loss_depth=scale_loss_depth))

        # Reprojection Error
        stereo_depth = self.reprojection(depth, *img_inputs[1:6])
        loss_stereo_depth = self.get_stereo_depth_loss(depth, stereo_depth)
        losses.update(dict(loss_stereo_depth=loss_stereo_depth))

        # Photometric Reprojection Error
        rgb_s, rgb_t = self.photo_reprojection(depth, *img_inputs[:6])
        loss_rgb = self.get_rgb_loss(rgb_s, rgb_t)
        losses.update(dict(loss_rgb=loss_rgb))


        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)

        img_feats = [x] # x shape: BEV Encoder output: torch.Size([8, 256, 128, 128])

        assert self.with_pts_bbox
        
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                        gt_labels_3d, img_metas,
                                        gt_bboxes_ignore)

        losses.update(losses_pts)

        return losses