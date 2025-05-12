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
from .bevssl import BEVSSL3

import matplotlib.pyplot as plt
import cv2
import os

@DETECTORS.register_module()
class BEVSSL3_unsup(BEVSSL3):
    def __init__(self, ssl_loss='bce_loss', ssl_weight=30.0, dbg=False, dbg_rgb=False, dbg_str=False, **kwargs):
        super(BEVSSL3_unsup, self).__init__(**kwargs)
        self.ssl_weight = ssl_weight
        self.dbg = dbg # no loss_depth if True
        self.dbg_rgb = dbg_rgb # no loss_rgb if True
        self.dbg_str = dbg_str # no loss_stereo_depth if True
        if ssl_loss == 'l1_loss' :
            self.ssl_loss = torch.nn.SmoothL1Loss()
        else : 
            self.ssl_loss = None

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
        x, depth = self.img_view_transformer([x] + img_inputs[1:]) # x shape: torch.Size([8, 64, 128, 128])        
        
        if not self.dbg :
            depth_gt = img_inputs[-1]
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
        if not self.dbg_str : 
            stereo_depth = self.reprojection(depth, *img_inputs[1:6])
            loss_stereo_depth = self.get_stereo_depth_loss(depth, stereo_depth, weight=self.ssl_weight)
            losses.update(dict(loss_stereo_depth=loss_stereo_depth))

        # Photometric Reprojection Error
        if not self.dbg_rgb : 
            rgb_s, rgb_t = self.photo_reprojection(depth, *img_inputs[:6])
            if rgb_s.shape[2] > 1 and rgb_t.shape[2] > 0 :
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
        
        if self.ssl_loss is not None : 
            stereo_depth = stereo_depth.view(B, N*2, self.img_view_transformer.D, H, W)
            stereo_depth_gt = stereo_depth_gt.view(B, N*2, self.img_view_transformer.D, H, W)
            loss_stereo_depth = self.ssl_loss(stereo_depth[loss_weight], stereo_depth_gt[loss_weight])
        else : # bce
            stereo_depth_gt = stereo_depth_gt.sigmoid().view(B, N*2, self.img_view_transformer.D, H, W)
            stereo_depth = stereo_depth.sigmoid().view(B, N*2, self.img_view_transformer.D, H, W)
            loss_stereo_depth = F.binary_cross_entropy(stereo_depth, stereo_depth_gt.detach(), weight=loss_weight)
        loss_stereo_depth = weight * loss_stereo_depth
        return loss_stereo_depth


    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):          
        for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        if not isinstance(img_inputs[0][0],list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], **kwargs)
        else:            
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)


@DETECTORS.register_module()
class debug_BEVSSL3(BEVSSL3):      
    def __init__(self, **kwargs):
        super(debug_BEVSSL3, self).__init__(**kwargs)
        self.depth_up=nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.vis_count = 0

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
        
        # 240512
        depth = self.depth_up(depth)
        self.depth_vis(depth, depth_gt, img_metas[0]['img_filename'])

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
        stereo_depth = self.reprojection(depth, *img_inputs[1:6], downsample=self.downsample)
        loss_stereo_depth = self.get_stereo_depth_loss(depth, stereo_depth)
        losses.update(dict(loss_stereo_depth=loss_stereo_depth))

        # Photometric Reprojection Error
        # rgb_s, rgb_t = self.photo_reprojection(depth, *img_inputs[:6], downsample=self.downsample)
        # loss_rgb = self.get_rgb_loss(rgb_s, rgb_t)
        # losses.update(dict(loss_rgb=loss_rgb))


        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)

        img_feats = [x] # x shape: BEV Encoder output: torch.Size([8, 256, 128, 128])

        assert self.with_pts_bbox
        
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                        gt_labels_3d, img_metas,
                                        gt_bboxes_ignore)

        losses.update(losses_pts)

        return losses

    # !!need to remove eval while training
    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):       
        return self.simple_test(points, img_metas, img_inputs, **kwargs)

    def simple_test(self, points, img_metas, img=None, rescale=False):        
        # Image Encoder
        depth_gt = img[-1]            
        imgs = img[0]
        B, N, C, imH, imW = imgs.shape        
        imgs = imgs.contiguous().view(B * N, C, imH, imW)
        
        with torch.no_grad():
            x = self.img_backbone(imgs)    
            if self.with_img_neck:
                x = self.img_neck(x)
            _, output_dim, ouput_H, output_W = x.shape
            x = x.view(B, N, output_dim, ouput_H, output_W)
            x, depth = self.img_view_transformer([x] + img[1:])
            
            # 240512
            depth = self.depth_up(depth)

            x = self.img_bev_encoder_backbone(x)
            x = self.img_bev_encoder_neck(x)
        # self.depth_vis(depth, depth_gt, img_metas[0]['img_filename'])    
        if self.vis_count < 1000 :     
            depth_vis(depth, depth_gt, img_metas[0]['img_filename'], 'vis/depth_ssl')
            self.vis_count += 1

        img_feats = [x]
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            if self.visual:
                result_dict['img_metas'] = img_metas
        
        return bbox_list


@DETECTORS.register_module()
class debug_BEVDepth(BEVDepth):      
    def __init__(self, **kwargs):
        super(BEVDepth, self).__init__(**kwargs)
        self.depth_up=nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.vis_count = 0
    
    # !!need to remove eval while training
    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):       
        return self.simple_test(points, img_metas, img_inputs, **kwargs)

    def simple_test(self, points, img_metas, img=None, rescale=False):        
        # Image Encoder
        depth_gt = img[-1]            
        imgs = img[0]
        B, N, C, imH, imW = imgs.shape        
        imgs = imgs.contiguous().view(B * N, C, imH, imW)
        
        with torch.no_grad():
            x = self.img_backbone(imgs)    
            if self.with_img_neck:
                x = self.img_neck(x)
            _, output_dim, ouput_H, output_W = x.shape
            x = x.view(B, N, output_dim, ouput_H, output_W)
            x, depth = self.img_view_transformer([x] + img[1:])
            
            # 240512
            depth = self.depth_up(depth)

            x = self.img_bev_encoder_backbone(x)
            x = self.img_bev_encoder_neck(x)
        
        if self.vis_count < 1000 :     
            depth_vis(depth, depth_gt, img_metas[0]['img_filename'], 'vis/depth')
            self.vis_count += 1

        img_feats = [x]
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            if self.visual:
                result_dict['img_metas'] = img_metas
        
        return bbox_list


def PIL2OpenCV(pil_image):
    numpy_image= np.array(pil_image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image

cmap = plt.cm.get_cmap("jet", 256)
cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 256

def depth_vis(depth_pred, depth_gt, img_filename, vis_dir='vis'):
    B, N, H, W = depth_gt.shape 
    BN, D, height, width = depth_pred.shape
    depth_pred = depth_pred.view(B, N, D, height, width)[0].clone().detach()
    max_range = 60
    for i in range(0,3): 
        gtmap = depth_gt[0][i].cpu() 
        depthmap = depth_pred[i].sigmoid().view(D, height, width).cpu()
        depthmap = np.argmax(depthmap, axis=0)
        
        # sparse depthmap인 경우 depth가 있는 곳만 추출합니다.        
        depth_pixel_v_s, depth_pixel_u_s = np.where(gtmap > 0)
        # depth_pixel_v_s, depth_pixel_u_s = np.where(depthmap > 1)

        H, W = depthmap.shape
        color_depthmap = np.zeros((H, W, 3)).astype(np.uint8)
        for depth_pixel_v, depth_pixel_u in zip(depth_pixel_v_s, depth_pixel_u_s):
            depth = depthmap[depth_pixel_v, depth_pixel_u]
            color_index = int(255 * min(depth, max_range) / max_range)
            color = cmap[color_index, :]
            cv2.circle(color_depthmap, (depth_pixel_u, depth_pixel_v), 1, color=tuple(color), thickness=1)
                
        # img_filename = (filename, img)
        out_file = os.path.splitext(os.path.basename(img_filename[i][0]))[0] 
        # img = cv2.imread(img_filename[i], cv2.IMREAD_UNCHANGED)            
        img = PIL2OpenCV(img_filename[i][1])
        resize_img = cv2.resize(img, dsize=(color_depthmap.shape[1], color_depthmap.shape[0]))
        added_image = cv2.addWeighted(resize_img,0.5,color_depthmap,0.5,0)         
        cv2.imwrite(f'{vis_dir}/{out_file}_pred.png', added_image)
    
        # gt_pixel_v_s, gt_pixel_u_s = np.where(gtmap > 0)
        H, W = gtmap.shape
        color_gtmap = np.zeros((H, W, 3)).astype(np.uint8)
        for depth_pixel_v, depth_pixel_u in zip(depth_pixel_v_s, depth_pixel_u_s):
            depth = gtmap[depth_pixel_v, depth_pixel_u]
            color_index = int(255 * min(depth, max_range) / max_range)
            color = cmap[color_index, :]
            cv2.circle(color_gtmap, (depth_pixel_u, depth_pixel_v), 1, color=tuple(color), thickness=1)  
        resize_gt_img = cv2.resize(img, dsize=(color_gtmap.shape[1], color_gtmap.shape[0]))
        gt_added_image = cv2.addWeighted(resize_gt_img,0.5,color_gtmap,0.5,0)         
        cv2.imwrite(f'{vis_dir}/{out_file}_gt.png', gt_added_image)
