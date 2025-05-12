# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import math

import torch
import torchvision
from PIL import Image
import mmcv
import json
import random
import copy
import numpy as np
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

import glob, json, re, en_vectors_web_lg


@PIPELINES.register_module()
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=16, vit=False, visual=False):
        self.downsample = downsample
        self.grid_config=grid_config
        self.vit = vit
        self.visual = visual

    def points2depthmap(self, points, height, width, canvas=None):
        height, width = height//self.downsample, width//self.downsample
        depth_map = torch.zeros((height,width), dtype=torch.float32)
        coor = torch.round(points[:,:2]/self.downsample)
        depth = points[:,2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) \
               & (coor[:, 1] >= 0) & (coor[:, 1] < height) \
                & (depth < self.grid_config['dbound'][1]) \
                & (depth >= self.grid_config['dbound'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks+depth/100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:,1],coor[:,0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        if self.vit or self.visual:
            imgs, rots, trans, intrins, post_rots, post_trans, origin_img = results['img_inputs']
        elif len(results['img_inputs']) == 8:
            imgs, rots, trans, intrins, post_rots, post_trans, crop_img, origin_img = results['img_inputs']
        else:
            imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs']
        depth_map_list = []
        for cid in range(rots.shape[0]):
            combine = rots[cid].matmul(torch.inverse(intrins[cid]))
            combine_inv = torch.inverse(combine)
            points_img = (points_lidar.tensor[:,:3] - trans[cid:cid+1,:]).matmul(combine_inv.T)
            points_img = torch.cat([points_img[:,:2]/points_img[:,2:3],
                                   points_img[:,2:3]], 1)
            points_img = points_img.matmul(post_rots[cid].T)+post_trans[cid:cid+1,:]
            depth_map = self.points2depthmap(points_img, imgs.shape[2], imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        if self.vit:
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map, origin_img)
        elif self.visual:
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map)
        elif len(results['img_inputs']) == 8:
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map, crop_img, origin_img)
        else:
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map)
        return results
        # torch.Size([6, 3, 256, 704]) torch.Size([6, 3, 3]) torch.Size([6, 3]) torch.Size([6, 3, 3]) torch.Size([6, 3, 3]) torch.Size([6, 3])



@PIPELINES.register_module()
class PointToMultiViewDepth_SSL(object):
    def __init__(self, grid_config, downsample=16, vit=False, visual=False):
        self.downsample = downsample
        self.grid_config=grid_config
        self.vit = vit
        self.visual = visual

    def points2depthmap(self, points, height, width, canvas=None):
        height, width = height//self.downsample, width//self.downsample
        depth_map = torch.zeros((height,width), dtype=torch.float32)
        coor = torch.round(points[:,:2]/self.downsample)
        depth = points[:,2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 1] >= 0) \
                & (depth < self.grid_config['dbound'][1]) \
                & (depth >= self.grid_config['dbound'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks+depth/100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:,1],coor[:,0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        if self.vit or self.visual:
            imgs, rots, trans, intrins, post_rots, post_trans, origin_img = results['img_inputs']
        elif len(results['img_inputs']) == 8:
            imgs, rots, trans, intrins, post_rots, post_trans, crop_img, origin_img = results['img_inputs']
        else:
            imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs']
        depth_map_list = []
        for cid in range(rots.shape[0]):
            combine = rots[cid].matmul(torch.inverse(intrins[cid]))
            combine_inv = torch.inverse(combine)
            points_img = (points_lidar.tensor[:,:3] - trans[cid:cid+1,:]).matmul(combine_inv.T)
            points_img = torch.cat([points_img[:,:2]/points_img[:,2:3],
                                   points_img[:,2:3]], 1)
            points_img = points_img.matmul(post_rots[cid].T)+post_trans[cid:cid+1,:]
            depth_map = self.points2depthmap(points_img, imgs.shape[2], imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        if self.vit:
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map, origin_img)
        elif self.visual:
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map)
        elif len(results['img_inputs']) == 8:
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map, crop_img, origin_img)
        else:
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map)
        return results
        # torch.Size([6, 3, 256, 704]) torch.Size([6, 3, 3]) torch.Size([6, 3]) torch.Size([6, 3, 3]) torch.Size([6, 3, 3]) torch.Size([6, 3])



# nuscenes: ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
# lyft: ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
# waymo: ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
@PIPELINES.register_module()
class PointToOverlapViewDepth(object):
    def __init__(self, grid_config):
        self.grid_config=grid_config
        self.cam_pair_idx = [[0,1],  # Front_left - Front
                             [1,2],  # Front - Front_right
                             [2,5],  # Front_right - Back_right
                             [5,4],  # Back_right - Back
                             [4,3],  # Back - Back_left
                             [3,0],] # Back_left - Front_left
        self.downsample=16

    def points2depthmap(self, points, height, width, canvas=None):
        height, width = height, width
        depth_map = torch.zeros((height,width), dtype=torch.float32)
        coor = torch.round(points[:,:2])
        depth = points[:,2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) \
               & (coor[:, 1] >= 0) & (coor[:, 1] < height) \
                & (depth < self.grid_config['dbound'][1]) \
                & (depth >= self.grid_config['dbound'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks+depth/100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:,1],coor[:,0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        if len(results['img_inputs']) == 8:
            imgs, rots, trans, intrins, post_rots, post_trans, depth_map, crop_img, origin_img = results['img_inputs']
        else:
            imgs, rots, trans, intrins, post_rots, post_trans, depth_map = results['img_inputs']

        valid_overlaps = []
        # new_imgs = [img for img in imgs]
        for pair in self.cam_pair_idx:
            valid_pair = []
            for cid in pair:
                combine = rots[cid].matmul(torch.inverse(intrins[cid]))
                combine_inv = torch.inverse(combine)
                points_img = (points_lidar.tensor[:,:3] - trans[cid:cid+1,:]).matmul(combine_inv.T)
                points_img = torch.cat([points_img[:,:2]/points_img[:,2:3],
                                    points_img[:,2:3]], 1)
                points_img = points_img.matmul(post_rots[cid].T)+post_trans[cid:cid+1,:]
                valid_pair.append(points_img)
            height, width = imgs.shape[2], imgs.shape[3]
            left_c, right_c = valid_pair
            valid = (left_c[:, 0] >= 0) & (left_c[:, 0] < width) \
                  & (left_c[:, 1] >= 0) & (left_c[:, 1] < height) \
                  & (right_c[:, 0] >= 0) & (right_c[:, 0] < width) \
                  & (right_c[:, 1] >= 0) & (right_c[:, 1] < height)
            valid_ov_pts = points_lidar.tensor[valid]
            
            ov_depth_map_list = []
            
            for cid in pair:
                combine = rots[cid].matmul(torch.inverse(intrins[cid]))
                combine_inv = torch.inverse(combine)
                points_img = (valid_ov_pts[:,:3] - trans[cid:cid+1,:]).matmul(combine_inv.T)
                points_img = torch.cat([points_img[:,:2]/points_img[:,2:3],
                                    points_img[:,2:3]], 1)
                points_img = points_img.matmul(post_rots[cid].T)+post_trans[cid:cid+1,:]
                ov_depth_map = self.points2depthmap(points_img, imgs.shape[2], imgs.shape[3])
                ov_depth_map_list.append(ov_depth_map)
                # rgb = torch.stack([torch.where(ov_depth_map>0,1,0),torch.zeros_like(ov_depth_map),torch.zeros_like(ov_depth_map)], dim=0)
                # valid = (rgb!=0)      
                # new_imgs[cid][valid] = rgb[valid]

            ov_depth_maps = torch.cat(ov_depth_map_list, -1)
            valid_overlaps.append(ov_depth_maps)

        # pts = results['pts_filename'].split('/')[-1]
        # from torchvision.utils import save_image
        # for idx, n_img in enumerate(new_imgs):
        #     save_image(n_img, f'dragon/{pts}_{idx}.png')
        

            
        

        
        if len(results['img_inputs']) == 8:
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map, crop_img, origin_img)
            results['ov_points'] = torch.stack(valid_overlaps)
        else:
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, depth_map)
            results['ov_points'] = torch.stack(valid_overlaps)
        return results



@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

def extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_waymo(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged', 
                    project_pts_to_img_depth=False, is_train=False, data_config=None, aligned=False,
                    cam_depth_range=[4.0, 45.0, 1.0],
                    constant_std=0.5):
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])))
        self.aligned = aligned
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type
        self.project_pts_to_img_depth = project_pts_to_img_depth
        self.cam_depth_range = cam_depth_range
        self.constant_std=constant_std
        self.is_train = is_train
        self.data_config = data_config

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.size[1] != self.img_scale[1]:
            result = Image.new(img.mode, (1920, 1280), (0, 0, 0))
            result.paste(img, (0, 0)) 
        return result

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __call__(self, results, flip=None, scale=None):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        now_name = results['img_filename']

        filename_splits = now_name[0].split("/")
        file_index = filename_splits[5].split(".")[0]
        calib_path = f"{filename_splits[0]}/{filename_splits[1]}/{filename_splits[2]}/{filename_splits[3]}/calib/{file_index}.txt"
        with open(calib_path, 'r') as f: lines = f.readlines()

        for filename in now_name:
            with Image.open(filename) as img:
                if img.size[1] != 1280: img = self.pad(img)
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                filename_splits = filename.split("/")

                cam_idx = int(filename_splits[-2].split("_")[-1])

                intrin = torch.Tensor(np.array([float(info) for info in lines[cam_idx].split(' ')[1:13]]).reshape([3, 4])[:3,:3])
                extrinsic_list = np.array([float(info) for info in lines[cam_idx+6].split(' ')[1:13]]).reshape([3, 4])
                extrinsic_list = extend_matrix(extrinsic_list)
                extrinsic_list = np.linalg.inv(extrinsic_list)
                rot = torch.Tensor(extrinsic_list[:3,:3])
                tran = torch.Tensor(extrinsic_list[:3,3].reshape(-1))

                # augmentation (resize, crop, horizontal flip, rotate)
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                                W=img.width,
                                                                                flip=flip,
                                                                                scale=scale)

                img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

                # for convenience, make augmentation matrices 3x3
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                imgs.append(self.normalize_img(img))
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)

        imgs, rots, trans, intrins, post_rots, post_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans))

        results['img_inputs'] = imgs, rots, trans, intrins, post_rots, post_trans

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)



@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_EXT_waymo(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged', 
                    project_pts_to_img_depth=False, is_train=False, data_config=None, aligned=False,
                    cam_depth_range=[4.0, 45.0, 1.0],
                    constant_std=0.5,aug=[0.1, 0.1, 0.1]):
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])))
        self.aligned = aligned
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type
        self.project_pts_to_img_depth = project_pts_to_img_depth
        self.cam_depth_range = cam_depth_range
        self.constant_std=constant_std
        self.is_train = is_train
        self.data_config = data_config
        self.aug = aug

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.size[1] != self.img_scale[1]:
            result = Image.new(img.mode, (1920, 1280), (0, 0, 0))
            result.paste(img, (0, 0)) 
        return result

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def matrix_to_euler_angles(self, matrix: torch.Tensor, convention: str) -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to Euler angles in radians.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).
            convention: Convention string of three uppercase letters.

        Returns:
            Euler angles in radians as tensor of shape (..., 3).
        """
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        # if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        #     raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        i0 = self._index_from_letter(convention[0])
        i2 = self._index_from_letter(convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            self._angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            self._angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        return torch.stack(o, -1)
                
    def euler_angles_to_matrix(self, euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
        """
        Convert rotations given as Euler angles in radians to rotation matrices.

        Args:
            euler_angles: Euler angles in radians as tensor of shape (..., 3).
            convention: Convention string of three uppercase letters from
                {"X", "Y", and "Z"}.

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
            raise ValueError("Invalid input euler angles.")
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        matrices = [
            self._axis_angle_rotation(c, e)
            for c, e in zip(convention, torch.unbind(euler_angles, -1))
        ]
        # return functools.reduce(torch.matmul, matrices)
        return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])
                
    def _angle_from_tan(self,
        axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool) -> torch.Tensor:
        """
        Extract the first or third Euler angle from the two members of
        the matrix which are positive constant times its sine and cosine.

        Args:
            axis: Axis label "X" or "Y or "Z" for the angle we are finding.
            other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                convention.
            data: Rotation matrices as tensor of shape (..., 3, 3).
            horizontal: Whether we are looking for the angle for the third axis,
                which means the relevant entries are in the same row of the
                rotation matrix. If not, they are in the same column.
            tait_bryan: Whether the first and third axes in the convention differ.

        Returns:
            Euler Angles in radians for each matrix in data as a tensor
            of shape (...).
        """

        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    def _axis_angle_rotation(self, axis: str, angle: torch.Tensor) -> torch.Tensor:
        """
        Return the rotation matrices for one of the rotations about an axis
        of which Euler angles describe, for each value of the angle given.

        Args:
            axis: Axis label "X" or "Y or "Z".
            angle: any shape tensor of Euler angles in radians

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)

        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either X, Y or Z.")

        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    def _index_from_letter(self, letter: str) -> int:
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2
        raise ValueError("letter must be either X, Y or Z.")    
    
    def ext_aug(self, results, img, intrin, rot, tran):
        min_num = 4
        img_src = np.array(img)
        height, width, _ = img_src.shape

        try: 
            gt_boxes = results['ann_info']['gt_bboxes_3d']
            src = gt_boxes.corners.reshape(-1,3)
        except AssertionError:
            return img, intrin, rot, tran

        combine = rot.matmul(torch.inverse(intrin))
        src_corner = (src[:,:3] - tran).matmul(torch.inverse(combine).T)
        src_corner[:, :2] = src_corner[:, :2] / src_corner[:, 2:3] 
        src_corner_wh = src_corner[:, :2]
        n,c = src_corner_wh.shape
        src_corner_wh = src_corner_wh.reshape(n//8,8,2)
        
        svalid = (src_corner_wh[:,:,0]>=0) & (src_corner_wh[:,:,0]<width) \
                  & (src_corner_wh[:,:,1]>=0) & (src_corner_wh[:,:,1]<height)
        svalid = svalid.reshape(-1,8)

        eul = self.matrix_to_euler_angles(torch.Tensor(rot), 'XYZ')
        if torch.rand(1) < 0.5:
            eul -= self.aug * torch.rand(3)
        else:
            eul += self.aug * torch.rand(3)
        rot_tar = self.euler_angles_to_matrix(eul, 'XYZ')
        
        combine = rot_tar.matmul(torch.inverse(intrin))
        tar_corner = (src[:,:3] - tran).matmul(torch.inverse(combine).T)
        tar_corner[:, :2] = tar_corner[:, :2] / tar_corner[:, 2:3] 
        tar_corner_wh = tar_corner[:, :2]
        n,c = tar_corner_wh.shape
        tar_corner_wh = tar_corner_wh.reshape(n//8,8,2)
        tvalid = (tar_corner_wh[:,:,0]>=0) & (tar_corner_wh[:,:,0]<width) \
            & (tar_corner_wh[:,:,1]>=0) & (tar_corner_wh[:,:,1]<height)
        tvalid = tvalid.reshape(-1,8)

        if len(svalid) > len(tvalid): min_len = len(tvalid)
        else:                         min_len = len(svalid)
        
        src_pts, tar_pts = [], []
        for i in range(min_len):
            for j in [0,3,4,7]:
                if svalid[i,j] and tvalid[i,j]:
                    src_pts.append(src_corner_wh[i,j])
                    tar_pts.append(tar_corner_wh[i,j])

        if len(src_pts) < min_num or len(tar_pts) < min_num:
            return img, intrin, rot, tran
        else:
            src_pts = torch.stack(src_pts, 0).numpy().reshape(-1,1,2)
            tar_pts = torch.stack(tar_pts, 0).numpy().reshape(-1,1,2)

            M, _ = cv2.findHomography(src_pts, tar_pts, cv2.RANSAC, 5.0)
            img_tar = cv2.warpPerspective(img_src, np.array(M), (width, height))

            return Image.fromarray(img_tar), intrin, rot_tar, tran
        

    def __call__(self, results, flip=None, scale=None):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        now_name = results['img_filename']

        filename_splits = now_name[0].split("/")
        file_index = filename_splits[5].split(".")[0]
        calib_path = f"{filename_splits[0]}/{filename_splits[1]}/{filename_splits[2]}/{filename_splits[3]}/calib/{file_index}.txt"
        with open(calib_path, 'r') as f: lines = f.readlines()

        for filename in now_name:
            with Image.open(filename) as img:
                if img.size[1] != 1280: img = self.pad(img)
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                filename_splits = filename.split("/")

                cam_idx = int(filename_splits[-2].split("_")[-1])

                intrin = torch.Tensor(np.array([float(info) for info in lines[cam_idx].split(' ')[1:13]]).reshape([3, 4])[:3,:3])
                extrinsic_list = np.array([float(info) for info in lines[cam_idx+6].split(' ')[1:13]]).reshape([3, 4])
                extrinsic_list = extend_matrix(extrinsic_list)
                extrinsic_list = np.linalg.inv(extrinsic_list)
                rot = torch.Tensor(extrinsic_list[:3,:3])
                tran = torch.Tensor(extrinsic_list[:3,3].reshape(-1))

                img, intrin, rot, tran = self.ext_aug(results, img, intrin, rot, tran)

                # augmentation (resize, crop, horizontal flip, rotate)
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                                W=img.width,
                                                                                flip=flip,
                                                                                scale=scale)

                img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

                # for convenience, make augmentation matrices 3x3
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                imgs.append(self.normalize_img(img))
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)

        imgs, rots, trans, intrins, post_rots, post_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans))

        results['img_inputs'] = imgs, rots, trans, intrins, post_rots, post_trans

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module()
class LoadVQA(object):
    def __init__(self, data_root=None, json_file=None):
        assert data_root is not None
        assert json_file is not None
        
        self.vqa_infos = json.load(open(data_root + json_file, 'r'))['questions']

        self.token_to_ix, self.pretrained_emb, max_token = self.tokenize(self.vqa_infos, True)
        self.token_size = self.token_to_ix.__len__()

        self.max_token = -1
        if self.max_token == -1:
            self.max_token = max_token

        self.ans_to_ix, self.ix_to_ans = self.ans_stat(self.vqa_infos)
        self.ans_size = self.ans_to_ix.__len__()

        import pickle as pkl
        pkl.dump(self.pretrained_emb, open('ckpts/nusc_glove.pkl', 'wb'))



    def tokenize(self, stat_ques_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        max_token = 0
        for ques in stat_ques_list:
            # ques: {'image_index': 0, 'program': [{'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [0], 'function': 'filter_size', 'value_inputs': ['large']}, {'inputs': [1], 'function': 'filter_color', 'value_inputs': ['green']}, {'inputs': [2], 'function': 'count', 'value_inputs': []}, {'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [4], 'function': 'filter_size', 'value_inputs': ['large']}, {'inputs': [5], 'function': 'filter_color', 'value_inputs': ['purple']}, {'inputs': [6], 'function': 'filter_material', 'value_inputs': ['metal']}, {'inputs': [7], 'function': 'filter_shape', 'value_inputs': ['cube']}, {'inputs': [8], 'function': 'count', 'value_inputs': []}, {'inputs': [3, 9], 'function': 'greater_than', 'value_inputs': []}], 'question_index': 0, 'image_filename': 'CLEVR_train_000000.png', 'question_family_index': 2, 'split': 'train', 'answer': 'yes', 'question': 'Are there more big green things than large purple shiny cubes?'}
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()
            # words: ['are', 'there', 'more', 'big', 'green', 'things', 'than', 'large', 'purple', 'shiny', 'cubes']

            if len(words) > max_token:
                max_token = len(words)

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb, max_token

    def ans_stat(self, stat_ans_list):
        ans_to_ix = {}
        ix_to_ans = {}

        for ans_stat in stat_ans_list:
            # ans_stat: {'image_index': 0, 'program': [{'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [0], 'function': 'filter_size', 'value_inputs': ['large']}, {'inputs': [1], 'function': 'filter_color', 'value_inputs': ['green']}, {'inputs': [2], 'function': 'count', 'value_inputs': []}, {'inputs': [], 'function': 'scene', 'value_inputs': []}, {'inputs': [4], 'function': 'filter_size', 'value_inputs': ['large']}, {'inputs': [5], 'function': 'filter_color', 'value_inputs': ['purple']}, {'inputs': [6], 'function': 'filter_material', 'value_inputs': ['metal']}, {'inputs': [7], 'function': 'filter_shape', 'value_inputs': ['cube']}, {'inputs': [8], 'function': 'count', 'value_inputs': []}, {'inputs': [3, 9], 'function': 'greater_than', 'value_inputs': []}], 'question_index': 0, 'image_filename': 'CLEVR_train_000000.png', 'question_family_index': 2, 'split': 'train', 'answer': 'yes', 'question': 'Are there more big green things than large purple shiny cubes?'}
            ans = ans_stat['answer']
            # ans: yes
            
            if ans not in ans_to_ix:
                ix_to_ans[ans_to_ix.__len__()] = ans
                ans_to_ix[ans] = ans_to_ix.__len__()

        return ans_to_ix, ix_to_ans

    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques.lower()
        ).replace('-', ' ').replace('/', ' ').split()
        # ['is', 'the', 'number', 'of', 'large', 'things', 'greater', 'than', 'the', 'number', 'of', 'small', 'brown', 'metallic', 'objects']

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix


    def proc_ans(self, ans, ans_to_ix):
        ans_ix = np.zeros(1, np.int64)
        ans_ix[0] = ans_to_ix[ans] # answer -> index

        return ans_ix

    def __call__(self, results):
        q = results['vqa']['question']
        a = results['vqa']['answer']

        ques_ix_iter = self.proc_ques(q, self.token_to_ix, max_token=self.max_token)

        ans_iter = np.zeros(1)
        ans_iter = self.proc_ans(a, self.ans_to_ix)

        results['question'] = ques_ix_iter
        results['answer'] = ans_iter

        return results
    

@PIPELINES.register_module()
class LoadHeight(object):
    def __init__(self, data_config, is_train=False):
        self.is_train = is_train
        self.data_config = data_config

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams
    
    def equation_plane(self, points): 
        x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
        x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
        x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1
        a = b1 * c2 - b2 * c1
        b = a2 * c1 - a1 * c2
        c = a1 * b2 - b1 * a2
        d = (- a * x1 - b * y1 - c * z1)
        return np.array([a, b, c, d])

    def get_denorm(self, ego2sen):
        ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
        ground_points_cam = np.matmul(ego2sen, ground_points_lidar.T).T
        denorm = -1 * self.equation_plane(ground_points_cam)
        return denorm

    def get_sensor2virtual(self, denorm):
        origin_vector = np.array([0, 1, 0])    
        target_vector = -1 * np.array([denorm[0], denorm[1], denorm[2]])
        target_vector_norm = target_vector / np.sqrt(target_vector[0]**2 + target_vector[1]**2 + target_vector[2]**2)       
        sita = math.acos(np.inner(target_vector_norm, origin_vector))
        n_vector = np.cross(target_vector_norm, origin_vector) 
        n_vector = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
        n_vector = n_vector.astype(np.float32)
        rot_mat, _ = cv2.Rodrigues(n_vector * sita)
        rot_mat = rot_mat.astype(np.float32)
        sensor2virtual = np.eye(4)
        sensor2virtual[:3, :3] = rot_mat
        return sensor2virtual.astype(np.float32)
    
    def get_reference_height(self, denorm):
        ref_height = np.abs(denorm[3]) / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)

        return ref_height.astype(np.float32)

    def get_inputs(self, results, flip=None, scale=None):
        sen2virs = list()
        reference_heights = list()
        cams = self.choose_cams()
        origin_img = []
        for cam in cams:
            cam_data = results['img_info'][cam]
        
            rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
            tran = torch.Tensor(cam_data['sensor2lidar_translation'])

            sen2ego = rot.new_zeros((4, 4))
            sen2ego[3, 3] = 1
            sen2ego[:3, :3] = rot
            sen2ego[:3, -1] = tran
            ego2sen = sen2ego.inverse()

            denorm = self.get_denorm(ego2sen.numpy())
            sen2vir = torch.from_numpy(self.get_sensor2virtual(denorm))
            sen2virs.append(sen2vir)

            reference_heights.append(self.get_reference_height(denorm))

        ref_heights = list()
        ref_heights.append(torch.tensor(reference_heights))
        sens2virs, ref_heights = (torch.stack(sen2virs), torch.stack(ref_heights))

        return (sens2virs, ref_heights,)


    def __call__(self, results):
        results['img_inputs'] += self.get_inputs(results)
        
        return results

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_BEVDet(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False,
                 sequential=False, aligned=False, trans_only=True, vit=False, visual=False):
        self.is_train = is_train
        self.data_config = data_config
        self.origin_transform = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),))
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])))
        self.sequential = sequential
        self.aligned = aligned
        self.trans_only = trans_only
        self.vit = vit
        self.visual = visual

        if self.vit:
            import open_clip
            _, _, self.vit_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='ckpts/open_clip_pytorch_model.bin')


    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cams = self.choose_cams()
        origin_img = []
        for cam in cams:
            cam_data = results['img_info'][cam]
            filename = cam_data['data_path']
            img = Image.open(filename)            
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
            tran = torch.Tensor(cam_data['sensor2lidar_translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                               W=img.width,
                                                                               flip=flip,
                                                                               scale=scale)
            img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            if self.vit: origin_img.append(self.vit_preprocess(img))
            if self.visual: origin_img.append(filename)
            # if self.visual: origin_img.append((filename,img)) 
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                if not type(results['adjacent']) is list:
                    filename_adjacent = results['adjacent']['cams'][cam]['data_path']
                    img_adjacent = Image.open(filename_adjacent)
                    img_adjacent = self.img_transform_core(img_adjacent,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
                else:
                    for id in range(len(results['adjacent'])):
                        filename_adjacent = results['adjacent'][id]['cams'][cam]['data_path']
                        img_adjacent = Image.open(filename_adjacent)
                        img_adjacent = self.img_transform_core(img_adjacent,
                                                               resize_dims=resize_dims,
                                                               crop=crop,
                                                               flip=flip,
                                                               rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
        
        if self.sequential:
            if self.trans_only:
                if not type(results['adjacent']) is list:
                    rots.extend(rots)
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        posi_curr = np.array(results['curr']['ego2global_translation'], dtype=np.float32)
                        posi_adj = np.array(results['adjacent']['ego2global_translation'], dtype=np.float32)
                        shift_global = posi_adj - posi_curr

                        l2e_r = results['curr']['lidar2ego_rotation']
                        e2g_r = results['curr']['ego2global_rotation']
                        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                        # shift_global = np.array([*shift_global[:2], 0.0])
                        shift_lidar = shift_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                            l2e_r_mat).T
                        trans.extend([tran + shift_lidar for tran in trans])
                    else:
                        trans.extend(trans)
                else:
                    assert False
            else:
                if not type(results['adjacent']) is list:
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        egocurr2global = np.eye(4, dtype=np.float32)
                        egocurr2global[:3,:3] = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
                        egocurr2global[:3,3] = results['curr']['ego2global_translation']

                        egoadj2global = np.eye(4, dtype=np.float32)
                        egoadj2global[:3,:3] = Quaternion(results['adjacent']['ego2global_rotation']).rotation_matrix
                        egoadj2global[:3,3] = results['adjacent']['ego2global_translation']

                        lidar2ego = np.eye(4, dtype=np.float32)
                        lidar2ego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                        lidar2ego[:3, 3] = results['curr']['lidar2ego_translation']

                        lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                                             @ egoadj2global @ lidar2ego

                        trans_new = []
                        rots_new =[]
                        for tran,rot in zip(trans, rots):
                            mat = np.eye(4, dtype=np.float32)
                            mat[:3,:3] = rot
                            mat[:3,3] = tran
                            mat = lidaradj2lidarcurr @ mat
                            rots_new.append(torch.from_numpy(mat[:3,:3]))
                            trans_new.append(torch.from_numpy(mat[:3,3]))
                        rots.extend(rots_new)
                        trans.extend(trans_new)
                    else:
                        rots.extend(rots)
                        trans.extend(trans)
                else:
                    assert False
        imgs, rots, trans, intrins, post_rots, post_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans))
        if self.vit or self.visual:
            return imgs, rots, trans, intrins, post_rots, post_trans, origin_img
        else:
            return imgs, rots, trans, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results



@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_EXT(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False,
                 sequential=False, aligned=False, trans_only=True, vit=False, visual=False,
                 aug=[0.1, 0.1, 0.1]):
        self.is_train = is_train
        self.data_config = data_config
        self.origin_transform = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),))
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])))
        self.sequential = sequential
        self.aligned = aligned
        self.trans_only = trans_only
        self.vit = vit
        self.visual = visual

        self.aug = torch.Tensor(aug)

        if self.vit:
            import open_clip
            _, _, self.vit_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='ckpts/open_clip_pytorch_model.bin')


    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate
    
    def matrix_to_euler_angles(self, matrix: torch.Tensor, convention: str) -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to Euler angles in radians.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).
            convention: Convention string of three uppercase letters.

        Returns:
            Euler angles in radians as tensor of shape (..., 3).
        """
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        # if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        #     raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        i0 = self._index_from_letter(convention[0])
        i2 = self._index_from_letter(convention[2])
        tait_bryan = i0 != i2
        if tait_bryan:
            central_angle = torch.asin(
                matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
            )
        else:
            central_angle = torch.acos(matrix[..., i0, i0])

        o = (
            self._angle_from_tan(
                convention[0], convention[1], matrix[..., i2], False, tait_bryan
            ),
            central_angle,
            self._angle_from_tan(
                convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
            ),
        )
        return torch.stack(o, -1)
                
    def euler_angles_to_matrix(self, euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
        """
        Convert rotations given as Euler angles in radians to rotation matrices.

        Args:
            euler_angles: Euler angles in radians as tensor of shape (..., 3).
            convention: Convention string of three uppercase letters from
                {"X", "Y", and "Z"}.

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
            raise ValueError("Invalid input euler angles.")
        if len(convention) != 3:
            raise ValueError("Convention must have 3 letters.")
        if convention[1] in (convention[0], convention[2]):
            raise ValueError(f"Invalid convention {convention}.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"):
                raise ValueError(f"Invalid letter {letter} in convention string.")
        matrices = [
            self._axis_angle_rotation(c, e)
            for c, e in zip(convention, torch.unbind(euler_angles, -1))
        ]
        # return functools.reduce(torch.matmul, matrices)
        return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])
                
    def _angle_from_tan(self,
        axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool) -> torch.Tensor:
        """
        Extract the first or third Euler angle from the two members of
        the matrix which are positive constant times its sine and cosine.

        Args:
            axis: Axis label "X" or "Y or "Z" for the angle we are finding.
            other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
                convention.
            data: Rotation matrices as tensor of shape (..., 3, 3).
            horizontal: Whether we are looking for the angle for the third axis,
                which means the relevant entries are in the same row of the
                rotation matrix. If not, they are in the same column.
            tait_bryan: Whether the first and third axes in the convention differ.

        Returns:
            Euler Angles in radians for each matrix in data as a tensor
            of shape (...).
        """

        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    def _axis_angle_rotation(self, axis: str, angle: torch.Tensor) -> torch.Tensor:
        """
        Return the rotation matrices for one of the rotations about an axis
        of which Euler angles describe, for each value of the angle given.

        Args:
            axis: Axis label "X" or "Y or "Z".
            angle: any shape tensor of Euler angles in radians

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)

        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either X, Y or Z.")

        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    def _index_from_letter(self, letter: str) -> int:
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2
        raise ValueError("letter must be either X, Y or Z.")    
    
    def ext_aug(self, results, img, intrin, rot, tran):
        min_num = 4
        img_src = np.array(img)
        height, width, _ = img_src.shape

        try: 
            gt_boxes = results['ann_info']['gt_bboxes_3d']
            src = gt_boxes.corners.reshape(-1,3)
        except AssertionError:
            return img, intrin, rot, tran

        combine = rot.matmul(torch.inverse(intrin))
        src_corner = (src[:,:3] - tran).matmul(torch.inverse(combine).T)
        src_corner[:, :2] = src_corner[:, :2] / src_corner[:, 2:3] 
        src_corner_wh = src_corner[:, :2]
        n,c = src_corner_wh.shape
        src_corner_wh = src_corner_wh.reshape(n//8,8,2)
        
        svalid = (src_corner_wh[:,:,0]>=0) & (src_corner_wh[:,:,0]<width) \
                  & (src_corner_wh[:,:,1]>=0) & (src_corner_wh[:,:,1]<height)
        svalid = svalid.reshape(-1,8)

        eul = self.matrix_to_euler_angles(torch.Tensor(rot), 'XYZ')
        if torch.rand(1) < 0.5:
            eul -= self.aug * torch.rand(3)
        else:
            eul += self.aug * torch.rand(3)
        rot_tar = self.euler_angles_to_matrix(eul, 'XYZ')
        
        combine = rot_tar.matmul(torch.inverse(intrin))
        tar_corner = (src[:,:3] - tran).matmul(torch.inverse(combine).T)
        tar_corner[:, :2] = tar_corner[:, :2] / tar_corner[:, 2:3] 
        tar_corner_wh = tar_corner[:, :2]
        n,c = tar_corner_wh.shape
        tar_corner_wh = tar_corner_wh.reshape(n//8,8,2)
        tvalid = (tar_corner_wh[:,:,0]>=0) & (tar_corner_wh[:,:,0]<width) \
            & (tar_corner_wh[:,:,1]>=0) & (tar_corner_wh[:,:,1]<height)
        tvalid = tvalid.reshape(-1,8)

        if len(svalid) > len(tvalid): min_len = len(tvalid)
        else:                         min_len = len(svalid)
        
        src_pts, tar_pts = [], []
        for i in range(min_len):
            for j in [0,3,4,7]:
                if svalid[i,j] and tvalid[i,j]:
                    src_pts.append(src_corner_wh[i,j])
                    tar_pts.append(tar_corner_wh[i,j])

        if len(src_pts) < min_num or len(tar_pts) < min_num:
            return img, intrin, rot, tran
        else:
            src_pts = torch.stack(src_pts, 0).numpy().reshape(-1,1,2)
            tar_pts = torch.stack(tar_pts, 0).numpy().reshape(-1,1,2)

            M, _ = cv2.findHomography(src_pts, tar_pts, cv2.RANSAC, 5.0)
            img_tar = cv2.warpPerspective(img_src, np.array(M), (width, height))

            return Image.fromarray(img_tar), intrin, rot_tar, tran
        

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cams = self.choose_cams()
        origin_img = []
        for cam in cams:
            cam_data = results['img_info'][cam]
            filename = cam_data['data_path']
            img = Image.open(filename)            
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
            tran = torch.Tensor(cam_data['sensor2lidar_translation'])

            img, intrin, rot, tran = self.ext_aug(results, img, intrin, rot, tran)

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                               W=img.width,
                                                                               flip=flip,
                                                                               scale=scale)
            img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            if self.vit: origin_img.append(self.vit_preprocess(img))
            if self.visual: origin_img.append(filename)
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                if not type(results['adjacent']) is list:
                    filename_adjacent = results['adjacent']['cams'][cam]['data_path']
                    img_adjacent = Image.open(filename_adjacent)
                    img_adjacent = self.img_transform_core(img_adjacent,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
                else:
                    for id in range(len(results['adjacent'])):
                        filename_adjacent = results['adjacent'][id]['cams'][cam]['data_path']
                        img_adjacent = Image.open(filename_adjacent)
                        img_adjacent = self.img_transform_core(img_adjacent,
                                                               resize_dims=resize_dims,
                                                               crop=crop,
                                                               flip=flip,
                                                               rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
        
        if self.sequential:
            if self.trans_only:
                if not type(results['adjacent']) is list:
                    rots.extend(rots)
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        posi_curr = np.array(results['curr']['ego2global_translation'], dtype=np.float32)
                        posi_adj = np.array(results['adjacent']['ego2global_translation'], dtype=np.float32)
                        shift_global = posi_adj - posi_curr

                        l2e_r = results['curr']['lidar2ego_rotation']
                        e2g_r = results['curr']['ego2global_rotation']
                        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                        # shift_global = np.array([*shift_global[:2], 0.0])
                        shift_lidar = shift_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                            l2e_r_mat).T
                        trans.extend([tran + shift_lidar for tran in trans])
                    else:
                        trans.extend(trans)
                else:
                    assert False
            else:
                if not type(results['adjacent']) is list:
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        egocurr2global = np.eye(4, dtype=np.float32)
                        egocurr2global[:3,:3] = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
                        egocurr2global[:3,3] = results['curr']['ego2global_translation']

                        egoadj2global = np.eye(4, dtype=np.float32)
                        egoadj2global[:3,:3] = Quaternion(results['adjacent']['ego2global_rotation']).rotation_matrix
                        egoadj2global[:3,3] = results['adjacent']['ego2global_translation']

                        lidar2ego = np.eye(4, dtype=np.float32)
                        lidar2ego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                        lidar2ego[:3, 3] = results['curr']['lidar2ego_translation']

                        lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                                             @ egoadj2global @ lidar2ego

                        trans_new = []
                        rots_new =[]
                        for tran,rot in zip(trans, rots):
                            mat = np.eye(4, dtype=np.float32)
                            mat[:3,:3] = rot
                            mat[:3,3] = tran
                            mat = lidaradj2lidarcurr @ mat
                            rots_new.append(torch.from_numpy(mat[:3,:3]))
                            trans_new.append(torch.from_numpy(mat[:3,3]))
                        rots.extend(rots_new)
                        trans.extend(trans_new)
                    else:
                        rots.extend(rots)
                        trans.extend(trans)
                else:
                    assert False
        imgs, rots, trans, intrins, post_rots, post_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans))
        if self.vit or self.visual:
            return imgs, rots, trans, intrins, post_rots, post_trans, origin_img
        else:
            return imgs, rots, trans, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_Crop(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False,
                 sequential=False, aligned=False, trans_only=True, vit=False):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])))
        self.sequential = sequential
        self.aligned = aligned
        self.trans_only = trans_only

        import open_clip
        self.vit, _, self.vit_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='ckpts/open_clip_pytorch_model.bin')


    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate    
    
    def draw_lidar_bbox3d_on_img(self,
                                 bboxes3d,
                                 valid,
                                 raw_img,
                                 lidar2img_rt,
                                 color=(0, 255, 0),
                                 thickness=1):
        """Project the 3D bbox on 2D plane and draw on input image.

        Args:
            bboxes3d (:obj:`LiDARInstance3DBoxes`):
                3d bbox in lidar coordinate system to visualize.
            raw_img (numpy.array): The numpy array of image.
            lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
                according to the camera intrinsic parameters.
            img_metas (dict): Useless here.
            color (tuple[int]): The color to draw bboxes. Default: (0, 255, 0).
            thickness (int, optional): The thickness of bboxes. Default: 1.
        """
        img = raw_img.copy()
        corners_3d = bboxes3d.corners[valid]
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate(
            [corners_3d.reshape(-1, 3),
            np.ones((num_bbox * 8, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        pts_2d = pts_4d @ lidar2img_rt.T

        pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
        
        crop_imgs = list()
        for i, v in enumerate(imgfov_pts_2d):
            box_rect = np.array([min(v[:,0]), min(v[:,1]) ,max(v[:,0]), max(v[:,1])])
            if 0 < box_rect[0] and box_rect[0] < img.width and \
               0 < box_rect[1] and box_rect[1] < img.height and \
               0 < box_rect[2] and box_rect[2] < img.width and \
               0 < box_rect[3] and box_rect[3] < img.height:
                crop_img = img.crop(box_rect)
                crop_imgs.append(self.vit_preprocess(crop_img))
        return torch.stack(crop_imgs)
        

    def get_inputs(self, results, flip=None, scale=None):
        gt_bboxes_3d = results['gt_bboxes_3d']
        valid = np.where(results['gt_labels_3d'] == 0)

        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cams = self.choose_cams()
        crop_imgs = []
        ori_imgs =[]
        for cam in cams:
            cam_data = results['img_info'][cam]
            filename = cam_data['data_path']
            img = Image.open(filename)            
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            lidar2cam_r = np.linalg.inv(cam_data['sensor2lidar_rotation'])
            lidar2cam_t = cam_data[
                'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam_data['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)

            try:
                crop_imgs.append(self.draw_lidar_bbox3d_on_img(gt_bboxes_3d, valid, img, lidar2img_rt))
            except RuntimeError:
                pass
            ori_imgs.append(self.vit_preprocess(img))

            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
            tran = torch.Tensor(cam_data['sensor2lidar_translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                               W=img.width,
                                                                               flip=flip,
                                                                               scale=scale)
            img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                if not type(results['adjacent']) is list:
                    filename_adjacent = results['adjacent']['cams'][cam]['data_path']
                    img_adjacent = Image.open(filename_adjacent)
                    img_adjacent = self.img_transform_core(img_adjacent,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
                else:
                    for id in range(len(results['adjacent'])):
                        filename_adjacent = results['adjacent'][id]['cams'][cam]['data_path']
                        img_adjacent = Image.open(filename_adjacent)
                        img_adjacent = self.img_transform_core(img_adjacent,
                                                               resize_dims=resize_dims,
                                                               crop=crop,
                                                               flip=flip,
                                                               rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            if self.trans_only:
                if not type(results['adjacent']) is list:
                    rots.extend(rots)
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        posi_curr = np.array(results['curr']['ego2global_translation'], dtype=np.float32)
                        posi_adj = np.array(results['adjacent']['ego2global_translation'], dtype=np.float32)
                        shift_global = posi_adj - posi_curr

                        l2e_r = results['curr']['lidar2ego_rotation']
                        e2g_r = results['curr']['ego2global_rotation']
                        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                        # shift_global = np.array([*shift_global[:2], 0.0])
                        shift_lidar = shift_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                            l2e_r_mat).T
                        trans.extend([tran + shift_lidar for tran in trans])
                    else:
                        trans.extend(trans)
                else:
                    assert False
            else:
                if not type(results['adjacent']) is list:
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        egocurr2global = np.eye(4, dtype=np.float32)
                        egocurr2global[:3,:3] = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
                        egocurr2global[:3,3] = results['curr']['ego2global_translation']

                        egoadj2global = np.eye(4, dtype=np.float32)
                        egoadj2global[:3,:3] = Quaternion(results['adjacent']['ego2global_rotation']).rotation_matrix
                        egoadj2global[:3,3] = results['adjacent']['ego2global_translation']

                        lidar2ego = np.eye(4, dtype=np.float32)
                        lidar2ego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                        lidar2ego[:3, 3] = results['curr']['lidar2ego_translation']

                        lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                                             @ egoadj2global @ lidar2ego

                        trans_new = []
                        rots_new =[]
                        for tran,rot in zip(trans, rots):
                            mat = np.eye(4, dtype=np.float32)
                            mat[:3,:3] = rot
                            mat[:3,3] = tran
                            mat = lidaradj2lidarcurr @ mat
                            rots_new.append(torch.from_numpy(mat[:3,:3]))
                            trans_new.append(torch.from_numpy(mat[:3,3]))
                        rots.extend(rots_new)
                        trans.extend(trans_new)
                    else:
                        rots.extend(rots)
                        trans.extend(trans)
                else:
                    assert False
        imgs, rots, trans, intrins, post_rots, post_trans, crop_imgs, ori_imgs = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans), torch.cat(crop_imgs, dim=0),
                                                             torch.stack(ori_imgs))
        
        return imgs, rots, trans, intrins, post_rots, post_trans, crop_imgs, ori_imgs

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in \
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int): The max possible cat_id in input segmentation mask.
            Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids. \
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points. \
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 dummy=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.dummy = dummy

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        if self.dummy:
            points = torch.ones([1,self.load_dim] ,dtype=torch.float32)
            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=None)
            results['points'] = points
            return results
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str

@PIPELINES.register_module()
class LoadPointsFromFile_temp(object):
    """Load Points From File.

    Load sunrgbd and scannet points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int]): Which dimensions of the points to be used.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 dummy=False,
                 data_root='./data/nuscenes/',
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.dummy = dummy
        self.data_root = data_root

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(self.data_root+pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data. \
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        if self.dummy:
            points = torch.ones([1,self.load_dim] ,dtype=torch.float32)
            points_class = get_points_type(self.coord_type)
            points = points_class(
                points, points_dim=points.shape[-1], attribute_dims=None)
            results['points'] = points
            return results
        pts_filename = results['adjacent']['lidar']['filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points_prev = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points_prev'] = points_prev
    
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps_temp(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 data_root='./data/nuscenes/',
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.data_root = data_root

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        points = results['points_prev']
        points.tensor[:, 4] = 0
        sweep_points_list = [points]
        ts = results['adjacent']['timestamp']
        if self.pad_empty_sweeps and len(results['adjacent']['lidar']['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['adjacent']['lidar']['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['adjacent']['lidar']['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['adjacent']['lidar']['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['adjacent']['lidar']['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, 4] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points_prev'] = points
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.long)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.long)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_waymo2(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, img_scale=None, color_type='unchanged', 
                    project_pts_to_img_depth=False, is_train=False, data_config=None, aligned=False,
                    cam_depth_range=[4.0, 45.0, 1.0],
                    constant_std=0.5):
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])))
        self.aligned = aligned
        self.to_float32 = to_float32
        self.img_scale = img_scale
        self.color_type = color_type
        self.project_pts_to_img_depth = project_pts_to_img_depth
        self.cam_depth_range = cam_depth_range
        self.constant_std=constant_std
        self.is_train = is_train
        self.data_config = data_config

    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.size[1] != self.img_scale[1]:
            result = Image.new(img.mode, (1920, 1280), (0, 0, 0))
            result.paste(img, (0, 0)) 
        return result

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def __call__(self, results, flip=None, scale=None):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        now_name = results['img_filename']

        filename_splits = now_name[0].split("/")
        file_index = filename_splits[5].split(".")[0]
        calib_path = f"{filename_splits[0]}/{filename_splits[1]}/{filename_splits[2]}/{filename_splits[3]}/calib/{file_index}.txt"
        with open(calib_path, 'r') as f: lines = f.readlines()

        for filename in now_name:
            with Image.open(filename) as img:
                if img.size[1] != 1280: img = self.pad(img)
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                filename_splits = filename.split("/")

                cam_idx = int(filename_splits[-2].split("_")[-1])

                intrin = torch.Tensor(np.array([float(info) for info in lines[cam_idx].split(' ')[1:13]]).reshape([3, 4])[:3,:3])
                extrinsic_list = np.array([float(info) for info in lines[0+6].split(' ')[1:13]]).reshape([3, 4])
                extrinsic_list = extend_matrix(extrinsic_list)
                extrinsic_list = np.linalg.inv(extrinsic_list)
                rot = torch.Tensor(extrinsic_list[:3,:3])
                tran = torch.Tensor(extrinsic_list[:3,3].reshape(-1))

                # augmentation (resize, crop, horizontal flip, rotate)
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                                W=img.width,
                                                                                flip=flip,
                                                                                scale=scale)

                img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

                # for convenience, make augmentation matrices 3x3
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                imgs.append(self.normalize_img(img))
                intrins.append(intrin)
                rots.append(rot)
                trans.append(tran)
                post_rots.append(post_rot)
                post_trans.append(post_tran)

        imgs, rots, trans, intrins, post_rots, post_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans))

        results['img_inputs'] = imgs, rots, trans, intrins, post_rots, post_trans

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (to_float32={}, color_type='{}')".format(
            self.__class__.__name__, self.to_float32, self.color_type)



@PIPELINES.register_module()
class LoadMultiViewImageFromFiles_bevformer(LoadMultiViewImageFromFiles):
    def __call__(self, results):
        filename = results['img_filename']
        img = np.stack(
            [mmcv.imread(name.replace('cvpr/', ''), self.color_type) for name in filename], axis=-1) # 240512
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results