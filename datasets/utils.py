# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

# yapf: disable
from mmdet3d.datasets.pipelines import (Collect3D, DefaultFormatBundle3D,
                                        LoadAnnotations3D,
                                        LoadImageFromFileMono3D,
                                        LoadMultiViewImageFromFiles,
                                        LoadPointsFromFile,
                                        LoadPointsFromMultiSweeps,
                                        MultiScaleFlipAug3D,
                                        PointSegClassMapping)
# yapf: enable
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadImageFromFile

import numpy as np
import os
import cv2
from .LyftEval import LB, check_point_in_img, lidar2img, depth2color 
from .CarlaEval import draw_boxes_indexes_bev, draw_boxes_indexes_img_view

def is_loading_function(transform):
    """Judge whether a transform function is a loading function.

    Note: `MultiScaleFlipAug3D` is a wrapper for multiple pipeline functions,
    so we need to search if its inner transforms contain any loading function.

    Args:
        transform (dict | :obj:`Pipeline`): A transform config or a function.

    Returns:
        bool | None: Whether it is a loading function. None means can't judge.
            When transform is `MultiScaleFlipAug3D`, we return None.
    """
    # TODO: use more elegant way to distinguish loading modules
    loading_functions = (LoadImageFromFile, LoadPointsFromFile,
                         LoadAnnotations3D, LoadMultiViewImageFromFiles,
                         LoadPointsFromMultiSweeps, DefaultFormatBundle3D,
                         Collect3D, LoadImageFromFileMono3D,
                         PointSegClassMapping)
    if isinstance(transform, dict):
        obj_cls = PIPELINES.get(transform['type'])
        if obj_cls is None:
            return False
        if obj_cls in loading_functions:
            return True
        if obj_cls in (MultiScaleFlipAug3D, ):
            return None
    elif callable(transform):
        if isinstance(transform, loading_functions):
            return True
        if isinstance(transform, MultiScaleFlipAug3D):
            return None
    return False


def get_loading_pipeline(pipeline):
    """Only keep loading image, points and annotations related configuration.

    Args:
        pipeline (list[dict] | list[:obj:`Pipeline`]):
            Data pipeline configs or list of pipeline functions.

    Returns:
        list[dict] | list[:obj:`Pipeline`]): The new pipeline list with only
            keep loading image, points and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='Resize',
        ...         img_scale=[(640, 192), (2560, 768)], keep_ratio=True),
        ...    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
        ...    dict(type='PointsRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='ObjectRangeFilter',
        ...         point_cloud_range=point_cloud_range),
        ...    dict(type='PointShuffle'),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadPointsFromFile',
        ...         coord_type='LIDAR', load_dim=4, use_dim=4),
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations3D',
        ...         with_bbox=True, with_label_3d=True),
        ...    dict(type='DefaultFormatBundle3D', class_names=class_names),
        ...    dict(type='Collect3D',
        ...         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ...    ]
        >>> assert expected_pipelines ==\
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline = []
    for transform in pipeline:
        is_loading = is_loading_function(transform)
        if is_loading is None:  # MultiScaleFlipAug3D
            # extract its inner pipeline
            if isinstance(transform, dict):
                inner_pipeline = transform.get('transforms', [])
            else:
                inner_pipeline = transform.transforms.transforms
            loading_pipeline.extend(get_loading_pipeline(inner_pipeline))
        elif is_loading:
            loading_pipeline.append(transform)
    assert len(loading_pipeline) > 0, \
        'The data pipeline in your config file must include ' \
        'loading step.'
    return loading_pipeline


def extract_result_dict(results, key):
    """Extract and return the data corresponding to key in result dict.

    ``results`` is a dict output from `pipeline(input_dict)`, which is the
        loaded data from ``Dataset`` class.
    The data terms inside may be wrapped in list, tuple and DataContainer, so
        this function essentially extracts data from these wrappers.

    Args:
        results (dict): Data loaded using pipeline.
        key (str): Key of the desired data.

    Returns:
        np.ndarray | torch.Tensor | None: Data term.
    """
    if key not in results.keys():
        return None
    # results[key] may be data or list[data] or tuple[data]
    # data may be wrapped inside DataContainer
    data = results[key]
    if isinstance(data, (list, tuple)):
        data = data[0]
    if isinstance(data, mmcv.parallel.DataContainer):
        data = data._data
    return data

def get_matrix(transform):
    x, y, z, yaw, pitch, roll = transform
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = x 
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return np.array(matrix)
    

def gt_vis(data_infos, n, vis_dir='vis/lyft'):
    canva_size = 800
    show_range = 50
    color_map = {0: (255, 255, 0), 1: (0, 255, 255)}
    scale_factor = 1        
    img_w, img_h = 400, 300

    for n_i in range(n) : 
        data_info = data_infos[-n_i]    
        car_idx = np.where(data_info["gt_names"] == "car")
        if len(data_info['gt_boxes'][car_idx]) == 0 :
            print('no car')
            continue;
        corners_lidar = LB(data_info['gt_boxes'][car_idx], origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
        imgs = []
        
        #['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        for view in views:
            img = cv2.imread(data_info['cams'][view]['data_path'])
            # draw instances
            corners_img, valid = lidar2img(corners_lidar, data_info['cams'][view])
            valid = np.logical_and(valid, check_point_in_img(corners_img,
                                                                img.shape[0],
                                                                img.shape[1]))
            valid = valid.reshape(-1, 8)
            corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)
            for aid in range(valid.shape[0]):
                for index in draw_boxes_indexes_img_view:
                    if valid[aid,index[0]] and valid[aid, index[1]]:
                        cv2.line(img,
                                tuple(corners_img[aid, index[0]]),
                                tuple(corners_img[aid, index[1]]),
                                color=color_map[0],
                                thickness=4)
            resized = cv2.resize(img, (img_w, img_h))
            imgs.append(resized)
        
        # bird-eye-view
        canvas = np.zeros((int(canva_size), int(canva_size), 3), dtype=np.uint8)
        # draw lidar points
        lidar_points = np.fromfile(data_info['lidar_path'], dtype=np.float32)
        lidar_points = lidar_points.reshape(-1, 5)[:,:3]
        if 'lidar2new' in data_info:
            new_points = np.ones((len(lidar_points), 4))
            new_points[:, :3] = lidar_points
            out = np.dot(data_info['lidar2new'], new_points.T)
            lidar_points = out.T[:, :3]

        lidar_points[:, 1] = -lidar_points[:, 1]
        lidar_points[:, :2] = (lidar_points[:, :2] + show_range) / \
                                show_range / 2.0 * canva_size
        for p in lidar_points:
            if check_point_in_img(p.reshape(1,3),
                                    canvas.shape[1],
                                    canvas.shape[0])[0]:
                color = depth2color(p[2])
                cv2.circle(canvas, (int(p[0]), int(p[1])),
                            radius=0, color=color, thickness=1)

        # draw instances
        corners_lidar = corners_lidar.reshape(-1, 8, 3)
        corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
        bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
        bottom_corners_bev = (bottom_corners_bev + show_range) / \
                                show_range / 2.0 * canva_size
        bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
        center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
        head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
        canter_canvas = (center_bev + show_range) / show_range / 2.0 * \
                        canva_size
        center_canvas = canter_canvas.astype(np.int32)
        head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
        head_canvas = head_canvas.astype(np.int32)

        for rid in range(len(corners_lidar)) :
            for index in draw_boxes_indexes_bev:
                cv2.line(canvas,
                            tuple(bottom_corners_bev[rid, index[0]]),
                            tuple(bottom_corners_bev[rid, index[1]]),
                            color=color_map[0],
                            thickness=1)
            cv2.line(canvas, tuple(center_canvas[rid]), tuple(head_canvas[rid]), color_map[0],
                        1, lineType=8)            
        cv2.circle(canvas, (int(canva_size/2), int(canva_size/2)), 2, (0, 0, 255), -1)

        # fuse image-view and bev
        img = np.zeros((img_h * 2 + canva_size * scale_factor, img_w * 3, 3),
                        dtype=np.uint8)
        img[:img_h, :, :] = np.concatenate(imgs[:3], axis=1)
        img_back = np.concatenate([imgs[3][:, ::-1, :],
                                    imgs[4][:, ::-1, :],
                                    imgs[5][:, ::-1, :]], axis=1)
        img[img_h + canva_size * scale_factor:, :, :] = img_back
        img = cv2.resize(img, (int(img_w / scale_factor * 3),
                                int(img_h / scale_factor * 2 + canva_size)))
        w_begin = int((img_w * 3 / scale_factor - canva_size) // 2)
        img[int(img_h / scale_factor):int(img_h / scale_factor) + canva_size, w_begin: w_begin + canva_size, :] = canvas

        out_path = os.path.join(vis_dir, "%s_%i.jpg" % (data_info['token'], n_i))
        print(out_path)
        cv2.imwrite(out_path, img)