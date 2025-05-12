# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import numpy as np
import os
import pandas as pd
import tempfile
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from lyft_dataset_sdk.utils.data_classes import Box as LyftBox
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
from pyquaternion import Quaternion

from mmdet3d.core.evaluation.lyft_eval import lyft_eval
from mmdet.datasets import DATASETS
from ..core import show_result, show_multi_modality_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
from .pipelines import Compose

map_vehicle_to_car_shift = {
    'bicycle': 'ignore',
    'bus': 'ignore',
    'car': 'car',
    'emergency_vehicle': 'ignore',
    'motorcycle': 'ignore',
    'other_vehicle': 'ignore',
    'pedestrian': 'ignore',
    'truck': 'ignore',
    'animal': 'ignore'
}

map_vehicle_to_car = {
    'bicycle': 'bicycle',
    'bus': 'ignore',
    'car': 'car',
    'emergency_vehicle': 'ignore',
    'motorcycle': 'ignore',
    'other_vehicle': 'ignore',
    'pedestrian': 'pedestrian',
    'truck': 'ignore',
    'animal': 'ignore'
}

    
@DATASETS.register_module()
class LyftDataset(Custom3DDataset):
    r"""Lyft Dataset.

    This class serves as the API for experiments on the Lyft Dataset.

    Please refer to
    `<https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """  # noqa: E501
    NameMapping = {
        'bicycle': 'bicycle',
        'bus': 'bus',
        'car': 'car',
        'emergency_vehicle': 'emergency_vehicle',
        'motorcycle': 'motorcycle',
        'other_vehicle': 'other_vehicle',
        'pedestrian': 'pedestrian',
        'truck': 'truck',
        'animal': 'animal'
    }
    DefaultAttribute = {
        'car': 'is_stationary',
        'truck': 'is_stationary',
        'bus': 'is_stationary',
        'emergency_vehicle': 'is_stationary',
        'other_vehicle': 'is_stationary',
        'motorcycle': 'is_stationary',
        'bicycle': 'is_stationary',
        'pedestrian': 'is_stationary',
        'animal': 'is_stationary'
    }
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=False,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_lyft',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 speed_mode='abs_dis',
                 max_interval=3,
                 min_interval=0,
                 prev_only=False,
                 next_only=False,
                 test_adj = 'prev',
                 fix_direction=False,
                 test_adj_ids=None,
                 visual=False,
                 lidar_vis=False,
                 domain_shift=False,
                 ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.with_velocity  = with_velocity
        self.eval_version   = eval_version
        self.visual         = visual
        self.lidar_vis      = lidar_vis
        self.domain_shift   = domain_shift

        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)

        self.CLASSES = ()
        for cls in classes:
            self.CLASSES += (cls,)

        if not self.domain_shift: 
            self.map_vehicle_to_car = map_vehicle_to_car
        else: 
            self.map_vehicle_to_car = map_vehicle_to_car_shift  

        for idx, cls in self.map_vehicle_to_car.items():                
            if cls == 'ignore':
                self.eval_detection_configs.class_range.pop(idx)


        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        self.img_info_prototype = img_info_prototype

        self.speed_mode = speed_mode
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.prev_only = prev_only
        self.next_only = next_only
        self.test_adj = test_adj
        self.fix_direction = fix_direction
        self.test_adj_ids = test_adj_ids
        print(self.CLASSES)

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            try: name = self.map_vehicle_to_car[name]
            except: pass
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
            else:
                print(name)
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos


    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - sweeps (list[dict]): infos of sweeps
                - timestamp (float): sample timestamp
                - img_filename (str, optional): image filename
                - lidar2img (list[np.ndarray], optional): transformations \
                    from lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]

        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))
            elif self.img_info_prototype == 'bevdet':
                input_dict.update(dict(img_info=info['cams']))        
            elif self.img_info_prototype == 'bevdet_sequential':
                if info ['prev'] is None or info['next'] is None:
                    adjacent= 'prev' if info['next'] is None else 'next'
                else:
                    if self.prev_only or self.next_only:
                        adjacent = 'prev' if self.prev_only else 'next'
                    elif self.test_mode:
                        adjacent = self.test_adj
                    else:
                        adjacent = np.random.choice(['prev', 'next'])
                if type(info[adjacent]) is list:
                    if self.test_mode:
                        if self.test_adj_ids is not None:
                            info_adj=[]
                            select_id = self.test_adj_ids
                            for id_tmp in select_id:
                                id_tmp = min(id_tmp, len(info[adjacent])-1)
                                info_adj.append(info[adjacent][id_tmp])
                        else:
                            select_id = min((self.max_interval+self.min_interval)//2,
                                            len(info[adjacent])-1)
                            info_adj = info[adjacent][select_id]
                    else:
                        if len(info[adjacent])<= self.min_interval:
                            select_id = len(info[adjacent])-1
                        else:
                            select_id = np.random.choice([adj_id for adj_id in range(
                                min(self.min_interval,len(info[adjacent])),
                                min(self.max_interval,len(info[adjacent])))])
                        info_adj = info[adjacent][select_id]
                else:
                    info_adj = info[adjacent]
                input_dict.update(dict(img_info=info['cams'],
                                       curr=info,
                                       adjacent=info_adj,
                                       adjacent_type=adjacent))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            # if self.img_info_prototype == 'bevdet_sequential':
            #     bbox = input_dict['ann_info']['gt_bboxes_3d'].tensor
            #     if 'abs' in self.speed_mode:
            #         bbox[:, 7:9] = bbox[:, 7:9] + torch.from_numpy(info['velo']).view(1,2).to(bbox)
            #     if input_dict['adjacent_type'] == 'next' and not self.fix_direction:
            #         bbox[:, 7:9] = -bbox[:, 7:9]
            #     if 'dis' in self.speed_mode:
            #         time = abs(input_dict['timestamp'] - 1e-6 * input_dict['adjacent']['timestamp'])
            #         bbox[:, 7:9] = bbox[:, 7:9] * time
            #     input_dict['ann_info']['gt_bboxes_3d'] = LiDARInstance3DBoxes(bbox,
            #                                                                   box_dim=bbox.shape[-1],
            #                                                                   origin=(0.5, 0.5, 0))
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]

        # filter out bbox containing no points

        gt_bboxes_3d = info['gt_boxes']
        gt_names_3d = info['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            try: cat = self.map_vehicle_to_car[cat]
            except: pass
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = np.zeros([info['gt_boxes'].shape[0], 2])
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the lyft box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d
        )
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        lyft_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_lyft_box(det, self.data_infos[sample_id],
                                       self.speed_mode, self.img_info_prototype,
                                       self.max_interval, self.test_adj,
                                       self.fix_direction,
                                       self.test_adj_ids)
            sample_token = self.data_infos[sample_id]['token']
            
            boxes = lidar_lyft_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs)
            
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                name = self.map_vehicle_to_car[name]
                lyft_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    score=box.score,
                    attribute_name="")
                annos.append(lyft_anno)
            lyft_annos[sample_token] = annos
        lyft_submissions = {
            'meta': self.modality,
            'results': lyft_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_lyft.json')
        print('Results writes to', res_path)
        mmcv.dump(lyft_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in Lyft protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from mmdet3d.datasets.LyftEval import LyftEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        lyft = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=True)
        eval_set_map = {
            'v1.0-trainval': 'val',
            'v1.01-train': 'val',
        }

        nusc_eval = LyftEval(
            lyft,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            visual=self.visual,
            lidar_vis=self.lidar_vis,
            metas=self.metas,
            data_infos=self.data_infos,
            domain_shift=self.domain_shift)
        nusc_eval.main(render_curves=False)


        '''
        metrics = lyft_eval(lyft, self.data_root, result_path,
                            eval_set_map[self.version], output_dir, logger)
        '''

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_Lyft'

        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        '''
        for i, name in enumerate(metrics['class_names']):
            AP = float(metrics['mAPs_cate'][i])
            detail[f'{metric_prefix}/{name}_AP'] = AP

        detail[f'{metric_prefix}/mAP'] = metrics['Final mAP']
        '''        
        # detail['mAP'] = metrics['mean_ap'] # 240502, for save_best while training
        detail['mAP'] = metrics['nd_score'] # 240509, for save_best while training
        return detail

    def format_results(self, results, jsonfile_prefix=None, csv_savepath=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            csv_savepath (str | None): The path for saving csv files.
                It includes the file path and the csv filename,
                e.g., "a/b/filename.csv". If not specified,
                the result will not be converted to csv file.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on Lyft
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in Lyft protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            csv_savepath (str | None): The path for saving csv files.
                It includes the file path and the csv filename,
                e.g., "a/b/filename.csv". If not specified,
                the result will not be converted to csv file.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Evaluation results.
        """
        
        '''
        self.this_pipeline = [    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=["car",],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])]

        self.show(results, "/home/sr5/rsu/users/kuaicv/cvpr/outputs", show=True, pipeline=self.this_pipeline)
        raise 0

        '''

        self.metas=None
        if self.visual:
            self.metas = []
            for idx, result in enumerate(results):
                self.metas.append(result['img_metas'])

                results[idx].pop("img_metas")

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print(f'Evaluating bboxes of {name}')
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        print(pipeline)
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show, snapshot=True)

    def json2csv(self, json_path, csv_savepath):
        """Convert the json file to csv format for submission.

        Args:
            json_path (str): Path of the result json file.
            csv_savepath (str): Path to save the csv file.
        """
        results = mmcv.load(json_path)['results']
        sample_list_path = osp.join(self.data_root, 'sample_submission.csv')
        data = pd.read_csv(sample_list_path)
        Id_list = list(data['Id'])
        pred_list = list(data['PredictionString'])
        cnt = 0
        print('Converting the json to csv...')
        for token in results.keys():
            cnt += 1
            predictions = results[token]
            prediction_str = ''
            for i in range(len(predictions)):
                prediction_str += \
                    str(predictions[i]['score']) + ' ' + \
                    str(predictions[i]['translation'][0]) + ' ' + \
                    str(predictions[i]['translation'][1]) + ' ' + \
                    str(predictions[i]['translation'][2]) + ' ' + \
                    str(predictions[i]['size'][0]) + ' ' + \
                    str(predictions[i]['size'][1]) + ' ' + \
                    str(predictions[i]['size'][2]) + ' ' + \
                    str(Quaternion(list(predictions[i]['rotation']))
                        .yaw_pitch_roll[0]) + ' ' + \
                    predictions[i]['name'] + ' '
            prediction_str = prediction_str[:-1]
            idx = Id_list.index(token)
            pred_list[idx] = prediction_str
        df = pd.DataFrame({'Id': Id_list, 'PredictionString': pred_list})
        mmcv.mkdir_or_exist(os.path.dirname(csv_savepath))
        df.to_csv(csv_savepath, index=False)


def output_to_lyft_box(detection, info, speed_mode,
                       img_info_prototype, max_interval, test_adj, fix_direction,
                       test_adj_ids):
    """Convert the output to the box class in the Lyft.

    Args:
        detection (dict): Detection results.

    Returns:
        list[:obj:`LyftBox`]: List of standard LyftBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    
    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    velocity_all = torch.zeros([box3d.tensor.shape[0], 2])#box3d.tensor[:, 7:9]
    if img_info_prototype =='bevdet_sequential':
        if info['prev'] is None or info['next'] is None:
            adjacent = 'prev' if info['next'] is None else 'next'
        else:
            adjacent = test_adj
        if adjacent == 'next' and not fix_direction:
            velocity_all = -velocity_all
        if type(info[adjacent]) is list:
            select_id = min(max_interval // 2, len(info[adjacent]) - 1)
            # select_id = min(2, len(info[adjacent]) - 1)
            info_adj = info[adjacent][select_id]
        else:
            info_adj = info[adjacent]
        if 'dis' in speed_mode and test_adj_ids is None:
            time = abs(1e-6 * info['timestamp'] - 1e-6 * info_adj['timestamp'])
            velocity_all = velocity_all / time

    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*velocity_all[i,:], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_lyft_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`LyftBox`]): List of predicted LyftBoxes.

    Returns:
        list: List of standard LyftBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        try: det_range = cls_range_map[classes[box.label]]
        except: continue
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
