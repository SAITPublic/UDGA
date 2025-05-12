# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

import json
from typing import Dict, Tuple
import torch
import numpy as np
import tqdm
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from nuscenes.utils.splits import create_splits_scenes
from ..core.bbox import LiDARInstance3DBoxes, get_box_type, CameraInstance3DBoxes

def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = True) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if not verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta

def load_waymo_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = True) \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if not verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    box_type_3d, box_mode_3d = get_box_type('LiDAR')

    for sample_token in all_results.sample_tokens:
        temp_boxes = all_results.boxes[str(sample_token)]

        for temp_box in temp_boxes:
            pred_boxes = []
            pred_box = temp_box.serialize()
            pred_boxes.append(list(pred_box["translation"])+list(pred_box["size"]))
            pred_boxes[0].append(Quaternion(pred_box["rotation"]).yaw_pitch_roll[0])

            boxes = LiDARInstance3DBoxes(np.array(pred_boxes, dtype=np.float32))
            temp_box.translation = (boxes.tensor[0,0], boxes.tensor[0,1], boxes.tensor[0,2])
            temp_box.size = boxes.tensor[0,3:6]
            temp_box.rotation = Quaternion(axis=[0, 0, 1], radians=boxes.tensor[0,6]).elements

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta

import os
def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def load_waymo_gt(data_infos, data_root, box_cls, verbose: bool = False, domain_shift: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    all_annotations = EvalBoxes()
    tracking_id_set = set()
    pose_dict = dict()
    for data_infos_idx in tqdm.tqdm(range(len(data_infos)), leave=verbose):

        info = data_infos[data_infos_idx]
        sample_annotation_tokens = info['annos']
        pose_dict[str(info["image"]["image_idx"])] = info['pose']
        rect = info['calib']['R0_rect'].astype(np.float32)

        now_name = "image_0"
        image_name_list = ["image_0", "image_1", "image_2", "image_3", "image_4"]
        cam_type_list = ["P0", "P1", "P2", "P3", "P4"]
        Trv2c_list = []
        img_filename = os.path.join(data_root, info['image']['image_path'])

        filename_splits = img_filename.split("/")
        file_index = filename_splits[5].split(".")[0]
        calib_path = f"{filename_splits[0]}/{filename_splits[1]}/{filename_splits[2]}/{filename_splits[3]}/calib/{file_index}.txt"
        with open(calib_path, 'r') as f: lines = f.readlines()

        intrins, extrins_r, extints_t = [], [], []
        for cam_idx, cam_type in enumerate(cam_type_list):
            img_filename = img_filename.replace(now_name, image_name_list[cam_idx])
            now_name = image_name_list[cam_idx]

            filename_splits = img_filename.split("/")

            cam_idx = int(filename_splits[-2].split("_")[-1])

            extrinsic_list = np.array([float(info) for info in lines[cam_idx+6].split(' ')[1:13]]).reshape([3, 4])
            Trv2c = _extend_matrix(extrinsic_list).astype(np.float32)
            Trv2c_list.append(Trv2c)

            intrin = torch.Tensor(np.array([float(info) for info in lines[cam_idx].split(' ')[1:13]]).reshape([3, 4])[:3,:3])
            extrinsic_list = np.array([float(info) for info in lines[cam_idx+6].split(' ')[1:13]]).reshape([3, 4])
            extrinsic_list = _extend_matrix(extrinsic_list)
            extrinsic_list = np.linalg.inv(extrinsic_list)
            ext_r = torch.Tensor(extrinsic_list[:3,:3])
            ext_t = torch.Tensor(extrinsic_list[:3,3].reshape(-1))
            intrins.append(intrin)
            extrins_r.append(ext_r)
            extints_t.append(ext_t)
        intrins, extrins_r, extints_t = torch.stack(intrins,0), torch.stack(extrins_r,0), torch.stack(extints_t,0)

        sample_boxes = []
        for anno_idx in range(len(sample_annotation_tokens['bbox'])):
            if np.sum(sample_annotation_tokens['bbox'][anno_idx])==0: continue

            if box_cls == DetectionBox:
                cam_num = sample_annotation_tokens['camera_id'][anno_idx]
                T_velo_to_front_cam = Trv2c_list[int(cam_num)].astype(np.float32)

                box_type_3d, box_mode_3d = get_box_type("LiDAR")
                # Get label name in detection task and filter unused labels.
                detection_name = sample_annotation_tokens['name'][anno_idx]

                loc = sample_annotation_tokens['location'][anno_idx].copy().reshape(-1, 3)
                dim = sample_annotation_tokens['dimensions'][anno_idx].copy().reshape(-1, 3)
                rot = sample_annotation_tokens['rotation_y'][anno_idx].copy().reshape(-1, 1)
                gt_boxes = np.concatenate([loc, dim, rot], axis=-1)
                corners_lidar_gt = CameraInstance3DBoxes(gt_boxes).convert_to(
                    box_mode_3d, np.linalg.inv(rect @ Trv2c_list[0]))

                from .waymo_dataset import map_vehicle_to_car, map_vehicle_to_car_shift
                if not domain_shift:
                    detection_name = map_vehicle_to_car[detection_name]
                else:
                    detection_name = map_vehicle_to_car_shift[detection_name]
                    if detection_name == "ignore": continue

                loc = corners_lidar_gt.tensor[0,:3].numpy()
                dim = corners_lidar_gt.tensor[0,3:6].numpy()
                rot = corners_lidar_gt.tensor[0,6].numpy()

                # Kitti to Waymo
                loc[2] += dim[2]/2
                # valid = fov_filter(loc, intrins, extrins_r, extrins_t)
                # if not valid: continue

                gt_boxes = []
                gt_boxes.append(list(loc) + list(dim))
                gt_boxes[0].append(rot)
                # gt_boxes[0].append(Quaternion(axis=[0, 0, 1], radians=corners_lidar_gt.tensor[0,6]).yaw_pitch_roll[0])

                sample_boxes.append(
                    box_cls(
                        sample_token=str(info["image"]["image_idx"]),
                        translation=gt_boxes[0][:3],
                        size=gt_boxes[0][3:6],
                        rotation=Quaternion(axis=[0, 0, 1], radians=gt_boxes[0][6]).elements,
                        # rotation=gt_boxes[0][6],
                        velocity=np.zeros([2]),
                        num_pts=int(sample_annotation_tokens['num_points_in_gt'][anno_idx]),
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=''#attribute_name
                    )
                )

                #sample_boxes.append(
                #    box_cls(
                #        sample_token=str(info["image"]["image_idx"]),
                #        translation=boxes.tensor[0,:3].numpy(),
                #        size=boxes.tensor[0,3:6].numpy(),
                #        rotation=Quaternion(axis=[0, 0, 1], radians=boxes.tensor[0,6]).elements,
                #        velocity=np.zeros([2]),
                #        num_pts=int(sample_annotation_tokens['num_points_in_gt'][anno_idx]),
                #        detection_name=detection_name,
                #        detection_score=-1.0,  # GT samples do not have a score.
                #        attribute_name=''#attribute_name
                #    )
                #)


            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(str(info["image"]["image_idx"]), sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations, pose_dict

def fov_filter(box, intrins, rots, trans):
    for intrin, rot, tran in zip(intrins, rots, trans):
        loc = torch.from_numpy(box)
        combine = torch.inverse(rot.matmul(torch.inverse(intrin)))
        loc = (loc - tran).matmul(combine.T)
        loc[:2] /= loc[2:3]
        
        valid = (loc[0] >= 0) & (loc[0] < 1920) & (loc[1] >= 0) & (loc[1] < 1280) 
        if valid: break 
    return valid
        
# 240506
# def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False, domain_shift: bool = False) -> EvalBoxes:
def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False, domain_shift: bool = False, tokens=None) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    # sample_tokens_all = [s['token'] for s in nusc.sample]
    sample_tokens_all = [s['token'] for s in nusc.sample] if tokens is None else tokens # 240506    
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    if nusc.version == "v1.01-train":
        try:
            with open("data/lyft/val.txt") as f:
                splits = dict()
                splits[eval_split] = f.readlines()
                splits[eval_split] = [line.rstrip('\n') for line in splits[eval_split]]
        except FileNotFoundError:
            with open("cvpr/data/lyft/val.txt") as f:
                splits = dict()
                splits[eval_split] = f.readlines()
                splits[eval_split] = [line.rstrip('\n') for line in splits[eval_split]]
    else: splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        if version == "v1.01-train": pass
        else:
            assert version.endswith('trainval'), \
                'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)

        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox:
                box_type_3d, box_mode_3d = get_box_type("LiDAR")
                # Get label name in detection task and filter unused labels.
    
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                from .nuscenes_dataset import map_vehicle_to_car, map_vehicle_to_car_shift
                if not domain_shift:
                    detection_name = map_vehicle_to_car[detection_name]
                else:
                    detection_name = map_vehicle_to_car_shift[detection_name]
                    if detection_name == "ignore": continue

                # Get attribute_name.
                '''
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')
                '''

                ''' original version '''

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=''#attribute_name
                    )
                )
                '''

                data_bbox = []
                data_bbox.extend(sample_annotation['translation'])
                data_bbox.extend(sample_annotation['size'])
                data_bbox.append(Quaternion(sample_annotation['rotation']).yaw_pitch_roll[0])
                data_bbox.extend(nusc.box_velocity(sample_annotation['token'])[:2])
                data_bbox = np.expand_dims(np.array(data_bbox), axis=0)

                after_data_bbox = LiDARInstance3DBoxes(
                    data_bbox,
                    box_dim=data_bbox.shape[-1],
                    origin=(0.5,0.5,0.5)
                ).convert_to(box_mode_3d)
                
                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=after_data_bbox.tensor[0,:3].tolist(),
                        size=after_data_bbox.tensor[0,3:6].tolist(),
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=''#attribute_name
                    )
                )
                '''
            elif box_cls == TrackingBox:
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors when motmetrics package is not installed.
                from nuscenes.eval.tracking.utils import category_to_tracking_name
                tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                if tracking_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    )
                )
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations


def load_lyft_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False, domain_shift: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    if nusc.version == "v1.01-train":
        try:
            with open("data/lyft/val.txt") as f:
                splits = dict()
                splits[eval_split] = f.readlines()
                splits[eval_split] = [line.rstrip('\n') for line in splits[eval_split]]
        except FileNotFoundError:
            with open("cvpr/data/lyft/val.txt") as f:
                splits = dict()
                splits[eval_split] = f.readlines()
                splits[eval_split] = [line.rstrip('\n') for line in splits[eval_split]]
    else: splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        if version == "v1.01-train": pass
        else:
            assert version.endswith('trainval'), \
                'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)

        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox:
                box_type_3d, box_mode_3d = get_box_type("LiDAR")
                # Get label name in detection task and filter unused labels.
    
                detection_name = sample_annotation['category_name']
                if detection_name is None:
                    continue

                from .lyft_dataset import map_vehicle_to_car, map_vehicle_to_car_shift
                if not domain_shift:
                    detection_name = map_vehicle_to_car[detection_name]
                else:
                    detection_name = map_vehicle_to_car_shift[detection_name]
                    if detection_name == "ignore": continue
                
                # Get attribute_name.
                '''
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')
                '''

                ''' original version '''

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=''#attribute_name
                    )
                )
                '''

                data_bbox = []
                data_bbox.extend(sample_annotation['translation'])
                data_bbox.extend(sample_annotation['size'])
                data_bbox.append(Quaternion(sample_annotation['rotation']).yaw_pitch_roll[0])
                data_bbox.extend(nusc.box_velocity(sample_annotation['token'])[:2])
                data_bbox = np.expand_dims(np.array(data_bbox), axis=0)

                after_data_bbox = LiDARInstance3DBoxes(
                    data_bbox,
                    box_dim=data_bbox.shape[-1],
                    origin=(0.5,0.5,0.5)
                ).convert_to(box_mode_3d)
                
                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=after_data_bbox.tensor[0,:3].tolist(),
                        size=after_data_bbox.tensor[0,3:6].tolist(),
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=''#attribute_name
                    )
                )
                '''
            elif box_cls == TrackingBox:
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors when motmetrics package is not installed.
                from nuscenes.eval.tracking.utils import category_to_tracking_name
                tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                if tracking_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    )
                )
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations

def waymo_add_center_dist(eval_boxes: EvalBoxes, pose_dict=None):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0] - pose_dict[sample_token][0,-1],
                               box.translation[1] - pose_dict[sample_token][1,-1],
                               box.translation[2] - pose_dict[sample_token][2,-1])
            if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
                box.ego_translation = ego_translation
            else:
                raise NotImplementedError

    return eval_boxes

def add_center_dist(nusc: NuScenes,
                    eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0] - pose_record['translation'][0],
                               box.translation[1] - pose_record['translation'][1],
                               box.translation[2] - pose_record['translation'][2])
            if isinstance(box, DetectionBox) or isinstance(box, TrackingBox):
                box.ego_translation = ego_translation
            else:
                raise NotImplementedError

    return eval_boxes

def height_diff(nusc: NuScenes, eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    heights = list()
    for sample_token in eval_boxes.sample_tokens:
        sample_rec = nusc.get('sample', sample_token)
        sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        
        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            heights.append(box.translation[2])

    return eval_boxes

def waymo_filter_eval_boxes(
                      eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)
    
    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):
        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < 50]#max_dist[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)
    
    return eval_boxes

def filter_eval_boxes(nusc: NuScenes,
                      eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      visual=False,
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)

    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    if visual: result_boxes = []
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):
        # Filter on distance first.
        total += len(eval_boxes[sample_token])
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          box.ego_dist < max_dist[box.__getattribute__(class_field)]]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        #if visual: result_boxes.extend(eval_boxes[sample_token])
        point_filter += len(eval_boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR and RADAR points based filtering: %d" % point_filter)

    if visual: return eval_boxes, result_boxes
    return eval_boxes


def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox):
        class_field = 'detection_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field
