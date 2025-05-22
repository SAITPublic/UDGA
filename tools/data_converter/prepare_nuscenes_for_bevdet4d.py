# Copyright (c) Phigent Robotics. All rights reserved.
import os
import pickle
import json
from nuscenes import NuScenes
import numpy as np
from pyquaternion import Quaternion


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep

def add_adj_info():
    interval = 3
    max_adj = 60
    for set in ['train', 'val']:
        nuscenes_version = 'v1.0-trainval'
        dataroot = './data/nuscenes/'
        # nuscenes_version = 'v1.0-mini'
        # dataroot = './data/nuscenes_mini/'
        dataset = pickle.load(open(f'{dataroot}nuscenes_infos_%s.pkl' % set, 'rb'))
        nuscenes = NuScenes(nuscenes_version, dataroot)
        map_token_to_id = dict()
        for id in range(len(dataset['infos'])):
            map_token_to_id[dataset['infos'][id]['token']] = id
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            sample = nuscenes.get('sample', info['token'])
            for adj in ['next', 'prev']:
                sweeps = []
                adj_list = dict()
                for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                    adj_list[cam] = []

                    sample_data = nuscenes.get('sample_data', sample['data'][cam])
                    adj_list[cam] = []
                    count = 0
                    while count < max_adj:
                        if sample_data[adj] == '':
                            break
                        sd_adj = nuscenes.get('sample_data', sample_data[adj])
                        sample_data = sd_adj
                        adj_list[cam].append(dict(data_path=dataroot + sd_adj['filename'],
                                                  timestamp=sd_adj['timestamp'],
                                                  ego_pose_token=sd_adj['ego_pose_token']))
                        count += 1
                for count in range(interval - 1, min(max_adj, len(adj_list['CAM_FRONT'])), interval):
                    timestamp_front = adj_list['CAM_FRONT'][count]['timestamp']
                    # get ego pose
                    pose_record = nuscenes.get('ego_pose', adj_list['CAM_FRONT'][count]['ego_pose_token'])

                    pts_info = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
                    pts_dict = pts_info
                    cs_record = nuscenes.get('calibrated_sensor',
                                          pts_info['calibrated_sensor_token'])
                    pose_record = nuscenes.get('ego_pose', pts_info['ego_pose_token'])

                    l2e_r = cs_record['rotation']
                    l2e_t = cs_record['translation']
                    e2g_r = pose_record['rotation']
                    e2g_t = pose_record['translation']
                    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
                    sweeps_pts=[]
                    while len(sweeps_pts) < 10:
                        if not pts_info['prev'] == '':
                            pts_info = nuscenes.get('sample_data', pts_info['prev'])
                            try:
                                sweep = obtain_sensor2top(nuscenes, pts_info['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                            except KeyError:
                                break
                            sweeps_pts.append(sweep)
                        else:
                            break
                    pts_dict['sweeps'] = sweeps_pts
                    # get cam infos
                    cam_infos = dict(CAM_FRONT=dict(data_path=adj_list['CAM_FRONT'][count]['data_path']))
                    for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                                'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
                        timestamp_curr_list = np.array([t['timestamp'] for t in adj_list[cam]], dtype=np.long)
                        diff = np.abs(timestamp_curr_list - timestamp_front)
                        selected_idx = np.argmin(diff)
                        cam_infos[cam] = dict(data_path=adj_list[cam][int(selected_idx)]['data_path'])
                        # print('%02d-%s'%(selected_idx, cam))
                    sweeps.append(dict(timestamp=timestamp_front, lidar=pts_dict, cams=cam_infos,
                                       ego2global_translation=pose_record['translation'],
                                       ego2global_rotation=pose_record['rotation']))
                dataset['infos'][id][adj] = sweeps if len(sweeps) > 0 else None

            # get ego speed and transfrom the targets velocity from global frame into ego-relative mode
            previous_id = id
            if not sample['prev'] == '':
                sample_tmp = nuscenes.get('sample', sample['prev'])
                previous_id = map_token_to_id[sample_tmp['token']]
            next_id = id
            if not sample['next'] == '':
                sample_tmp = nuscenes.get('sample', sample['next'])
                next_id = map_token_to_id[sample_tmp['token']]
            time_pre = 1e-6 * dataset['infos'][previous_id]['timestamp']
            time_next = 1e-6 * dataset['infos'][next_id]['timestamp']
            time_diff = time_next - time_pre
            posi_pre = np.array(dataset['infos'][previous_id]['ego2global_translation'], dtype=np.float32)
            posi_next = np.array(dataset['infos'][next_id]['ego2global_translation'], dtype=np.float32)
            velocity_global = (posi_next - posi_pre) / time_diff

            l2e_r = info['lidar2ego_rotation']
            l2e_t = info['lidar2ego_translation']
            e2g_r = info['ego2global_rotation']
            e2g_t = info['ego2global_translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            velocity_global = np.array([*velocity_global[:2], 0.0])
            velocity_lidar = velocity_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T
            velocity_lidar = velocity_lidar[:2]

            dataset['infos'][id]['velo'] = velocity_lidar
            dataset['infos'][id]['gt_velocity'] = dataset['infos'][id]['gt_velocity'] - velocity_lidar.reshape(1, 2)

        with open(f'{dataroot}nuscenes_infos_%s_4d_interval%d_max%d.pkl' % (set, interval, max_adj), 'wb') as fid:
            pickle.dump(dataset, fid)

if __name__=='__main__':
    add_adj_info()