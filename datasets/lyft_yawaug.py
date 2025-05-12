# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import numpy as np
import mmcv
from mmdet.datasets import DATASETS

from .lyft_dataset import *
import random 
import math
import copy
from .utils import gt_vis, get_matrix

@DATASETS.register_module()
class LyftYawAug(LyftDataset):    
    def __init__(self, naug=0, shuffle=False, target_ann_file=None, aug_ratio=1.0, aug_inplace=False, **kwargs):
        self.naug = naug
        self.aug_ratio = aug_ratio
        self.aug_inplace = aug_inplace
        super().__init__(**kwargs)        

        if not self.test_mode :
            if shuffle : 
                random.shuffle(self.data_infos)
            if target_ann_file is not None : 
                self.data_infos = self.target_aug(self.data_infos, target_ann_file)
            gt_vis(self.data_infos, 1)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']                
        if self.naug > 0 and not self.test_mode :          
            aug_data_infos = self.yaw_aug(data_infos)
            if self.aug_inplace : 
                data_infos = aug_data_infos
            else : 
                data_infos += aug_data_infos
        return data_infos

    def yaw_aug(self, data_infos):                
        idx = [i for i in range(len(data_infos))]        
        random.shuffle(idx)
        idx = idx[:int(len(data_infos)*self.aug_ratio)]        
        aug_infos = []                  
        for index in idx:  
            info = data_infos[index]                 
            l2e = np.eye(4)
            l2e[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            l2e[:3, 3] = info['lidar2ego_translation']                                    
            if self.aug_inplace :                  
                yaw_augs = random.sample([-90, 0, 90, 180], self.naug)
            else : 
                yaw_augs = random.sample([-90, 90, 180], self.naug)            
            for yaw in yaw_augs :
                aug_info = copy.deepcopy(info) 
                if yaw == 0 : 
                    aug_infos.append(aug_info)
                    continue
                
                n2l = get_matrix([0,0,0,yaw,0,0])
                l2n = np.linalg.inv(n2l)       
                n2e = np.dot(l2e, n2l)
                
                aug_info['lidar2ego_rotation'] = Quaternion(matrix=n2e[:3, :3]).elements
                # aug_info['lidar2ego_translation'] = n2e[:3, 3] # same
                for gt in aug_info['gt_boxes']:  
                    loc = np.ones(4, )
                    loc[:3] = gt[:3]
                    nloc = np.dot(l2n, loc)
                    gt[:3] = nloc[:3]
                    gt[-1] += math.radians(yaw)                
                for cam_info in aug_info['cams'].values(): 
                    s2l = np.eye(4)
                    s2l[:3, :3] = cam_info['sensor2lidar_rotation']
                    # syaw = Quaternion(matrix=s2l[:3, :3]).yaw_pitch_roll[0]
                    # import pdb; pdb.set_trace()
                    s2l[:3, 3] = cam_info['sensor2lidar_translation']
                    s2n = np.dot(l2n, s2l)             
                    cam_info['sensor2lidar_rotation'] = s2n[:3, :3]
                    cam_info['sensor2lidar_translation'] = s2n[:3, 3]
                aug_info['lidar2new']=l2n
                aug_infos.append(aug_info)        
        return aug_infos

    def target_aug(self, data_infos, target_ann_file):
        target_data = mmcv.load(target_ann_file)
        target_data_infos = list(sorted(target_data['infos'], key=lambda e: e['timestamp']))

        cams = list(target_data_infos[0]['cams'].keys())
        results = defaultdict(list)
        global_keys = ['lidar2ego_rotation', 'lidar2ego_translation']
        for tinfo in target_data_infos:
            for gkey in global_keys :
                results[gkey].append(tinfo[gkey])
        results_mean = dict()
        for key in results.keys() :
            results_mean[key] = np.percentile(results[key], 50, axis=0)
        
        new2e = np.eye(4)
        new2e[:3, :3] = Quaternion(results_mean['lidar2ego_rotation']).rotation_matrix        
        new2e[:3, 3] = results_mean['lidar2ego_translation']      
        for info in data_infos[:int(len(data_infos)*self.aug_ratio)]:
            l2e = np.eye(4) 
            l2e[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix    
            l2e[:3, 3] = info['lidar2ego_translation']      
            l2n = np.dot(np.linalg.inv(new2e), l2e)
            yaw = Quaternion(matrix=np.linalg.inv(l2n)).yaw_pitch_roll[0]            
            for gt in info['gt_boxes']:  
                loc = np.ones(4, )
                loc[:3] = gt[:3]
                nloc = np.dot(l2n, loc)
                gt[:3] = nloc[:3]
                gt[-1] += yaw
            for cam in cams :   
                s2l = np.eye(4)
                s2l[:3, :3] = info['cams'][cam]['sensor2lidar_rotation']
                s2l[:3, 3] = info['cams'][cam]['sensor2lidar_translation']
                s2n = np.dot(l2n, s2l)
                info['cams'][cam]['sensor2lidar_rotation'] = s2n[:3, :3]
                info['cams'][cam]['sensor2lidar_translation'] = s2n[:3, 3]                                                               
            info['lidar2ego_rotation'] = results_mean['lidar2ego_rotation']
            info['lidar2ego_translation'] = results_mean['lidar2ego_translation']  
            info['lidar2new'] = l2n
        
        return data_infos
