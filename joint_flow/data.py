import os
import os.path as osp
import sys
import torch
import pickle

import cv2 as cv
import numpy as np
import open3d as o3d

from tqdm import tqdm
from torch.utils import data
from pathlib import Path
from dependencies.manolayer import ManoLayer


# Sample adjacent frames from the dataset (to create train batches)
class TrainSampler():
    """
    Arguments:
    data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, seq_len=17, seq_num=99999999):
        self.data_source = data_source
        self.seq_len = seq_len
        self.seq_num = seq_num

        self.n_samples = self.seq_len * self.seq_num

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):

        print(f'Indexing train dataset...')

        n = len(self.data_source)
        seed_n_list = torch.randperm(n).tolist()

        idx_list = []
        for seed_n in tqdm(seed_n_list):

            start_cam, start_frame = self.data_source.return_cam_frame(seed_n)
            end_cam, end_frame = self.data_source.return_cam_frame(seed_n + self.seq_len - 1)

            if start_cam != end_cam:
                continue

            if int(start_frame) + (self.seq_len-1)*3 != int(end_frame):
                continue

            for i in range(self.seq_len):
                idx_list.append(seed_n + i)

            if len(idx_list) > self.n_samples: 
                break

        return iter(idx_list)

    def __len__(self):
        return self.num_samples


# Sample adjacent frames from the dataset (to create test batches)
class TestSampler():
    """
    Arguments:
    data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, seq_len=17, seq_num=3000):
        self.data_source = data_source
        self.seq_len = seq_len
        self.seq_num = seq_num

        self.n_samples = self.seq_len * self.seq_num

    @property
    def num_samples(self):
        return self.n_samples

    def __iter__(self):

        print(f'Indexing val/test dataset...')
        torch.manual_seed(0)

        n = len(self.data_source)
        seed_n_list = torch.randperm(n).tolist()

        idx_list = []
        for seed_n in tqdm(seed_n_list):

            start_cam, start_frame = self.data_source.return_cam_frame(seed_n)
            end_cam, end_frame = self.data_source.return_cam_frame(seed_n + self.seq_len*3 - 3)

            if start_cam != end_cam:
                continue

            if int(start_frame) + (self.seq_len-1)*3 != int(end_frame):
                continue

            for i in range(self.seq_len):
                idx_list.append(seed_n + i*3)

            if len(idx_list) > self.n_samples: 
                break

        return iter(idx_list)

    def __len__(self):
        return self.n_samples


def get_data(img_dir, anno_dir, joints_dir, mano_layer, hand_mode, cur_filename):

    img_path = osp.join(img_dir, f'{cur_filename}.jpg')
    img = cv.imread(img_path)   
    img = torch.from_numpy(img)

    joints = np.asarray(o3d.io.read_point_cloud(osp.join(joints_dir, f'{cur_filename}_{hand_mode}.ply')).points)
    joints = torch.Tensor(joints).float()

    with open(osp.join(anno_dir, f'{cur_filename}.pkl'), 'rb') as anno_file:
        anno = pickle.load(anno_file)

    camera_params = {}
    for key in anno['camera'].keys():
        camera_params[key] = torch.Tensor(anno['camera'][key]).unsqueeze(0).float()

    mano_params = anno['mano_params'][hand_mode]

    handV, handJ = mano_layer[hand_mode](torch.from_numpy(mano_params['R']).float(),
                                    torch.from_numpy(mano_params['pose']).float(),
                                    torch.from_numpy(mano_params['shape']).float(),
                                    trans=torch.from_numpy(mano_params['trans']).float())
    
    camera_params['root_xyz'] = torch.Tensor(handJ[0, 9]).unsqueeze(0).float()
    camera_params['img_path'] = img_path
    
    gt_joints = handJ

    return img.squeeze(), joints.squeeze(), camera_params, gt_joints.squeeze()


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        mano_layer['left'].shapedirs[:, 0, :] *= -1


class InterHandSeqDataset(data.Dataset):

    def __init__(self, split_mode, hand_mode, data_path, joints_path):

        assert split_mode in ['train', 'val', 'test']
        assert hand_mode in ['left', 'right']

        self.split = split_mode
        self.hand_mode = hand_mode

        self.img_dir = f'{data_path}/{split_mode}/img'
        self.anno_dir = f'{data_path}/{split_mode}/anno'
        self.joints_dir = f'{joints_path}/{split_mode}'

        self.mano_layer = {'right': ManoLayer(osp.join(os.getcwd(), 'dependencies/models/MANO_RIGHT.pkl'), center_idx=None),
                           'left': ManoLayer(osp.join(os.getcwd(), 'dependencies/models/MANO_LEFT.pkl'), center_idx=None)}
        fix_shape(self.mano_layer)

        # load file path list to reduce path traverse time
        with open(f"config/{split_mode}_file_list.pkl", "rb") as fp:
            self.data_list = pickle.load(fp)
       
        if split_mode != 'test':
            self.data_list.sort()

        self.len = len(self.data_list)

    def __len__(self):
        return self.len
    
    def get_first_seed_of_data_list(self):

        first_seed_list = []
        cur_cam_name = ''

        for a_index, data_path in enumerate(self.data_list):
            cam_name = '/'.join(data_path.split('/')[-3:-1])

            if cur_cam_name == cam_name:
                continue

            else:
                first_seed_list.append(a_index)
                cur_cam_name = cam_name
        
        return first_seed_list
    
    def return_cam_frame(self, idx):
        try:  
            data_path = str(self.data_list[idx]) 
            cam_name = data_path.split('/')[-2]
            frame_name = data_path.split('/')[-1]
        except: 
            cam_name = 'n/a'    
            frame_name = 'n/a'

        return cam_name, frame_name

    def __getitem__(self, idx):

        cur_path = self.data_list[idx]

        img, joints, camera_params, gt_joints = get_data(self.img_dir, self.anno_dir, self.joints_dir, self.mano_layer, self.hand_mode, cur_path)
        
        return img, joints, camera_params, gt_joints


def preprocess_batch(batch, test=False):

    imgs, joints, camera_params, gt_joints = batch

    imgs = imgs.to('cuda')
    joints = joints.to('cuda')
    gt_joints = gt_joints.to('cuda')

    for key in camera_params.keys(): 
        if key not in ['img_path']:  
            camera_params[key] = camera_params[key].to(device='cuda').squeeze()   

    # normalize joints on the mid joint at each frame
    gt_midjoints = torch.unsqueeze(torch.clone(gt_joints[:,9,:]), dim=1)
    joints = joints - torch.unsqueeze(joints[:,9,:], dim=1)
    gt_joints = gt_joints - torch.unsqueeze(gt_joints[:,9,:], dim=1)

    if not test:
        return imgs, joints, camera_params, gt_joints
    else:
        return imgs, joints, camera_params, gt_joints, gt_midjoints


# for sanity check
if __name__ == '__main__':
    dataset = InterHandSeqDataset('val')

