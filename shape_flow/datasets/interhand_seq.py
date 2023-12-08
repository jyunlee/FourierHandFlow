import os
import sys
import json
import yaml
import torch
import random
import pickle
import os.path as osp
import cv2 as cv
import numpy as np
import open3d as o3d

from tqdm import tqdm
from glob import glob
from pathlib import Path
from trimesh import Trimesh
from trimesh.remesh import subdivide
#from trimesh.repair import fix_normals
#from trimesh.geometry import mean_vertex_normals

from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as transforms

from lib.manolayer import ManoLayer, rodrigues_batch
from lib.libmesh import check_mesh_contains 

from lib.halo_adapter.converter import PoseConverter, transform_to_canonical
from lib.halo_adapter.interface import (convert_joints, change_axes, scale_halo_trans_mat)
from lib.halo_adapter.projection import get_projection_layer
from lib.halo_adapter.transform_utils import xyz_to_xyz1


def export_point_cloud(pts, f_name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts) 
    o3d.io.write_point_cloud(f"debug/{f_name}.ply", pcd) 


class TrainSampler():
    """
    Arguments:
    data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, seq_len=17, seq_num=999999):
        self.data_source = data_source
        self.seq_len = seq_len
        self.seq_num = seq_num

        self.n_samples = self.seq_len * self.seq_num

    @property
    def num_samples(self):
        return len(self.data_source)

    def __iter__(self):

        n = len(self.data_source)
        seed_n_list = torch.randperm(n).tolist()

        idx_list = []
        for seed_n in tqdm(seed_n_list):

            start_cam, start_frame, start_joint_check = self.data_source.return_cam_frame(seed_n) 
            end_cam, end_frame, end_joint_check = self.data_source.return_cam_frame(seed_n + self.seq_len-1) 

            if start_joint_check == False or end_joint_check == False:
                continue

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


class ValSampler():
    """
    Arguments:
    data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, seq_len=17, seq_num=2000):
        self.data_source = data_source
        self.seq_len = seq_len
        self.seq_num = seq_num

        self.n_samples = self.seq_len * self.seq_num

    @property
    def num_samples(self):
        return self.n_samples

    def __iter__(self):

        torch.manual_seed(0)

        n = len(self.data_source)
        seed_n_list = torch.randperm(n).tolist()

        idx_list = []
        for seed_n in tqdm(seed_n_list):

            start_cam, start_frame, start_joint_check = self.data_source.return_cam_frame(seed_n)      
            end_cam, end_frame, end_joint_check = self.data_source.return_cam_frame(seed_n + (self.seq_len-1)*3)

            if start_joint_check == False or end_joint_check == False:
                continue

            if start_cam != end_cam:
                continue

            if int(start_frame) + (self.seq_len-1)*3 != int(end_frame):        
                continue
            
            for i in range(self.seq_len):
                idx_list.append(seed_n + i*3) # since the sampling rates of the original train and val set is different

            if len(idx_list) > self.n_samples: 
                break

        return iter(idx_list)

    def __len__(self):
        return self.n_samples


class TestSampler():
    """
    Arguments:
    data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, seq_len=17, seq_num=2000):
        self.data_source = data_source
        self.seq_len = seq_len
        self.seq_num = seq_num

        self.n_samples = self.seq_len * self.seq_num

    @property
    def num_samples(self):
        return self.n_samples

    def __iter__(self):

        torch.manual_seed(0)

        n = len(self.data_source)
        seed_n_list = torch.randperm(n).tolist()

        idx_list = []
        for seed_n in tqdm(seed_n_list):

            start_cam, start_frame, start_joint_check = self.data_source.return_cam_frame(seed_n)      
            end_cam, end_frame, end_joint_check = self.data_source.return_cam_frame(seed_n + (self.seq_len)*3)


            if start_joint_check == False or end_joint_check == False:
                continue

            if start_cam != end_cam:
                continue

            if int(start_frame) + (self.seq_len)*3 != int(end_frame):        
                continue
            
            for i in range(self.seq_len):
                idx_list.append(seed_n + i)

            if len(idx_list) > self.n_samples: 
                break

        return iter(idx_list)

    def __len__(self):
        return self.n_samples


class Jr():
    def __init__(self, J_regressor,
                 device='cpu'):
        self.device = device
        self.process_J_regressor(J_regressor)

    def process_J_regressor(self, J_regressor):
        J_regressor = J_regressor.clone().detach()
        tip_regressor = torch.zeros_like(J_regressor[:5])
        tip_regressor[0, 745] = 1.0
        tip_regressor[1, 317] = 1.0
        tip_regressor[2, 444] = 1.0
        tip_regressor[3, 556] = 1.0
        tip_regressor[4, 673] = 1.0
        J_regressor = torch.cat([J_regressor, tip_regressor], dim=0)
        new_order = [0, 13, 14, 15, 16,
                     1, 2, 3, 17,
                     4, 5, 6, 18,
                     10, 11, 12, 19,
                     7, 8, 9, 20]
        self.J_regressor = J_regressor[new_order].contiguous().to(self.device)

    def __call__(self, v):
        return torch.matmul(self.J_regressor, v)


def can2posed_points(points, point_weights, fwd_transformation):
    B, T, K = point_weights.shape
    point_weights = point_weights.view(B * T, 1, K)  

    fwd_transformation = fwd_transformation.unsqueeze(1).repeat(1, T, 1, 1, 1)  
    fwd_transformation = fwd_transformation.view(B * T, K, -1)  
    trans = torch.bmm(point_weights, fwd_transformation).view(B * T, 4, 4)

    points = torch.cat([points, torch.ones(B, T, 1, device=points.device)], dim=-1).view(B * T, 4, 1)
    posed_points = torch.bmm(trans, points)[:, :3, 0].view(B, T, 3)

    return posed_points


def batch_rigid_transform(rot_mats, joints):
    """ Rigid transformations over joints

    Args:
        rot_mats (torch.tensor): Rotation matrices (BxNx3x3).
        joints (torch.tensor): Joint locations (BxNx3).

    Returns:
        posed_joints (torch.tensor): The locations of the joints after applying transformations (BxNx3).
        rel_transforms (torch.tensor): Relative wrt root joint rigid transformations (BxNx4x4).
    """
    rot_mats = rot_mats[:, :, :3, :3]
    kintree_table = torch.Tensor([-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  0, 10, 11,  0, 13, 14])
    joints = torch.Tensor(joints).unsqueeze(0)
    B, K = rot_mats.shape[0], joints.shape[1]

    parents = kintree_table.long()

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()

    rel_joints[:, 1:] -= joints[:, parents[1:]]


    transforms_mat = torch.cat([
        F.pad(rot_mats.reshape(-1, 3, 3), [0, 0, 0, 1]),
        F.pad(rel_joints.reshape(-1, 3, 1), [0, 0, 0, 1], value=1)
    ], dim=2).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)
    joints_hom = torch.cat([
        joints,
        torch.zeros([B, K, 1, 1])
    ], dim=2)
    init_bone = F.pad(torch.matmul(transforms, joints_hom), [3, 0, 0, 0, 0, 0, 0, 0])
    rel_transforms = transforms - init_bone

    return rel_transforms, rel_joints



def normalize(bv, eps=1e-8):
    """
    Normalizes the last dimension of bv such that it has unit length in
    euclidean sense
    """
    eps_mat = torch.tensor(eps, device=bv.device)
    norm = torch.max(torch.norm(bv, dim=-1, keepdim=True), eps_mat)
    bv_n = bv / norm
    return bv_n


def get_mano_path():
    mano_path = {'right': '/workspace/AFOF/leap/body_models/mano/models/MANO_RIGHT.pkl', 
                 'left': '/workspace/AFOF/leap/body_models/mano/models/MANO_LEFT.pkl'}
    return mano_path


def load_config(path):
    """ Loads config file.

    Args:
        path (str): path to config file
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def fix_shape(mano_layer): 
     if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:   
         print('Fix shapedirs bug of MANO')     
         mano_layer['left'].shapedirs[:, 0, :] *= -1   


def seal(mesh_to_seal):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    mesh_to_seal.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = mesh_to_seal.vertices.shape[0] - 1

    # pylint: disable=unsubscriptable-object # pylint/issues/3139
    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, new_faces])

    return mesh_to_seal


class InterHandSeqDataset(data.Dataset):
    def __init__(self, cfg, split):
        assert split in ['train', 'test', 'val']
        self.split = split
        mano_path = get_mano_path()
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)
        self.mano_faces = {'left': self.mano_layer['left'].get_faces(),
                           'right': self.mano_layer['right'].get_faces()}
        self.J_regressor = {'left': Jr(self.mano_layer['left'].J_regressor),
                            'right': Jr(self.mano_layer['right'].J_regressor)}

        self.data_path = cfg.get('dataset_folder', '')
        self.leap_data_path = cfg.get('leap_dataset_folder', '')
        self.pred_joints_path = cfg.get('joints_folder', '')
        self.anno_dir = osp.join(self.data_path, split, 'anno')
        self.use_gt_joints = cfg.get('use_gt_joints', False)

        valid_seqs = open(osp.join(self.data_path, f'{split}_seqs.txt'), 'r')
        seqs = valid_seqs.readlines()
        seqs = [seq.replace('\n', '') for seq in seqs]

        self.data_list = []

        if split == 'test':
            f = open('test_file.txt', 'r')
            for line in f.readlines():
                self.data_list.append(line[:-1] + '.pkl')

            #self.data_list = self.data_list[:17*101]

        else:
            if split == 'train':
                for seq in seqs:
                    self.data_list += list(Path(osp.join(self.anno_dir, seq)).rglob('*.pkl')) 
            else:
                self.data_list = list(Path(self.anno_dir).rglob('*.pkl')) 

            self.data_list.sort()

        self.seq_len = cfg.get('seq_len', 17)

        self.size = len(self.data_list)

        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        # Sampling config
        sampling_config = cfg.get('sampling_config', {})
        self.points_uniform_ratio = sampling_config.get('points_uniform_ratio', 0.5)
        self.bbox_padding = sampling_config.get('bbox_padding', 0)
        self.points_padding = sampling_config.get('points_padding', 0.1)
        self.points_sigma = sampling_config.get('points_sigma', 0.01)
        self.n_points_posed = sampling_config.get('n_points_posed', 2048)

        self.can_vertices = {'right': np.load(osp.join(self.data_path, 'canonical_right.npz'))['can_vertices'].astype(np.float32),
                             'left': np.load(osp.join(self.data_path, 'canonical_right.npz'))['can_vertices'].astype(np.float32)}
                             #'left': np.load(osp.join(self.data_path, 'canonical_left.npz'))['can_vertices'].astype(np.float32)* np.array([-1., 1., 1.])}

        self.can_rel_joints = {'right': np.load(osp.join(self.data_path, 'canonical_right.npz'))['rel_joints'].astype(np.float32),
                               'left': np.load(osp.join(self.data_path, 'canonical_left.npz'))['rel_joints'].astype(np.float32)* np.array([-1., 1., 1.])}

        self.can_pose = {'right': np.load(osp.join(self.data_path, 'canonical_right.npz'))['pose_mat'].astype(np.float32),
                         'left': np.load(osp.join(self.data_path, 'canonical_left.npz'))['pose_mat'].astype(np.float32)* np.array([-1., 1., 1.])}

        self.can_joints = {'right': self.J_regressor['right'](torch.Tensor(self.can_vertices['right'])),  
                           'left': self.J_regressor['left'](torch.Tensor(self.can_vertices['left'])) * torch.Tensor([-1., 1., 1.])}

        self.can_org_joints = {'right': np.load(osp.join(self.data_path, 'canonical_right.npz'))['can_joints'].astype(np.float32),
                             'left': np.load(osp.join(self.data_path, 'canonical_left.npz'))['can_joints'].astype(np.float32)* np.array([-1., 1., 1.])}

        self.annots_path = os.path.join('/data/hand_data/AFOF/InterHand_HFPS/InterHand2.6M_30fps_batch1/annotations', self.split, f'InterHand2.6M_{self.split}_MANO_NeuralAnnot.json')


        with open(self.annots_path, "r") as f:
            self.annots = json.load(f)

    def __len__(self):
        return self.size

    def sample_points(self, mesh, n_points, prefix='', compute_occupancy=False, frame=None):
        # Get extents of model.
        bb_min = np.min(mesh.vertices, axis=0)
        bb_max = np.max(mesh.vertices, axis=0)
        total_size = (bb_max - bb_min).max()

        # Scales all dimensions equally.
        scale = total_size / (1 - self.bbox_padding)
        loc = np.array([(bb_min[0] + bb_max[0]) / 2.,
                        (bb_min[1] + bb_max[1]) / 2.,
                        (bb_min[2] + bb_max[2]) / 2.], dtype=np.float32)

        n_points_uniform = int(n_points * self.points_uniform_ratio)
        n_points_surface = n_points - n_points_uniform

        box_size = 1 + self.points_padding
        points_uniform = np.random.rand(n_points_uniform, 3)
        points_uniform = box_size * (points_uniform - 0.5)
        # Scale points in (padded) unit box back to the original space
        points_uniform *= scale
        points_uniform += loc
        # Sample points around posed-mesh surface
        n_points_surface_cloth = n_points_surface
        points_surface = mesh.sample(n_points_surface_cloth)

        points_surface = points_surface[:n_points_surface_cloth]
        points_surface += np.random.normal(scale=self.points_sigma, size=points_surface.shape)

        # Check occupancy values for sampled points
        query_points = np.vstack([points_uniform, points_surface]).astype(np.float32)

        to_ret = {
            f'{prefix}points': query_points,
            f'{prefix}loc': loc,
            f'{prefix}scale': np.asarray(scale),
        }
        if compute_occupancy:
            to_ret[f'{prefix}occ'] = check_mesh_contains(mesh, query_points).astype(np.float32)

        return to_ret

    def return_cam_frame(self, idx, check_joints=True):
        try:
            data_path = str(self.data_list[idx])
            cam_name = data_path.split('/')[-2]
            frame_name = data_path.split('/')[-1][:-4]       

            joints_path = os.path.join(self.split, data_path.split('/')[-4], data_path.split('/')[-3], data_path.split('/')[-2], data_path.split('/')[-1][:-4]) 

            left_joint_check = os.path.exists(os.path.join(self.pred_joints_path, joints_path + '_left.ply')) 
            right_joint_check = os.path.exists(os.path.join(self.pred_joints_path, joints_path + '_left.ply')) 
            joint_check = left_joint_check and right_joint_check

        except:
            cam_name = '0'
            frame_name = '0'
            joint_check = False
        
        return cam_name, frame_name, joint_check

    def __getitem__(self, idx):

        hand_dict = {}

        data_path = str(self.data_list[idx])
        capture_name = data_path.split('/')[-4] 
        seq_name = data_path.split('/')[-3] 
        cam_name = data_path.split('/')[-2] 
        frame_name = data_path.split('/')[-1][:-4] 

        hand_dict = {'left': {}, 'right': {}}

        subdir = osp.join(capture_name, seq_name, cam_name)

        img_path = osp.join(self.data_path, self.split, 'img', subdir, '{}.jpg'.format(frame_name))
        img = cv.imread(osp.join(self.data_path, self.split, 'img', subdir, '{}.jpg'.format(frame_name)))

        try:
            imgTensor = torch.tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        except:
            print(img_path)
            exit()

        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)

        with open(os.path.join(self.data_path, self.split, 'anno', subdir, '{}.pkl'.format(frame_name)), 'rb') as file:
            data = pickle.load(file)

        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']

        for hand_type in ['left', 'right']:

            params = data['mano_params'][hand_type]

            handV, handJ = self.mano_layer[hand_type](torch.from_numpy(params['R']).float(),
                                                      torch.from_numpy(params['pose']).float(),
                                                      torch.from_numpy(params['shape']).float(),
                                                      trans=torch.from_numpy(params['trans']).float())
            
            if hand_type == 'left':
                handV *= torch.Tensor([-1., 1., 1.])
                handJ *= torch.Tensor([-1., 1., 1.])

            handV = handV[0].numpy()
            handJ = handJ_org = handJ[0].numpy()
            hand_root = handJ[0]

            handV = handV @ R.T + T
            handJ = handJ @ R.T + T

            handV2d = handV @ camera.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            handJ2d = handJ @ camera.T
            handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]

            posed_mesh = seal(Trimesh(handV, self.mano_faces['right'], process=False))
            hand_dict[hand_type].update(self.sample_points(posed_mesh, self.n_points_posed, compute_occupancy=True))
            can_mesh = seal(Trimesh(self.can_vertices['right'], self.mano_faces['right'], process=False))
            hand_dict[hand_type].update(self.sample_points(can_mesh, self.n_points_posed, prefix='can_', compute_occupancy=True))

            if self.use_gt_joints and self.split=='train':
                joints = handJ_org

                if hand_type == 'left':
                    joints *= np.array([-1., 1., 1.])

            else:
                joints_path = osp.join(self.pred_joints_path, self.split, subdir, f'{frame_name}_{hand_type}.ply') 
                joints = np.asarray(o3d.io.read_point_cloud(joints_path).points)

            if hand_type == 'left':
                joints *= np.array([-1., 1., 1.])

            permute_mat = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
            joints_root = joints[0]
            joints = joints - joints[0]
            joints = joints[permute_mat] 

            # HALO code segments - start
            joints = torch.Tensor(joints).unsqueeze(0)
            halo_order_joints = joints
            joints = convert_joints(joints, source='halo', target='biomech')
            is_right_vec = torch.ones(joints.shape[0])

            palm_align_kps_local_cs, glo_rot_right = transform_to_canonical(joints, is_right=is_right_vec)
            palm_align_kps_local_cs_nasa_axes, swap_axes_mat = change_axes(palm_align_kps_local_cs)

            swap_axes_mat = swap_axes_mat.unsqueeze(0)
            rot_then_swap_mat = torch.matmul(swap_axes_mat, glo_rot_right).unsqueeze(0)

            can_verts = self.can_vertices['right']
            can_joints = self.can_joints['right']
            can_permute_mat = [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
            can_joints_root = can_joints[0]
            can_joints = can_joints - can_joints_root
            can_joints = can_joints[permute_mat] 

            can_joints = torch.Tensor(can_joints).unsqueeze(0)
            
            can_joints = convert_joints(can_joints, source='halo', target='biomech')
            can_is_right_vec = torch.ones(can_joints.shape[0])

            can_palm_align_kps_local_cs, can_glo_rot_right = transform_to_canonical(can_joints, is_right=can_is_right_vec)
            can_palm_align_kps_local_cs_nasa_axes, can_swap_axes_mat = change_axes(can_palm_align_kps_local_cs)

            can_swap_axes_mat = can_swap_axes_mat.unsqueeze(0)
            can_rot_then_swap_mat = torch.matmul(can_swap_axes_mat, can_glo_rot_right).unsqueeze(0)

            can_verts = torch.Tensor(can_verts) - can_joints_root
            can_verts = torch.matmul(can_rot_then_swap_mat.squeeze(), xyz_to_xyz1(can_verts).unsqueeze(-1))[:, :3, 0]
            can_verts = can_verts + can_joints_root

            trans_can_joints = convert_joints(can_palm_align_kps_local_cs_nasa_axes, source='biomech', target='halo').squeeze().numpy()

            trans_org_joints = convert_joints(palm_align_kps_local_cs_nasa_axes, source='biomech', target='halo').squeeze().numpy()

            pose_converter = PoseConverter()

            trans_mat_pc = pose_converter(palm_align_kps_local_cs_nasa_axes, is_right_vec)
            trans_mat_pc = trans_mat_pc[0]

            trans_mat_pc = convert_joints(trans_mat_pc, source='biomech', target='halo')

            joints_for_nasa_input = torch.tensor([0, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16])

            trans_mat_pc = trans_mat_pc[:, joints_for_nasa_input]

            can_trans_mat_pc = pose_converter(can_palm_align_kps_local_cs_nasa_axes, is_right_vec)
            can_trans_mat_pc = can_trans_mat_pc[0] 

            can_trans_mat_pc = convert_joints(can_trans_mat_pc, source='biomech', target='halo')

            joints_for_nasa_input = torch.tensor([0, 2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16])

            can_trans_mat_pc = can_trans_mat_pc[:, joints_for_nasa_input]

            handV -= handJ[0]

            # subdivide meshes to extract dense points w/ correspondence
            if self.split == 'train' or self.split == 'val': # WARNING!        
                can_mesh_v, can_mesh_f = subdivide(can_verts, self.mano_faces['right'].astype(int))
                can_mesh_v, can_mesh_f = subdivide(can_mesh_v, can_mesh_f)
                can_mesh_v, can_mesh_f = subdivide(can_mesh_v, can_mesh_f)
                posed_mesh_v, posed_mesh_f = subdivide(torch.Tensor(handV), self.mano_faces['right'].astype(int))
                posed_mesh_v, posed_mesh_f = subdivide(posed_mesh_v, posed_mesh_f)
                posed_mesh_v, posed_mesh_f = subdivide(posed_mesh_v, posed_mesh_f)

            else:
                can_mesh_v, can_mesh_f = can_verts, self.mano_faces['right'].astype(int)
                posed_mesh_v, posed_mesh_f = torch.Tensor(handV), self.mano_faces['right'].astype(int)  

            can_mesh = Trimesh(can_mesh_v, can_mesh_f, process=False) 
            posed_mesh = Trimesh(posed_mesh_v, posed_mesh_f, process=False) 

            if self.split == 'train' or self.split == 'val': # WARNING!
                subset_idx = torch.randperm(can_mesh_v.shape[0])[:self.n_points_posed]
            else:
                subset_idx = torch.arange(can_mesh_v.shape[0])

            hand_dict[hand_type]['out_dir'] = subdir
            hand_dict[hand_type]['out_file'] = frame_name
            hand_dict[hand_type]['inv_transformation'] = torch.Tensor(trans_mat_pc)
            hand_dict[hand_type]['fwd_transformation'] = torch.inverse(torch.Tensor((trans_mat_pc)))
            hand_dict[hand_type]['can_fwd_transformation'] = torch.inverse(torch.Tensor(can_trans_mat_pc))
            hand_dict[hand_type]['can_inv_transformation'] = torch.Tensor(can_trans_mat_pc)

            hand_dict[hand_type]['can_vertices'] = can_mesh_v[subset_idx] # WARNING!
            hand_dict[hand_type]['org_can_vertices'] = can_verts #self.can_vertices[hand_type]
            hand_dict[hand_type]['can_faces'] = self.mano_faces['right'].astype(int) #)self.can_vertices[hand_type]

            hand_dict[hand_type]['can_joints'] = self.can_joints['right']
            hand_dict[hand_type]['can_joints_root'] = can_joints_root
            hand_dict[hand_type]['can_rel_joints'] = self.can_rel_joints['right'] # 16 x 3 x 1
            hand_dict[hand_type]['can_pose'] = self.can_pose['right'] # 16 x 3 x 3
            hand_dict[hand_type]['root_rot_mat'] = rot_then_swap_mat.squeeze() 
            hand_dict[hand_type]['joints'] = joints
            hand_dict[hand_type]['joints_root'] = joints_root

            hand_dict[hand_type]['posed_verts'] = handV 
            hand_dict[hand_type]['posed_faces'] = self.mano_faces['right'].astype(int)

            hand_dict[hand_type]['gt_mano_verts'] = posed_mesh_v[subset_idx] # here
            hand_dict[hand_type]['gt_mano_faces'] = posed_mesh_f #self.mano_faces[hand_type].astype(int)
            hand_dict[hand_type]['gt_mano_joints'] = handJ

            hand_dict[hand_type]['trans_org_joints'] = trans_org_joints
            hand_dict[hand_type]['trans_can_joints'] = trans_can_joints

            hand_dict[hand_type]['camera_params'] = {}
            hand_dict[hand_type]['camera_params']['R'] = R
            hand_dict[hand_type]['camera_params']['T'] = T
            hand_dict[hand_type]['camera_params']['camera'] = camera
            hand_dict[hand_type]['camera_params']['root_xyz'] = torch.Tensor(hand_root).double() #joints_root
            hand_dict[hand_type]['camera_params']['img_path'] = img_path

        return imgTensor, hand_dict


if __name__ == '__main__':

    dataset = InterHand_Seq_4D('val', '/workspace/AFOF/leap/configurations/4d_baseline.yaml')

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False)

    for img, hand_dict in loader:
        print(hand_dict['left']['loc'], hand_dict['left']['scale'], hand_dict['left']['occ'].shape, hand_dict['left']['occ'].sum())
