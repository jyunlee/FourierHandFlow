from collections import defaultdict

import os
import tqdm
import torch
import trimesh
import numpy as np
import open3d as o3d

import sys

from lib.im2mesh.utils import libmcubes
from lib.im2mesh.utils.libsimplify import simplify_mesh
from lib.im2mesh.utils.libmise import MISE
from lib.im2mesh.common import make_3d_grid
from lib.libmesh import check_mesh_contains
from trimesh.geometry import mean_vertex_normals


def export_point_cloud(pts, f_name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts) 
    o3d.io.write_point_cloud(f"debug/{f_name}.ply", pcd) 


def set_grad_param(dic, require_grad=True):
    for attr in dic.keys():
        if torch.is_tensor(dic[attr]):
            dic[attr].require_grad = require_grad
    
def xyz_to_xyz1(xyz):
    """ Convert xyz vectors from [BS, ..., 3] to [BS, ..., 4] for matrix multiplication
    """
    ones = torch.ones([*xyz.shape[:-1], 1], device=xyz.device)
    return torch.cat([xyz, ones], dim=-1)


def seal(mesh_to_seal):
    '''
    Seal MANO hand wrist to make it wathertight.
    An average of wrist vertices is added along with its faces to other wrist vertices.
    '''
    circle_v_id = np.array([108, 79, 78, 121, 214, 215, 279, 239, 234, 92, 38, 122, 118, 117, 119, 120], dtype = np.int32)
    center = (mesh_to_seal.vertices[circle_v_id, :]).mean(0)

    mesh_to_seal.vertices = np.vstack([mesh_to_seal.vertices, center])
    center_v_id = mesh_to_seal.vertices.shape[0] - 1

    for i in range(circle_v_id.shape[0]):
        new_faces = [circle_v_id[i-1], circle_v_id[i], center_v_id] 
        mesh_to_seal.faces = np.vstack([mesh_to_seal.faces, new_faces])

    return mesh_to_seal


def can2posed_points(points, point_weights, fwd_transformation, can_inv_transformation):
    B, T, K = point_weights.shape
    point_weights = point_weights.view(B * T, 1, K)  # B*T x 1 x K

    fwd_transformation = fwd_transformation.unsqueeze(1).repeat(1, T, 1, 1, 1)  # B X K x 4 x 4 -> B x T x K x 4 x 4
    can_inv_transformation = can_inv_transformation.unsqueeze(1).repeat(1, T, 1, 1, 1)  # B X K x 4 x 4 -> B x T x K x 4 x 4

    fwd_transformation = fwd_transformation.view(B * T, K, -1)  # B*T x K x 16
    can_inv_transformation = can_inv_transformation.view(B * T, K, -1)  # B*T x K x 16

    trans = torch.bmm(point_weights, fwd_transformation).view(B * T, 4, 4)
    can_inv_trans = torch.bmm(point_weights, can_inv_transformation).view(B * T, 4, 4)

    mano_can_points = torch.cat([points, torch.ones(B, T, 1, device=points.device)], dim=-1).view(B * T, 4, 1)
    halo_can_points = torch.bmm(can_inv_trans, mano_can_points)[:, :3, 0].view(B, T, 3)

    halo_can_points = torch.cat([halo_can_points, torch.ones(B, T, 1, device=points.device)], dim=-1).view(B * T, 4, 1)
    posed_points = torch.bmm(trans, halo_can_points)[:, :3, 0].view(B, T, 3)

    return posed_points


def posed2can_points(points, point_weights, inv_transformation, can_fwd_transformation):
    B, T, K = point_weights.shape
    point_weights = point_weights.view(B * T, 1, K)  # B*T x 1 x K

    inv_transformation = inv_transformation.unsqueeze(1).repeat(1, T, 1, 1, 1)  # B X K x 4 x 4 -> B x T x K x 4 x 4
    can_fwd_transformation = can_fwd_transformation.unsqueeze(1).repeat(1, T, 1, 1, 1)  # B X K x 4 x 4 -> B x T x K x 4 x 4

    inv_transformation = inv_transformation.view(B * T, K, -1)  # B*T x K x 16
    can_fwd_transformation = can_fwd_transformation.view(B * T, K, -1)  # B*T x K x 16

    inv_trans = torch.bmm(point_weights, inv_transformation).view(B * T, 4, 4)
    can_trans = torch.bmm(point_weights, can_fwd_transformation).view(B * T, 4, 4)

    mano_can_points = torch.cat([points, torch.ones(B, T, 1, device=points.device)], dim=-1).view(B * T, 4, 1)
    mano_can_points = torch.bmm(inv_trans, mano_can_points)[:, :3, 0].view(B, T, 3)

    halo_can_points = torch.cat([mano_can_points, torch.ones(B, T, 1, device=points.device)], dim=-1).view(B * T, 4, 1)
    halo_can_points = torch.bmm(can_trans, halo_can_points)[:, :3, 0].view(B, T, 3)

    return halo_can_points


class BaseTrainer:
    """ Base trainers class.

    Args:
        model (torch.nn.Module): Occupancy Network model
        optimizer (torch.optim.Optimizer): pytorch optimizer object
        cfg (dict): configuration
    """

    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.device = cfg['device']

    def evaluate(self, val_loader):
        """ Performs an evaluation.
        Args:
            val_loader (torch.DataLoader): pytorch dataloader
        """
        eval_list = defaultdict(list)

        for data in tqdm.tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def _train_mode(self):
        self.model.train()

    def train_step(self, data):
        self._train_mode()
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data)
        loss_dict['total_loss'].backward()
        self.optimizer.step()
        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad() 
    def eval_step(self, data):
        """ Performs an evaluation step.

        Args:
            data (dict): datasets dictionary
        """
        self.model.eval()
        eval_loss_dict = self.compute_eval_loss(data)
        return {k: v.item() for k, v in eval_loss_dict.items()}

    def compute_loss(self, *kwargs):
        """ Computes the training loss.

        Args:
            kwargs (dict): datasets dictionary
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute_eval_loss(self, data):
        """ Computes the validation loss.

        Args:
            data (dict): datasets dictionary
        """
        return self.compute_loss(data)


class ShapeFlowTrainer(BaseTrainer):
    def __init__(self, model, optimizer, cfg):
        super().__init__(model, optimizer, cfg)

        self._eval_lbs_mode()
        self.occ_loss = torch.nn.MSELoss()
        self.corr_loss = torch.nn.MSELoss()

        self.occ = {'left': None, 'right': None}
        self.gt_occ = {'left': None, 'right': None}
        self.corr_pts = {'left': None, 'right': None}
        self.gt_corr_pts = {'left': None, 'right': None}
        self.refined_pts = {'left': None, 'right': None}
        self.posed_pts = {'left': None, 'right': None}
        self.vis_pts = {'left': None, 'right': None}

        self.num_corr_pts = {'left': 0, 'right': 0}
        self.fwd_point_weights = {'left': None, 'right': None}
        self.pred_dist = {'left': None, 'right': None}
        self.grad_norm = {'left': None, 'right': None}


    @torch.no_grad()
    def visualize(self, vis_loader, mesh_dir='mesh'):
        self.model.eval()

        eval_list = defaultdict(list)

        for idx, data in enumerate(tqdm.tqdm(vis_loader)):
            
            if idx == 0:
                can_mesh = self.generate_can_mesh(data)
                can_mesh = simplify_mesh(can_mesh, 8000, 5.)

                can_mesh.export(os.path.join(mesh_dir, 'can_mesh.ply'))
                can_vertices = torch.Tensor(can_mesh.vertices).unsqueeze(0).to(device=self.device)
                can_faces = can_mesh.faces
             
            # WARNING
            data[1]['right']['can_points'] = can_vertices.repeat_interleave(17, 0)
            data[1]['left']['can_points'] = can_vertices.repeat_interleave(17, 0)
            
            self.forward_pass(data)

            # covert verts to world coords
            posed_vertices_list = {'left': None, 'right': None}
                                
            for side in ['left', 'right']:
                posed_verts = self.vis_pts[side] 
                camera_params = data[1][side]['camera_params']

                posed_verts = torch.bmm(posed_verts.float(), camera_params['R']) 

                posed_verts += camera_params['root_xyz'].unsqueeze(1)

                if side == 'left':
                    posed_verts *= torch.Tensor([-1., 1., 1.]).cuda()

                posed_verts = torch.bmm(posed_verts.float(), camera_params['R'].transpose(1,2)) 
                posed_verts += camera_params['T'].double().unsqueeze(1)

                posed_vertices_list[side] = posed_verts

            for b_idx in range(17): 
                out_dir = data[1]['right']['out_dir'][b_idx].replace('mesh', 'mesh_vis')
                out_fname = data[1]['right']['out_file'][b_idx]

                for side in ['left', 'right']:
                    posed_vertices = posed_vertices_list[side][b_idx].detach().cpu().numpy()
                    posed_mesh = trimesh.Trimesh(posed_vertices, can_faces, process=False)

                    trimesh.repair.fix_normals(posed_mesh)

                    if not os.path.exists(os.path.join(mesh_dir, out_dir)):
                        os.system('mkdir -p ' + os.path.join(mesh_dir, out_dir))

                    print(os.path.join(mesh_dir, out_dir, out_fname + f'_{side}.ply'))

                    posed_mesh.export(os.path.join(mesh_dir, out_dir, out_fname + f'_{side}.ply'))

    def _eval_lbs_mode(self):
        self.model.fwd_lbs.require_grad = False
        self.model.fwd_lbs.eval()

    def _train_mode(self):
        self.model.train()
        self._eval_lbs_mode()

    def forward_pass(self, data, input_can_points=None, compute_grad=False):

        img, two_hand_dict = data
        img = img.to(device=self.device)

        for side in ['left', 'right']:

            hand_dict = two_hand_dict[side] 

            can_vert = hand_dict['can_vertices'].to(device=self.device)

            if input_can_points is None:
                can_points = hand_dict['can_points'].to(device=self.device)
                can_eval = False
            else:
                can_points = input_can_points
                can_eval = True

            self.num_corr_pts[side] = can_vert.shape[1]

            org_can_vert = hand_dict['org_can_vertices'].to(device=self.device)
            org_can_face = hand_dict['can_faces'].to(device=self.device)
            can_rel_joints = hand_dict['can_rel_joints'].to(device=self.device)
            can_pose = hand_dict['can_pose'].to(device=self.device)
            gt_mano_verts = hand_dict['gt_mano_verts'].to(device=self.device)
            gt_mano_faces = hand_dict['gt_mano_faces'].to(device=self.device)
            root_rot_mat = hand_dict['root_rot_mat'].to(device=self.device)
            camera_params = hand_dict['camera_params']

            fwd_transformation = hand_dict['fwd_transformation'].to(device=self.device)
            inv_transformation = hand_dict['inv_transformation'].to(device=self.device)
            can_fwd_transformation = hand_dict['can_fwd_transformation'].to(device=self.device)
            can_inv_transformation = hand_dict['can_inv_transformation'].to(device=self.device)

            org_can_points = can_points
            
            can_points = torch.cat((can_vert, can_points), 1)

            for key in camera_params.keys():
                if key not in ['img_path']:
                    camera_params[key] = camera_params[key].to(device=self.device)

            with torch.no_grad():
                self.fwd_point_weights[side] = self.model.fwd_lbs(can_points, org_can_vert)

                for idx in range(can_points.shape[0]):
                    can_root = hand_dict['can_joints_root'][idx].cuda()

                    can_points[idx] -= can_root

                posed_points = can2posed_points(can_points, self.fwd_point_weights[side], fwd_transformation.squeeze(), can_inv_transformation.squeeze())

            if side == 'left':
                self.model.shape_net(img)

            if compute_grad:
                self.posed_pts[side], self.pred_dist[side], self.grad_norm[side] = self.model.shape_net.query(posed_points, can_points, self.fwd_point_weights[side], root_rot_mat, camera_params, fixed=True, side=side, compute_grad=True) 
            else:
                self.posed_pts[side], self.pred_dist[side] = self.model.shape_net.query(posed_points, can_points, self.fwd_point_weights[side], root_rot_mat, camera_params, fixed=True, side=side) 

            self.refined_pts[side] = -self.pred_dist[side] + self.posed_pts[side]

            occ_points = self.refined_pts[side]

            # backward start
            inv_occ_points = occ_points + self.pred_dist[side]

            inv_occ_points = torch.bmm(inv_occ_points.float(), camera_params['R']) 
            inv_occ_points = torch.bmm(xyz_to_xyz1(inv_occ_points), root_rot_mat.transpose(1,2))[:, :, :3] 

            inv_occ_points = posed2can_points(inv_occ_points, self.fwd_point_weights[side], inv_transformation.squeeze(), can_fwd_transformation.squeeze())

            for idx in range(inv_occ_points.shape[0]):
                can_root = hand_dict['can_joints_root'][idx].cuda()
                inv_occ_points[idx] += can_root

                if can_eval:
                    can_points[idx] += can_root

            if not can_eval:
                occupancy = torch.sigmoid(self.model.leap_occupancy_decoder(
                    can_points=inv_occ_points, point_weights=self.fwd_point_weights[side], rot_mats=can_pose, rel_joints=can_rel_joints))
            else:
                occupancy = torch.sigmoid(self.model.leap_occupancy_decoder(
                    can_points=can_points, point_weights=self.fwd_point_weights[side], rot_mats=can_pose, rel_joints=can_rel_joints))

            gt_occ = torch.zeros((inv_occ_points.shape[0], inv_occ_points.shape[1])).to(device=self.device)
            can_mesh = seal(trimesh.Trimesh(org_can_vert[0].detach().cpu().numpy(), org_can_face[0].detach().cpu().numpy(), process=False))

            if can_eval:
                inv_occ_points = can_points

            for b_idx in range(inv_occ_points.shape[0]):
                occ_points_curr = inv_occ_points[b_idx].clone()
                occ_points_curr = occ_points_curr.detach().cpu().numpy()
                gt_occ_curr = check_mesh_contains(can_mesh, occ_points_curr).astype(np.float32) 
                gt_occ[b_idx] = torch.Tensor(gt_occ_curr)

            self.occ[side] = occupancy[:, self.num_corr_pts[side]:]
            self.gt_occ[side] = gt_occ[:, self.num_corr_pts[side]:]
            self.corr_pts[side] = self.refined_pts[side][:, :self.num_corr_pts[side], :] 
            self.vis_pts[side] = self.refined_pts[side][:, self.num_corr_pts[side]:, :] 
            self.gt_corr_pts[side] = gt_mano_verts
            self.posed_pts[side] = self.posed_pts[side][:, :self.num_corr_pts[side], :] 


    def compute_eval_loss(self, data):

        self.model.eval()
        
        self.forward_pass(data)

        eval_loss_dict = {
            'left_iou': self.compute_iou(self.occ['left'] >= 0.5, self.gt_occ['left'] >= 0.5).mean(),
            'left_corr_l2': self.corr_loss(self.corr_pts['left'], self.gt_corr_pts['left']),
            'left_corr_loss_before_shape': self.corr_loss(self.posed_pts['left'], self.gt_corr_pts['left']),
            'right_iou': self.compute_iou(self.occ['right'] >= 0.5, self.gt_occ['right'] >= 0.5).mean(),
            'right_corr_l2': self.corr_loss(self.corr_pts['right'], self.gt_corr_pts['right']),
            'right_corr_loss_before_shape': self.corr_loss(self.posed_pts['right'], self.gt_corr_pts['right']),
            }

        eval_loss_dict['iou'] = (eval_loss_dict['left_iou'] + eval_loss_dict['right_iou']) / 2
        eval_loss_dict['corr_l2'] = (eval_loss_dict['left_corr_l2'] + eval_loss_dict['right_corr_l2']) / 2
        eval_loss_dict['corr_loss_before_shape'] = (eval_loss_dict['left_corr_loss_before_shape'] + eval_loss_dict['right_corr_loss_before_shape']) / 2

        return eval_loss_dict

    def compute_loss(self, data):

        self._train_mode()

        self.forward_pass(data, compute_grad=False)

        loss_dict = {
            'left_occ_loss': self.occ_loss(self.occ['left'], self.gt_occ['left']),
            'left_corr_loss': self.corr_loss(self.corr_pts['left'], self.gt_corr_pts['left']),
            'left_corr_loss_before_shape': self.corr_loss(self.posed_pts['left'], self.gt_corr_pts['left']),
            #'left_smoothness': self.grad_norm['left'],
            'right_occ_loss': self.occ_loss(self.occ['right'], self.gt_occ['right']),
            'right_corr_loss': self.corr_loss(self.corr_pts['right'], self.gt_corr_pts['right']),
            'right_corr_loss_before_shape': self.corr_loss(self.posed_pts['right'], self.gt_corr_pts['right']),
            #'right_smoothness': self.grad_norm['right'],
        } 
        
        loss_dict['total_loss'] = 10 * loss_dict['left_corr_loss'] + loss_dict['left_occ_loss'] + 10 * loss_dict['right_corr_loss'] + loss_dict['right_occ_loss']

        #print('left ', loss_dict['left_occ_loss'].item(), loss_dict['left_corr_loss'].item(), loss_dict['left_corr_loss_before_shape'].item())#, loss_dict['left_smoothness'].item())
        #print('right ', loss_dict['right_occ_loss'].item(), loss_dict['right_corr_loss'].item(), loss_dict['right_corr_loss_before_shape'].item()) #, loss_dict['right_smoothness'].item())

        return loss_dict

    @staticmethod
    def compute_iou(occ1, occ2):
        """ Computes the Intersection over Union (IoU) value for two sets of
        occupancy values.

        Args:
            occ1 (tensor): first set of occupancy values
            occ2 (tensor): second set of occupancy values
        """
        # Also works for 1-dimensional data
        if len(occ1.shape) >= 2:
            occ1 = occ1.reshape(occ1.shape[0], -1)
        if len(occ2.shape) >= 2:
            occ2 = occ2.reshape(occ2.shape[0], -1)

        # Convert to boolean values
        occ1 = (occ1 >= 0.5)
        occ2 = (occ2 >= 0.5)

        # Compute IOU
        area_union = (occ1 | occ2).float().sum(axis=-1)
        area_intersect = (occ1 & occ2).float().sum(axis=-1)

        iou = (area_intersect / area_union)

        return iou


    @torch.no_grad()
    def eval_points(self, data, pts, pts_batch_size=240000):
        p_split = torch.split(pts, pts_batch_size)
        occ_hats = []

        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            pi = pi.repeat_interleave(17, 0)

            self.forward_pass(data, input_can_points=pi)
            occ_hat = self.occ['right'][0]
            print(occ_hat.max())

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat


    @torch.no_grad()
    def extract_mesh(self, occ_hat, threshold=0.20, padding=0.14):

        n_x, n_y, n_z = occ_hat.shape
        box_size = 0.1 + padding

        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)

        batch_size = 1
        shape = occ_hat.shape

        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)

        vertices -= 0.5
        vertices -= 1

        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)

        mesh = trimesh.Trimesh(vertices, triangles,
                               process=False)

        return mesh
  

    @torch.no_grad()
    def generate_can_mesh(self, data, resolution0=32, upsampling_steps=2, threshold=0.20, padding=0.14):

        box_size = 0.1 + padding

        mesh_extractor = MISE(
            resolution0, upsampling_steps, threshold)

        points = mesh_extractor.query()

        if upsampling_steps == 0:
            nx = resolution0
            pointsf = box_size * make_3d_grid(
                    (-0.5,)*3, (0.5,)*3, (nx,)*3)
            values = self.eval_points(data, pointsf).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)

        else:
            while points.shape[0] != 0:
                pointsf = torch.FloatTensor(points).to(self.device)

                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)

                values = self.eval_points(
                    data, pointsf).cpu().numpy()

                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        mesh = self.extract_mesh(value_grid)

        return mesh


