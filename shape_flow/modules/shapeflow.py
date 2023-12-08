import torch.nn as nn
import sys
import cv2
import numpy as np
import torchvision
import torch
import torch.nn.functional as F

from .encoders import BaseModule
from .hg_filters import HGFilter

import sys
from modules import geometry


def export_point_cloud(pts, f_name): 
    import open3d as o3d
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(pts)  
    o3d.io.write_point_cloud(f"debug/{f_name}.ply", pcd)  


def xyz_to_xyz1(xyz):
    """ Convert xyz vectors from [BS, ..., 3] to [BS, ..., 4] for matrix multiplication
    """
    ones = torch.ones([*xyz.shape[:-1], 1], device=xyz.device)
    return torch.cat([xyz, ones], dim=-1)


def pts_to_img_coord(p, root_rot_mat, camera_params, side='right'): 

    img_p = torch.bmm(xyz_to_xyz1(p), root_rot_mat)[:, :, :3]

    if side == 'left':
        trans_img_p = torch.bmm(img_p.float(), camera_params['R'].transpose(1,2)) 
        trans_img_p *= torch.Tensor([-1., 1., 1.]).cuda()
        
    else:
        trans_img_p = torch.bmm(img_p.float(), camera_params['R'].transpose(1,2)) 
    
    img_p = img_p + (camera_params['root_xyz'].unsqueeze(1))  

    if side == 'left':
        img_p *= torch.Tensor([-1., 1., 1.]).cuda()

    img_p = torch.bmm(img_p, camera_params['R'].double().transpose(1,2)) 
    img_p += camera_params['T'].double().unsqueeze(1) # this aligns well with world-coordinate mesh 

    img_p = torch.bmm(img_p, camera_params['camera'].transpose(1,2))
    
    proj_img_p = torch.zeros((img_p.shape[0], img_p.shape[1], 2)).cuda()
    for i in range(img_p.shape[0]):
        proj_img_p[i] = img_p[i, :, :2] / img_p[i, :, 2:] 

    norm_z = torch.ones(img_p[:, :, 2].shape).cuda() - img_p[:, :, 2]

    return trans_img_p, proj_img_p, norm_z 


# inherit both LEAP and LVD base networks
class NetworkBase(BaseModule):
    def __init__(self):
        super(BaseModule, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                m.weight.data.normal_(0.0, 0.02)
            except:
                for i in m.children():
                    i.apply(self._weights_init_fn)
                return
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer


class ShapeFlow(NetworkBase):

    def __init__(self, input_point_dimensions=3, input_channels=3, pred_dimensions=3):
        super(NetworkBase, self).__init__()

        self._name = 'DeformationNet'

        self.image_filter = HGFilter(2, 2, input_channels, 64, 'group', 'no_down', False)

        input_point_dimensions = 3 
        pred_dimensions = 32 * 3

        self.temp_conv1_x = nn.utils.weight_norm(nn.Conv1d(166, 128, kernel_size=3, stride=2,  bias=True))
        self.temp_conv2_x = nn.utils.weight_norm(nn.Conv1d(128, 128, kernel_size=3, stride=2,  bias=True))
        self.temp_conv3_x = nn.utils.weight_norm(nn.Conv1d(128, 14, kernel_size=3, stride=2,  bias=True))

        self.temp_conv1_y = nn.utils.weight_norm(nn.Conv1d(166, 128, kernel_size=3, stride=2,  bias=True))
        self.temp_conv2_y = nn.utils.weight_norm(nn.Conv1d(128, 128, kernel_size=3, stride=2,  bias=True))
        self.temp_conv3_y = nn.utils.weight_norm(nn.Conv1d(128, 14, kernel_size=3, stride=2,  bias=True))

        self.temp_conv1_z = nn.utils.weight_norm(nn.Conv1d(166, 128, kernel_size=3, stride=2,  bias=True))
        self.temp_conv2_z = nn.utils.weight_norm(nn.Conv1d(128, 128, kernel_size=3, stride=2,  bias=True))
        self.temp_conv3_z = nn.utils.weight_norm(nn.Conv1d(128, 14, kernel_size=3, stride=2,  bias=True))

        TOTAL_NUM_SAMPLE = 512
        z = (torch.arange(TOTAL_NUM_SAMPLE, dtype=torch.float32, device='cuda')-255.5)/256
        z = torch.arange(7, dtype=torch.float32, device='cuda').view(1, 7) * z.view(TOTAL_NUM_SAMPLE, 1) * np.pi
        z = torch.cat([ z, z-(np.pi/2) ], dim=1)  
        self.z = torch.cos(z)
                                                        
        self.sampled_indices = (torch.linspace(0.25, 0.75, 17, device='cuda') * TOTAL_NUM_SAMPLE).long()

    def forward(self, image):
        self.im_feat_list, self.normx = self.image_filter(image)
        return

    @torch.backends.cudnn.flags(enabled=False)
    def query(self, points, can_points, lbs_weights, root_rot_mat, camera_params, scale=1, fixed=True, side='right', compute_grad=False):

        trans_img_p, points, norm_z = pts_to_img_coord(points, root_rot_mat, camera_params, side)
        trans_img_p.requires_grad_(True)
        xy = (points - 128) / 128
        xy = torch.cat((xy, norm_z.unsqueeze(-1)), -1)

        if side == 'right':
            class_cond = torch.Tensor([0, 1]).cuda()
        else:
            class_cond = torch.Tensor([1, 0]).cuda()
                    
        class_cond = torch.repeat_interleave(class_cond, 8).unsqueeze(0).unsqueeze(0)
        class_cond = torch.repeat_interleave(class_cond, xy.shape[0], dim=0)
        class_cond = torch.repeat_interleave(class_cond, xy.shape[1], dim=1)

        lbs_weights.requires_grad_(True)
        can_points *= scale

        if fixed:
            intermediate_preds_list = torch.cat((trans_img_p, can_points, lbs_weights, class_cond) , 2).transpose(2, 1)

        else:
            print('Please use fixed=True.')

        for j, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [intermediate_preds_list, geometry.index(im_feat.float(), xy.float())]
            intermediate_preds_list = torch.cat(point_local_feat_list, 1)

        intermediate_preds_list = intermediate_preds_list.permute(2, 1, 0) # 1024 x 166 x 17 P x F x T

        fourier_x = F.relu(self.temp_conv1_x(intermediate_preds_list))
        fourier_x = F.relu(self.temp_conv2_x(fourier_x))
        fourier_x = self.temp_conv3_x(fourier_x)

        fourier_y = F.relu(self.temp_conv1_y(intermediate_preds_list))
        fourier_y = F.relu(self.temp_conv2_y(fourier_y))
        fourier_y = self.temp_conv3_y(fourier_y)

        fourier_z = F.relu(self.temp_conv1_z(intermediate_preds_list))
        fourier_z = F.relu(self.temp_conv2_z(fourier_z))
        fourier_z = self.temp_conv3_z(fourier_z)

        x = torch.cat( [fourier_x, fourier_y, fourier_z] , dim=-1)
        x = x.squeeze()
        x = x.reshape(x.shape[0], 14, 3)
        x = x.permute(1,0,2)
        
        x = torch.einsum("dc, chw -> dhw", self.z, x)  
        x = x[self.sampled_indices, :, :]  
        x = x/10

        if side == 'left':
            trans_img_p = trans_img_p * torch.Tensor([-1., 1., 1.]).cuda() 
            x = x * torch.Tensor([-1., 1., 1.]).cuda() 

        if compute_grad:
            return trans_img_p, x, grad_deform

        return trans_img_p, x

    @classmethod
    def from_cfg(cls, config):
        model = cls()

        return model


if __name__ == '__main__':
    x = ShapeNet()
    print(x)
