import os
import copy
import torch
import numpy as np
import torch.nn as nn 
import torchvision.transforms as transforms

from modules import geometry
from modules.gcn import GraphLayer
from modules.hg_filters import HGFilter
from modules.projection import joints_to_img_coord

def hand_joint_graph(v_num=21):
    graph = torch.zeros((v_num,v_num))
    edges = torch.tensor([[0,13],
                            [13,14],
                            [14,15],
                            [15,16],
                            [0,1],
                            [1,2],
                            [2,3],
                            [3,17],
                            [0,4],
                            [4,5],
                            [5,6],
                            [6,18],
                            [0,10],
                            [10,11],
                            [11,12],
                            [12,19],
                            [0,7],
                            [7,8],
                            [8,9],
                            [9,20]])
    graph[edges[:,0], edges[:,1]] = 1.0

    return graph


class JointFlowNet(nn.Module):

    def __init__(self, seq_len=17, basis_num=6, num_joints=21):
        super(JointFlowNet, self).__init__()
        
        self.seq_len = seq_len
        self.basis_num = basis_num 
        self.num_joints = num_joints

        self.image_enc = HGFilter(2, 2, 3, 64, 'group', 'no_down', False)
        self.pos_enc = nn.Linear(3, 128)

        self.gcn = GraphLayer(in_dim=259, out_dim=128, 
                              graph_L=hand_joint_graph().to('cuda'), graph_k=2, 
                              graph_layer_num=3, drop_out=0)
       
        self.refiner_x = nn.Sequential(
                            nn.Conv1d(128, 96, 3, stride=2),
                            nn.ReLU(False),
                            nn.Conv1d(96, 64, 3, stride=2),
                            nn.ReLU(False),
                            nn.Conv1d(64, 64, 3, stride=1),
                         )
        self.refiner_y = copy.deepcopy(self.refiner_x) 
        self.refiner_z = copy.deepcopy(self.refiner_x) 

        self.final_layer = nn.Sequential(
                              nn.Linear(78, 128),
                              nn.ReLU(False),
                              nn.Linear(128, 64),
                              nn.ReLU(False),
                              nn.Linear(64, 14),
                           )

        # Create basis vectors (refer to FOF [Feng et al., NeurIPS 2022])
        TOTAL_NUM_SAMPLE = 512

        z = (torch.arange(TOTAL_NUM_SAMPLE, dtype=torch.float32, device='cuda')-255.5) / 256
        z = torch.arange(self.basis_num + 1, dtype=torch.float32, device='cuda').view(1, self.basis_num+1) * z.view(TOTAL_NUM_SAMPLE, 1) * np.pi
        z = torch.cat([ z, z-(np.pi/2) ], dim=1)

        self.z = torch.cos(z)
        self.sampled_indices = (torch.linspace(0.25, 0.75, self.seq_len, device='cuda') * TOTAL_NUM_SAMPLE).long()

    def forward(self, imgs, joints, camera_params):

        # obtain initial joint coefficients
        init_coef = torch.zeros(((self.basis_num + 1) * 2, self.seq_len, self.num_joints * 3)).cuda() 
        discrete_basis = self.z[self.sampled_indices].T

        for i in range(init_coef.shape[0]):
            init_coef[i] = (discrete_basis[i].unsqueeze(0) * joints.reshape(-1, self.seq_len)).T 

        init_coef = torch.trapz(init_coef, dim=1)
        init_coef = init_coef.reshape((self.basis_num + 1) * 2, self.num_joints, 3) / self.seq_len

        # extract pixel-aligned features
        img_feat, _ = self.image_enc(imgs.permute(0, 3, 1, 2).float())
        img_feat = torch.cat([img_feat[0], img_feat[1]], axis=1)

        proj_img_p = joints_to_img_coord(joints, camera_params).to('cuda') 
        sampled_features = geometry.index(img_feat, (proj_img_p - 128) / 128)  # sample using normalized coordinates

        # refine the initial coefficients
        joints_feat = self.pos_enc(joints)
        joints_feat = torch.cat([sampled_features.transpose(1, 2), joints_feat, joints], axis=-1)

        joints_feat = self.gcn(joints_feat).permute(1, 2, 0)

        coef_x = self.refiner_x(joints_feat)   
        coef_y = self.refiner_y(joints_feat)   
        coef_z = self.refiner_z(joints_feat)

        coef = torch.cat([coef_x, coef_y, coef_z], dim=-1)
        coef = torch.cat([coef.transpose(0, 1), init_coef], dim=0)

        coef = self.final_layer(coef.permute(1, 2, 0)).permute(2, 0, 1) 

        # Multiply fourier coefficients with sine and cosine basis vectors
        joint_flow = torch.einsum("dc, chw -> dhw", self.z, coef)
        joint_flow = joint_flow[self.sampled_indices, :, :]
        
        return joint_flow
