from torch.utils.tensorboard import SummaryWriter

import os
import torch
import random
import argparse
import os.path as osp
import open3d as o3d

from tqdm import tqdm
from dependencies.manolayer import ManoLayer

#import joint_visualizer
from data import InterHandSeqDataset, preprocess_batch
from model import JointFlowNet

# sequence length is fixed to 17 (following OFlow)
SEQ_LENGTH = 17

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--hand_mode', type=str, help='Hand mode (TO BE DELETED)', default='left')
parser.add_argument('--save_path', type=str, help='Path to saved model weights', default='/data/jihyun/3d_hand_fourier_polish/generated/interhand_v3_fixed_basis_coef')
parser.add_argument('--data_path', type=str, help='Dataset path', default='/data/hand_data/AFOF/InterHand_processed_intag_seq')
parser.add_argument('--joints_path', type=str, help='Initial joints path', default='/workspace/IntagHand/joints/seq_test')
parser.add_argument('--seed', type=int, help='Random seed number', default=12345)

args = parser.parse_args()
hand_mode = args.hand_mode
save_path = args.save_path

random.seed(args.seed)


def load_ckpt(path, model, optimizer=None):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, checkpoint['it'], optimizer

    else:
        return model, checkpoint['it']


def export_point_cloud(pts, f_name):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(f"{f_name}.ply", pcd)


def test(save_path, model_name, hand_mode):
    print('Start Prediction', save_path, model_name, hand_mode)
    
    pred_dir = osp.join(save_path, 'pred') 

    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)

    vis_dir = f'{save_path}/vis'
    if not os.path.isdir(vis_dir):
        os.makedirs(vis_dir)

    test_dataset = InterHandSeqDataset('test', hand_mode, args.data_path, args.joints_path)

    # create subdirs
    for data_path in test_dataset.data_list:
        subdir = '/'.join(data_path.split('/')[:-1])

        save_pred_dir = osp.join(pred_dir, subdir)
        if not os.path.isdir(save_pred_dir):
            os.makedirs(save_pred_dir)
        
        save_vis_dir = osp.join(vis_dir, subdir)
        if not os.path.isdir(save_vis_dir):
            os.makedirs(save_vis_dir)

    test_loader = torch.utils.data.DataLoader(   
            test_dataset,  
            batch_size=SEQ_LENGTH,
            num_workers=4,
            shuffle=False)

    jf_net = JointFlowNet().to('cuda')
    #jf_net, _ = load_ckpt(osp.join(save_path, model_name), jf_net)
    jf_net.eval()

    mpjpe = 0.0
    base_mpjpe = 0.0
    counter = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):

            counter += 1
            imgs, joints, camera_params, gt_joints, gt_midjoints = preprocess_batch(batch, test=True)

            ref_joints = jf_net(imgs, joints, camera_params)
            ref_joints -= ref_joints[:, 9, :].unsqueeze(1)  # following previous works

            err = torch.mean( torch.linalg.norm((ref_joints - gt_joints), ord=2, dim=-1) ) 
            mpjpe += err.item() * 1000

            err = torch.mean( torch.linalg.norm((joints - gt_joints), ord=2, dim=-1) )
            base_mpjpe += err.item() * 1000
            
            filename_list = camera_params['img_path']
                
            ref_joints = ref_joints + gt_midjoints 

            fname = camera_params['img_path']
            path = '/'.join(fname[0].split('/')[-4:])
            path = path.replace('.jpg', '')

            for i in range(ref_joints.shape[0]):
                export_point_cloud(ref_joints[i].cpu(), osp.join(pred_dir, f'{path}_{hand_mode}_{i}'))           
    mpjpe /= counter
    base_mpjpe /= counter

    print('counter', counter)
    print(f'MPJPE: {mpjpe :.8f}   Baseline MPJPE: {base_mpjpe :.8f}')

if __name__ == '__main__':
    test('backup', 'pose_refiner_63500.pth', 'left') 

