from torch.utils.tensorboard import SummaryWriter

import os
import math
import random
import torch
import pickle
import argparse
import numpy as np
import os.path as osp
import torch.nn as nn 
import torch.optim as optim

from glob import glob
from tqdm import tqdm

from data import InterHandSeqDataset, TrainSampler, TestSampler, preprocess_batch
from model import JointFlowNet

# sequence length is fixed to 17 (following OFlow)
SEQ_LENGTH = 17

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--hand_mode', type=str, help='Hand mode (TO BE DELETED)', default='left')
parser.add_argument('--save_path', type=str, help='Path to save model weights', default='/data/jihyun/3d_hand_fourier_polish/generated/interhand_v3_fixed_basis_coef')
parser.add_argument('--data_path', type=str, help='Dataset path', default='/data/hand_data/AFOF/InterHand_processed_intag_seq')
parser.add_argument('--joints_path', type=str, help='Initial joints path', default='/workspace/IntagHand/joints/seq_test')
parser.add_argument('--max_epochs', type=int, help='Number of maximum epochs', default=30)
parser.add_argument('--print_it', type=int, help='Print statistics every N iterations', default=10)
parser.add_argument('--val_it', type=int, help='Run validation every N iterations', default=1000)
parser.add_argument('--seed', type=int, help='Random seed number', default=12345)

args = parser.parse_args()
hand_mode = args.hand_mode
save_path = args.save_path
max_epochs = args.max_epochs

random.seed(args.seed)


def load_ckp(checkpoint_fpath, model, optimizer=None):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, checkpoint['it'], optimizer

    else:
        return model, checkpoint['it']


if __name__ == '__main__':

    writer = SummaryWriter(save_path)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    print('Start Training', save_path, hand_mode)
    
    # initialize datasets and dataloaders
    train_dataset = InterHandSeqDataset('train', hand_mode, args.data_path, args.joints_path)
    val_dataset = InterHandSeqDataset('val', hand_mode, args.data_path, args.joints_path)

    train_loader = torch.utils.data.DataLoader(   
            train_dataset,  
            batch_size=SEQ_LENGTH,
            num_workers=4,
            sampler=TrainSampler(train_dataset))

    val_loader = torch.utils.data.DataLoader(   
            val_dataset,  
            batch_size=SEQ_LENGTH,
            num_workers=4,
            sampler=TestSampler(val_dataset, seq_num=500)) # evaluate on 500 val samples

    # initialize model and optimizer
    jf_net = JointFlowNet()
    jf_net.to('cuda')

    optimizer = optim.Adam(jf_net.parameters(), lr=1e-4)

    min_validation_loss = 10000000
    epoch_it, it = -1, -1

    while epoch_it < max_epochs:
        epoch_it += 1

        # training loop
        for batch in train_loader:
            it += 1

            jf_net.train()
            
            imgs, joints, camera_params, gt_joints = preprocess_batch(batch)

            optimizer.zero_grad()

            refined_joints = jf_net(imgs, joints, camera_params)

            ref_loss = torch.mean(torch.linalg.norm((refined_joints - gt_joints), ord=2, dim=-1))
            unref_loss = torch.mean(torch.linalg.norm((joints - gt_joints), ord=2, dim=-1)) # for statistics

            ref_loss.backward()
            optimizer.step()

            # print statistics
            if it > 0 and it % args.print_it == 0:
                print(f'[epoch {epoch_it}, iter {it}] train loss: {ref_loss.item() :.8f}     noisy joints loss: {unref_loss.item() :.8f}')
                writer.add_scalar("Loss/train", ref_loss.item(), it)

            # validataion loop
            if it > 0 and it % args.val_it == 0:
                validation_loss = 0.0
                validation_unref_loss = 0.0
                valid_counter = 0
                jf_net.eval()

                with torch.no_grad():
                    for val_batch in val_loader:

                        imgs, joints, camera_params, gt_joints = preprocess_batch(val_batch)

                        refined_joints = jf_net(imgs, joints, camera_params)
                        
                        ref_loss = torch.mean(torch.linalg.norm((refined_joints - gt_joints), ord=2, dim=-1))
                        unref_loss = torch.mean( torch.linalg.norm((joints - gt_joints), ord=2, dim=-1) ) 

                        validation_loss += ref_loss.item()
                        validation_unref_loss += unref_loss.item()

                        valid_counter = valid_counter + 1

                avg_valid_loss = validation_loss/valid_counter
                print(f'[epoch {epoch_it}, iter {it}] valid loss: {avg_valid_loss :.8f}   noisy joints loss: {validation_unref_loss/valid_counter :.8f}')
                writer.add_scalar("Loss/valid", avg_valid_loss, it)

                # save when minimum validation loss
                if min_validation_loss > avg_valid_loss:
                    print(f'Saving model. Lower validation loss {avg_valid_loss :.8f} than previous validation loss {min_validation_loss :.8f}')

                    min_validation_loss = avg_valid_loss
                    
                    checkpoint = {
                        'epoch_it': epoch_it,
                        'it': it,
                        'state_dict': jf_net.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    torch.save(checkpoint, osp.join(save_path, f'joint_flow_{it}.pth'))

                jf_net.train()
        
    writer.flush()
    writer.close()
