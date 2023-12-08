import argparse
import os
import json
from glob import glob
from os.path import basename, join, splitext
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch

from leap.leap_hand_model import LEAPHandModel


@torch.no_grad()
def main(args):

    for split in args.splits:

        annot_f_name = f'InterHand2.6M_{split}_MANO_NeuralAnnot.json'
        annot_f_path = os.path.join(args.src_dataset_path, split, annot_f_name)

        bm_path = {'right': '/workspace/AFOF/leap/body_models/mano/models/MANO_RIGHT.pkl',
                   'left': '/workspace/AFOF/leap/body_models/mano/models/MANO_LEFT.pkl'}

        with open(annot_f_path) as annot_f:
            annots = json.load(annot_f)

        for i, seq_idx in enumerate(annots.keys()):
            print(f'Processing {i+1} out of {len(annots.keys())} seqs...')

            for idx in tqdm(annots[seq_idx].keys()):
                for side in ['left', 'right']:

                    annot = annots[seq_idx][idx][side]

                    if annot is None:
                        continue

                    #print(annot['pose']) # 48
                    #print(annot['shape']) # 10
                    #print(annot['trans']) # 3

                    b_size = 1  #45 - len(annot['pose'])

                    leap_body_model = LEAPHandModel(bm_path=bm_path[side], num_betas=len(annot['shape']), batch_size=b_size)
    
                    leap_body_model.set_parameters(
                        betas=torch.Tensor(annot['shape']).unsqueeze(0).repeat(b_size, 1),
                        pose_body=None,
                        pose_hand=torch.Tensor(annot['pose'][3:])) # they discarded the root pose in body model too
                    leap_body_model.forward_parametric_model()

                    to_save = {
                        'can_vertices': leap_body_model.can_vert,
                        'posed_vertices': leap_body_model.posed_vert,
                        'pose_mat': leap_body_model.pose_rot_mat, #V
                        'rel_joints': leap_body_model.rel_joints, #V
                        'fwd_transformation': leap_body_model.fwd_transformation,
                        'seq_idx': f'{seq_idx}',
                        'frame_idx': f'{idx}',
                        'side': f'{side}'
                    }

                    dir_path = join(args.dst_dataset_path, split)

                    Path(dir_path).mkdir(parents=True, exist_ok=True)

                    #print(f'Saving:\t{dir_path}')
                    for b_ind in range(b_size):
                        with open(join(dir_path, f'{seq_idx}_{idx}_{side}.npz'), 'wb') as file:
                            np.savez(file, **{key: to_np(val[b_ind]) for key, val in to_save.items()})

                    exit()

                    # for shape check
                    '''
                    shape_dir_path = join(args.dst_dataset_path, 'shapes')
                    Path(shape_dir_path).mkdir(parents=True, exist_ok=True)

                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(leap_body_model.can_vert.detach().cpu().numpy()[0])

                    shape_fname = f'{seq_idx}_{idx}_{side}.ply'
                    o3d.io.write_point_cloud(join(shape_dir_path, shape_fname), pcd)
                    '''

def to_np(variable):
    if torch.is_tensor(variable):
        variable = variable.detach().cpu().numpy()

    return variable


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocess AMASS dataset.')
    parser.add_argument('--src_dataset_path', type=str, default='InterHand',
                        help='Path to AMASS dataset.')
    parser.add_argument('--dst_dataset_path', type=str, default='InterHand_can',
                        help='Directory path to store preprocessed dataset.')
    parser.add_argument('--splits', type=list, default=['train', 'val', 'test'],#'train,val,test',
                        help='Subsets of AMASS to use, separated by comma.')
    parser.add_argument('--bm_dir_path', type=str, default='../body_models/mano/models',
                        help='Path to body model')

    main(parser.parse_args())
