import os
import torch
import open3d as o3d


def joints_to_img_coord(joints, camera_params):
    # joints: n_batch x n_joints x 3
    img_p = joints + (camera_params['root_xyz'].unsqueeze(1))
    img_p = torch.bmm(img_p, camera_params['R'].transpose(1,2))
    img_p += camera_params['t'].double().unsqueeze(1)

    img_p = torch.bmm(img_p, camera_params['camera'].transpose(1,2))

    proj_img_p = torch.zeros((img_p.shape[0], img_p.shape[1], 2))

    for i in range(img_p.shape[0]):
        proj_img_p[i] = img_p[i, :, :2] / img_p[i, :, 2:]

    return proj_img_p


def export_point_cloud(pts, f_name):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(f"{f_name}.ply", pcd)


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        mano_layer['left'].shapedirs[:, 0, :] *= -1
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    import cv2
    import pickle
    import numpy as np
    import open3d as o3d
    import os.path as osp
    from dependencies.manolayer import ManoLayer

    mano_layer = {'right': ManoLayer(osp.join(os.getcwd(), 'dependencies/models/MANO_RIGHT.pkl'), center_idx=None),
                  'left': ManoLayer(osp.join(os.getcwd(), 'dependencies/models/MANO_LEFT.pkl'), center_idx=None)}
    fix_shape(mano_layer)

    img_dir = 'data/img'
    anno_dir = 'data/anno' 
    joints_dir = 'data/noisy_joints'

    f_name = '23328'

    # read input files
    img = cv2.imread(osp.join(img_dir, f'{f_name}.jpg'))   
    joints = np.asarray(o3d.io.read_point_cloud(osp.join(joints_dir, f'{f_name}_right.ply')).points)
    joints -= joints[0]
    
    with open(osp.join(anno_dir, f'{f_name}.pkl'), 'rb') as anno_file:
        anno = pickle.load(anno_file)

    # change format (torch tensor w/ batch dimension)
    joints = torch.Tensor(joints).unsqueeze(0).float()

    camera_params = {}
    for key in anno['camera'].keys():
        camera_params[key] = torch.Tensor(anno['camera'][key]).unsqueeze(0).float()

    mano_params = anno['mano_params']['right']
    # world coordinate joint using MANO
    handV, handJ = mano_layer['right'](torch.from_numpy(mano_params['R']).float(),
                                       torch.from_numpy(mano_params['pose']).float(),
                                       torch.from_numpy(mano_params['shape']).float(),
                                       trans=torch.from_numpy(mano_params['trans']).float())  # Vertices, surface points, gt joint points

    camera_params['root_xyz'] = torch.Tensor(handJ[0, 0]).unsqueeze(0).float()

    proj_img_p = joints_to_img_coord(joints, camera_params) 

    # check GT joints
    #joints = handJ - handJ[:, 0, :]
    #export_point_cloud(joints.numpy()[0], 'gt')
    #proj_img_p = joints_to_img_coord(joints, camera_params) 


    # for sanity check (visualization)
    for idx in range(proj_img_p.shape[1]):
        pt = proj_img_p[0, idx] # retrieve first batch
        if pt.min() < 0 or pt.max() > 255: continue
        import pdb;pdb.set_trace()
        print('check', idx)
        img[int(pt[1]), int(pt[0])] = [255, 0, 255]

    cv2.imwrite('check.jpg', img)




