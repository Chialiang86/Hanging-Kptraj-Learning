import argparse, yaml, os, glob, time, cv2
import open3d as o3d
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from utils.bullet_utils import get_matrix_from_pose
from utils.training_utils import get_model_module, get_dataset_module, optimizer_to_device, normalize_pc, kl_annealing
from utils.testing_utils import refine_waypoint_rotation, robot_kptraj_hanging, recover_trajectory
from pointnet2_ops.pointnet2_utils import furthest_point_sample

from PIL import Image
from scipy.spatial.transform import Rotation as R

def main(args):
    dataset_dir = args.dataset_dir
    config_file = args.config

    assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
    assert os.path.exists(config_file), f'{config_file} not exists'


    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for training
    dataset_mode = 0 if 'absolute' in dataset_dir else 1
    dataset_name = config['dataset_module']
    dataset_class_name = config['dataset_class']
    dataset_dataset_inputs = config['dataset_inputs']
    
    # training batch and iters
    batch_size = 1
    sample_num_points = config['dataset_inputs']['sample_num_points']
    wpt_dim = config['dataset_inputs']['wpt_dim']

    dataset_class = get_dataset_module(dataset_name, dataset_class_name)

    # ================== Simulator ==================

    hook_pose = [
        0.5,
        -0.1,
        1.3,
        4.329780281177466e-17,
        0.7071067811865475,
        0.7071067811865476,
        4.329780281177467e-17
    ]

    train_set = dataset_class(dataset_dir=f'{dataset_dir}/val', **dataset_dataset_inputs)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_set.print_data_shape()
    train_batches = enumerate(train_loader, 0)
    for i_batch, (sample_pcds, sample_affords, sample_trajs) in tqdm(train_batches, total=len(train_loader)):

        sample_trajs = sample_trajs.cpu().detach().numpy().tolist()
        points = sample_pcds.cpu().detach().squeeze().numpy()
        sample_affords = sample_affords[0].cpu().detach().squeeze().numpy()
        sample_affords = (sample_affords - np.min(sample_affords)) / (np.max(sample_affords) - np.min(sample_affords))
        colors = cv2.applyColorMap((255 * sample_affords).astype(np.uint8), colormap=cv2.COLORMAP_JET).squeeze()

        geometries = []

        contact_point = sample_pcds[0, 0].cpu().detach().squeeze().numpy()
        contact_point_coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        contact_point_coor.translate(contact_point.reshape((3, 1)))
        geometries.append(contact_point_coor)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255)
        geometries.append(point_cloud)

        for i, sample_traj in enumerate(sample_trajs):
            for wpt_i, wpt in enumerate(sample_traj):
                wpt_trans = get_matrix_from_pose(wpt)
                contact_point_coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                contact_point_coor.transform(wpt_trans)
                geometries.append(contact_point_coor)

        o3d.visualization.draw_geometries(geometries)

if __name__=="__main__":

    default_dataset = [
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000",
    ]

    parser = argparse.ArgumentParser()
    # about dataset
    parser.add_argument('--dataset_dir', '-dd', type=str, default=default_dataset[0])

    # other info
    parser.add_argument('--config', '-cfg', type=str, default='../config/traj_af_mutual/traj_fusion_mutual_noise.yaml')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.2)
    parser.add_argument('--verbose', '-vb', action='store_true')
    args = parser.parse_args()

    main(args)