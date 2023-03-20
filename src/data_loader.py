import argparse, yaml, os, glob, time
import open3d as o3d
import numpy as np
from tqdm import tqdm
from time import strftime
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchsummary import summary
from utils.training_utils import get_model_module, get_dataset_module, optimizer_to_device

import pybullet as p
from PIL import Image
from scipy.spatial.transform import Rotation as R
from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose, \
                               pose_6d_to_7d, pose_7d_to_6d, draw_coordinate

def main(args):
    dataset_dir = args.dataset_dir
    config_file = args.config

    assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
    assert os.path.exists(config_file), f'{config_file} not exists'

    dataset_dir = args.dataset_dir
    config_file = args.config

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for training
    dataset_name = config['dataset_module']
    dataset_class_name = config['dataset_class']
    
    # training batch and iters
    batch_size = config['batch_size']

    dataset_class = get_dataset_module(dataset_name, dataset_class_name)

    train_set = dataset_class(dataset_dir=f'{dataset_dir}/val', enable_traj=1, affordance_type=0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    train_set.print_data_shape()
    train_batches = enumerate(train_loader, 0)
    for i_batch, (sample_pcds, sample_trajs) in tqdm(train_batches, total=len(train_loader)):
        print(f'batch {i_batch}: pcd shape: {sample_pcds.shape}, traj shape: {sample_trajs.shape}')
    
    # train_set = dataset_class(dataset_dir=f'{dataset_dir}/train', enable_traj=0, affordance_type=1)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # train_set.print_data_shape()
    # train_batches = enumerate(train_loader, 0)
    # for i_batch, (sample_pcds, sample_affords) in tqdm(train_batches, total=len(train_loader)):
    #     print(f'batch {i_batch}: pcd shape: {sample_pcds.shape}, affordance shape: {sample_affords.shape}')

    # train_set = dataset_class(dataset_dir=f'{dataset_dir}/train', enable_traj=1, affordance_type=0)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # train_set.print_data_shape()
    # train_batches = enumerate(train_loader, 0)
    # for i_batch, (sample_pcds, sample_trajs) in tqdm(train_batches, total=len(train_loader)):
    #     print(f'batch {i_batch}: pcd shape: {sample_pcds.shape}, traj shape: {sample_trajs.shape}')

    # train_set = dataset_class(dataset_dir=f'{dataset_dir}/train', enable_traj=1, affordance_type=0)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # train_set.print_data_shape()
    # train_batches = enumerate(train_loader, 0)
    # for i_batch, (sample_pcds, sample_affords, sample_trajs) in tqdm(train_batches, total=len(train_loader)):
    #     print(f'batch {i_batch}: pcd shape: {sample_pcds.shape}, affordance shape: {sample_affords.shape}, traj shape: {sample_trajs.shape}')
    

if __name__=="__main__":

    default_dataset = [
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40/03.15.13.59-1000",
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40/03.15.14.01-1000"
    ]

    parser = argparse.ArgumentParser()
    # about dataset
    parser.add_argument('--dataset_dir', '-dd', type=str, default=default_dataset[1])

    # other info
    parser.add_argument('--config', '-cfg', type=str, default='../config/traj_af.yaml')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.2)
    parser.add_argument('--verbose', '-vb', action='store_true')
    args = parser.parse_args()

    main(args)