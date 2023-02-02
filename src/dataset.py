import os, glob, json
import numpy as np
import open3d as o3d
from pointnet2_ops.pointnet2_utils import furthest_point_sample

import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from utils.training_utils import get_model_module, optimizer_to_device, normalize_pc

class KptReconDataset(Dataset):
    def __init__(self, dataset_dir, num_steps=30, wpt_dim=6, enable_gravity=False):
        
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'

        traj_files = glob.glob(f'{dataset_dir}/*/*.json') # trajectory in 7d format

        traj_list = []
        self.max_num_waypoints = 0
        for traj_file in traj_files:
            f_traj = open(traj_file, 'r')
            traj = json.load(f_traj)
            waypoints = np.array(traj['trajectory'])
            self.max_num_waypoints = max(self.max_num_waypoints, waypoints.shape[0])
            traj_list.append(waypoints)

        self.size = len(traj_list)
        self.traj_len = num_steps
        self.wpt_dim = wpt_dim
        self.traj_array = np.zeros((self.size, self.traj_len, wpt_dim), dtype=np.float32) # N x T x 6
        # self.traj_array = np.zeros((self.size, self.max_num_waypoints, 6), dtype=np.float32) # N x T x 6
        for i in range(len(traj_list)):
            length = traj_list[i].shape[0]
            self.traj_array[i, :self.traj_len, :] = traj_list[i][:self.traj_len]
            self.traj_array[i] = traj_list[i]

            # if 'residual' in dataset_dir:
            #     self.traj_array[i,1:,:3] *= 1000 # relative position will be very small
            #     self.traj_array[i,1:,3:] *= (180.0 / np.pi) # relative rotation will be very small

    def print_data_shape(self):
        print(f"trajectory : {self.traj_array.shape}")

    def plot_distribution(self):
        x = self.traj_array[:, 0, 0]
        y = self.traj_array[:, 0, 1]
        z = self.traj_array[:, 0, 2]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(x, y, z, marker='o')
        ax.set_title('Ecludian Position (3D)')
        ax.set_xlabel('X label')
        ax.set_xlabel('Y label')
        ax.set_xlabel('Z label')
        plt.show()

        rx = self.traj_array[:, 0, 3]
        ry = self.traj_array[:, 0, 4]
        rz = self.traj_array[:, 0, 5]
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(rx, ry, rz, marker='o')
        ax.set_title('Ecludian Rotation (3D)')
        ax.set_xlabel('X label')
        ax.set_xlabel('Y label')
        ax.set_xlabel('Z label')
        plt.show()

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):

        waypoints = self.traj_array[index]

        return waypoints

class KptReconShapeDataset(Dataset):
    def __init__(self, dataset_dir, num_steps=30, wpt_dim=6, enable_gravity=False):
        
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'

        shape_files = glob.glob(f'{dataset_dir}/*/*.ply') # point cloud
        traj_files = glob.glob(f'{dataset_dir}/*/*.json') # trajectory in 7d format

        shape_list = []
        self.max_num_points = 0
        self.min_num_points = 1e10
        for shape_file in shape_files:
            pcd = o3d.io.read_point_cloud(shape_file)
            points = np.asarray(pcd.points, dtype=np.float32)
            self.max_num_points = max(self.max_num_points, points.shape[0])
            self.min_num_points = min(self.min_num_points, points.shape[0])
            shape_list.append(points)
        
        base_num = 500
        self.sample_num_points = base_num * (self.min_num_points // base_num)
        
        traj_list = []
        self.max_num_waypoints = 0
        for traj_file in traj_files:
            f_traj = open(traj_file, 'r')
            traj = json.load(f_traj)
            waypoints = np.array(traj['trajectory'], dtype=np.float32)
            self.max_num_waypoints = max(self.max_num_waypoints, waypoints.shape[0])
            traj_list.append(waypoints)

        self.type = "residual" if "residual" in dataset_dir else "absolute"
        self.size = len(traj_list)
        self.traj_len = num_steps
        self.wpt_dim = wpt_dim
        
        # assign to data
        self.shape_array = shape_list
        self.traj_array = traj_list

        # create array
        # TODO: check why "Nx4" point cloud not work
        # self.shape_array = np.zeros((self.size, self.max_num_points, 3), dtype=np.float32) # N x P x 4
        # for i in range(len(shape_list)):
        #     length = shape_list[i].shape[0]
        #     self.shape_array[i, :length, :3] = shape_list[i]
        #     self.shape_array[i, :length, 3:] = 1.0 # valid flag
        #     self.shape_array[i, length:, 3:] = 0.0 # valid flag

        # self.traj_array = np.zeros((self.size, self.traj_len, wpt_dim), dtype=np.float32) # N x T x 6
        # for i in range(len(traj_list)):
        #     length = traj_list[i].shape[0]
        #     self.traj_array[i, :self.traj_len, :] = traj_list[i][:self.traj_len]
        #     self.traj_array[i] = traj_list[i]

            # # TODO: do we need this? => maybe no
            # if 'residual' in dataset_dir:
            #     self.traj_array[i,1:,:3] *= 1000 # relative position will be very small
            #     self.traj_array[i,1:,3:] *= (180.0 / np.pi) # relative rotation will be very small
    
    def print_data_shape(self):
        print(f"shape : {len(self.shape_array)}")
        print(f"trajectory : {len(self.traj_array)}")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):

        points = self.shape_array[index]
        waypoints = self.traj_array[index]

        centroid_points, centroid, max_ratio = normalize_pc(points[:,:3]) # points will be in a unit sphere
        if self.type == "absolute":
            waypoints[:,:3] = (waypoints[:,:3] - centroid) / max_ratio
        elif self.type == "residual":
            waypoints[0,:3] = (waypoints[0,:3] - centroid) / max_ratio
            waypoints[1:,:3] = waypoints[1:,:3] / max_ratio
        else :
            print(f"dataset type undefined : {self.type}")
            exit(-1)
        # points[:,:3] = centroid_points
        
        points = torch.from_numpy(centroid_points).unsqueeze(0).to('cuda').contiguous()
        input_pcid = furthest_point_sample(points, self.sample_num_points).long().reshape(-1)  # BN
        points = points[0, input_pcid, :].squeeze()

        return points, waypoints

class KptReconAffordanceDataset(Dataset):
    def __init__(self, dataset_dir, num_steps=30, wpt_dim=6, enable_gravity=False):
        
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'

        shape_files = glob.glob(f'{dataset_dir}/*/*.npy') # point cloud with affordance score (Nx4), the first element is the contact point
        traj_files = glob.glob(f'{dataset_dir}/*/*.json') # trajectory in 7d format

        shape_list = []
        self.max_num_points = 0
        self.min_num_points = 1e10
        for shape_file in shape_files:
            pcd = np.load(shape_file)
            points = pcd[:,:3]
            self.max_num_points = max(self.max_num_points, points.shape[0])
            self.min_num_points = min(self.min_num_points, points.shape[0])
            shape_list.append(points)
        
        base_num = 500
        self.sample_num_points = base_num * (self.min_num_points // base_num)
        
        traj_list = []
        self.max_num_waypoints = 0
        for traj_file in traj_files:
            f_traj = open(traj_file, 'r')
            traj = json.load(f_traj)
            waypoints = np.array(traj['trajectory'], dtype=np.float32)
            self.max_num_waypoints = max(self.max_num_waypoints, waypoints.shape[0])
            traj_list.append(waypoints)

        self.type = "residual" if "residual" in dataset_dir else "absolute"
        self.size = len(traj_list)
        self.traj_len = num_steps
        self.wpt_dim = wpt_dim
        
        # assign to data
        self.shape_array = shape_list
        self.traj_array = traj_list

        # create array
        # TODO: check why "Nx4" point cloud not work
        # self.shape_array = np.zeros((self.size, self.max_num_points, 3), dtype=np.float32) # N x P x 4
        # for i in range(len(shape_list)):
        #     length = shape_list[i].shape[0]
        #     self.shape_array[i, :length, :3] = shape_list[i]
        #     self.shape_array[i, :length, 3:] = 1.0 # valid flag
        #     self.shape_array[i, length:, 3:] = 0.0 # valid flag

        # self.traj_array = np.zeros((self.size, self.traj_len, wpt_dim), dtype=np.float32) # N x T x 6
        # for i in range(len(traj_list)):
        #     length = traj_list[i].shape[0]
        #     self.traj_array[i, :self.traj_len, :] = traj_list[i][:self.traj_len]
        #     self.traj_array[i] = traj_list[i]

            # # TODO: do we need this? => maybe no
            # if 'residual' in dataset_dir:
            #     self.traj_array[i,1:,:3] *= 1000 # relative position will be very small
            #     self.traj_array[i,1:,3:] *= (180.0 / np.pi) # relative rotation will be very small
    
    def print_data_shape(self):
        print(f"shape : {len(self.shape_array)}")
        print(f"trajectory : {len(self.traj_array)}")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):

        points = self.shape_array[index]
        waypoints = self.traj_array[index]

        centroid_points, centroid, max_ratio = normalize_pc(points[:,:3]) # points will be in a unit sphere
        if self.type == "absolute":
            waypoints[:,:3] = (waypoints[:,:3] - centroid) / max_ratio
        elif self.type == "residual":
            waypoints[0,:3] = (waypoints[0,:3] - centroid) / max_ratio
            waypoints[1:,:3] = waypoints[1:,:3] / max_ratio
        else :
            print(f"dataset type undefined : {self.type}")
            exit(-1)
        # points[:,:3] = centroid_points
        
        points = torch.from_numpy(centroid_points).unsqueeze(0).to('cuda').contiguous()
        input_pcid = furthest_point_sample(points, self.sample_num_points).long().reshape(-1)  # BN
        points = points[0, input_pcid, :].squeeze()

        return points, waypoints

if __name__=="__main__":
    
    dataset_dir = "../data/traj_recon/hook-kptraj_1104_aug-absolute-30/01.16.13.53/train"
    dataset = KptReconDataset(dataset_dir)
    dataset.print_data_shape()
    dataset_dir = "../data/traj_recon/hook-kptraj_1104_aug-residual-30/01.16.13.55/train"
    dataset = KptReconDataset(dataset_dir)
    dataset.print_data_shape()
    dataset_dir = "../data/traj_recon/hook-kptraj_1104-absolute-30/01.16.13.41/train"
    dataset = KptReconDataset(dataset_dir)
    dataset.print_data_shape()
    dataset_dir = "../data/traj_recon/hook-kptraj_1104-residual-30/01.16.13.41/train"
    dataset = KptReconDataset(dataset_dir)
    dataset.print_data_shape()
    
    dataset_dir = "../data/traj_recon_shape/hook-kptraj_1104_aug-absolute-30/01.16.13.56/train"
    dataset = KptReconShapeDataset(dataset_dir)
    dataset.print_data_shape()
    dataset_dir = "../data/traj_recon_shape/hook-kptraj_1104_aug-residual-30/01.16.13.57/train"
    dataset = KptReconShapeDataset(dataset_dir)
    dataset.print_data_shape()
    dataset_dir = "../data/traj_recon_shape/hook-kptraj_1104-absolute-30/01.16.13.41/train"
    dataset = KptReconShapeDataset(dataset_dir)
    dataset.print_data_shape()
    dataset_dir = "../data/traj_recon_shape/hook-kptraj_1104-residual-30/01.16.13.42/train"
    dataset = KptReconShapeDataset(dataset_dir)
    dataset.print_data_shape()

