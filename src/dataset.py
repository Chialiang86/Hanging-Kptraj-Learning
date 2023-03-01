import os, glob, json
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from utils.training_utils import get_model_module, optimizer_to_device, normalize_pc

# class KptrajReconDataset(Dataset):
#     def __init__(self, dataset_dir, num_steps=30, wpt_dim=6, enable_gravity=False):
        
#         assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'

#         traj_files = glob.glob(f'{dataset_dir}/*/*.json') # trajectory in 7d format

#         traj_list = []
#         self.max_num_waypoints = 0
#         for traj_file in traj_files:
#             f_traj = open(traj_file, 'r')
#             traj = json.load(f_traj)
#             waypoints = np.array(traj['trajectory'])
#             self.max_num_waypoints = max(self.max_num_waypoints, waypoints.shape[0])
#             traj_list.append(waypoints)

#         self.size = len(traj_list)
#         self.traj_len = num_steps
#         self.wpt_dim = wpt_dim
#         self.traj_array = np.zeros((self.size, self.traj_len, wpt_dim), dtype=np.float32) # N x T x 6
#         # self.traj_array = np.zeros((self.size, self.max_num_waypoints, 6), dtype=np.float32) # N x T x 6
#         for i in range(len(traj_list)):
#             length = traj_list[i].shape[0]
#             self.traj_array[i, :self.traj_len, :] = traj_list[i][:self.traj_len]
#             self.traj_array[i] = traj_list[i]

#             # if 'residual' in dataset_dir:
#             #     self.traj_array[i,1:,:3] *= 1000 # relative position will be very small
#             #     self.traj_array[i,1:,3:] *= (180.0 / np.pi) # relative rotation will be very small

#     def print_data_shape(self):
#         print(f"trajectory : {self.traj_array.shape}")

#     def plot_distribution(self):
#         x = self.traj_array[:, 0, 0]
#         y = self.traj_array[:, 0, 1]
#         z = self.traj_array[:, 0, 2]

#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')

#         ax.scatter(x, y, z, marker='o')
#         ax.set_title('Ecludian Position (3D)')
#         ax.set_xlabel('X label')
#         ax.set_xlabel('Y label')
#         ax.set_xlabel('Z label')
#         plt.show()

#         rx = self.traj_array[:, 0, 3]
#         ry = self.traj_array[:, 0, 4]
#         rz = self.traj_array[:, 0, 5]
        
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='3d')

#         ax.scatter(rx, ry, rz, marker='o')
#         ax.set_title('Ecludian Rotation (3D)')
#         ax.set_xlabel('X label')
#         ax.set_xlabel('Y label')
#         ax.set_xlabel('Z label')
#         plt.show()

#     def __len__(self):
#         return self.size
    
#     def __getitem__(self, index):

#         waypoints = self.traj_array[index]

#         return waypoints

class KptrajReconShapeDataset(Dataset):
    def __init__(self, dataset_dir, num_steps=30, wpt_dim=6, sample_num_points=1000, enable_traj=True, enable_affordance=False):
        
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'

        dataset_subdirs = glob.glob(f'{dataset_dir}/*')
        self.enable_affordance = enable_affordance
        self.enable_traj = enable_traj

        self.type = "residual" if "residual" in dataset_dir else "absolute"
        self.traj_len = num_steps
        self.wpt_dim = wpt_dim
        self.sample_num_points = sample_num_points

        self.shape_list = [] # torch tensor
        self.center_list = []
        self.scale_list = [] 
        self.affordance_list = []
        self.traj_list = []
        for dataset_subdir in dataset_subdirs:

            shape_files = glob.glob(f'{dataset_subdir}/affordance*.npy') # point cloud with affordance score (Nx4), the first element is the contact point
            shape_list_tmp = []
            affordance_list_tmp = []
            center_list_tmp = []
            scale_list_tmp = [] 
            for shape_file in shape_files:
                pcd = np.load(shape_file).astype(np.float32)
                points = pcd[:,:3]

                centroid_points, center, scale = normalize_pc(points, copy_pts=True) # points will be in a unit sphere
                centroid_points = torch.from_numpy(centroid_points).unsqueeze(0).to('cuda').contiguous()
                input_pcid = furthest_point_sample(centroid_points, self.sample_num_points).long().reshape(-1)  # BN
                centroid_points = centroid_points[0, input_pcid, :].squeeze()

                center = torch.from_numpy(center).to('cuda')

                shape_list_tmp.append(centroid_points)
                center_list_tmp.append(center)
                scale_list_tmp.append(scale)

                if enable_affordance == True:
                    affordance = pcd[:,3]
                    affordance = torch.from_numpy(affordance).to('cuda')
                    affordance = affordance[input_pcid]
                    affordance_list_tmp.append(affordance)
            
            self.shape_list.append(shape_list_tmp)
            self.center_list.append(center_list_tmp)
            self.scale_list.append(scale_list_tmp)
            if enable_affordance == True:
                self.affordance_list.append(affordance_list_tmp)

            traj_list_tmp = []
            if enable_traj: 
                traj_files = glob.glob(f'{dataset_subdir}/*.json') # trajectory in 7d format
                for traj_file in traj_files:
                    
                    f_traj = open(traj_file, 'r')
                    traj_dict = json.load(f_traj)
                    waypoints = np.asarray(traj_dict['trajectory'])
                    waypoints = torch.FloatTensor(waypoints).to('cuda')

                    traj_list_tmp.append(waypoints)

                self.traj_list.append(traj_list_tmp)

        if enable_traj and enable_affordance:
            assert len(self.shape_list) == len(self.traj_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance and trajectories'
        elif enable_affordance:
            assert len(self.shape_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance'
        elif enable_traj:
            assert len(self.shape_list) == len(self.traj_list), 'inconsistent length of shapes and trajectories'
        

        self.size = len(self.shape_list)

    def print_data_shape(self):
        print(f'sample_num_points : {self.sample_num_points}')
        print(f"shape : {len(self.shape_list)}")
        if self.enable_affordance:
            print(f"affordance : {len(self.affordance_list)}")
        if self.enable_traj:
            print(f"trajectory : {len(self.traj_list)}")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):

        # for point cloud processing
        num_pcd = len(self.shape_list[index])
        shape_id = np.random.randint(0, num_pcd)
        points = self.shape_list[index][shape_id]
        center = self.center_list[index][shape_id]
        scale = self.scale_list[index][shape_id]

        # for affordance processing if enabled
        affordance = None
        if self.enable_affordance:
            affordance = self.affordance_list[index][shape_id]

        # for waypoint preprocessing
        waypoints = None
        if self.enable_traj:
            num_traj = len(self.traj_list[index])
            traj_id = np.random.randint(0, num_traj)
            wpts = self.traj_list[index][traj_id]
            waypoints = wpts.clone()
            if self.type == "absolute":
                waypoints[:,:3] = (waypoints[:,:3] - center) / scale
            elif self.type == "residual":
                waypoints[1:,:3] = waypoints[1:,:3] / scale
            else :
                print(f"dataset type undefined : {self.type}")
                exit(-1)

        # ret value
        if self.enable_affordance and self.enable_traj:
            return points, affordance, waypoints
        if self.enable_traj:
            return points, waypoints
        if self.enable_affordance:
            return points, affordance
        return points

class KptrajReconAffordanceRot3dDataset(Dataset):
    def __init__(self, dataset_dir, num_steps=30, wpt_dim=6, sample_num_points=1000, enable_traj=True, enable_affordance=False):
        
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'

        dataset_subdirs = glob.glob(f'{dataset_dir}/*')
        self.enable_affordance = enable_affordance
        self.enable_traj = enable_traj

        self.type = "residual" if "residual" in dataset_dir else "absolute"
        self.traj_len = num_steps
        self.wpt_dim = wpt_dim
        self.sample_num_points = sample_num_points

        self.shape_list = [] # torch tensor
        self.center_list = []
        self.scale_list = [] 
        self.affordance_list = []
        self.traj_list = []
        for dataset_subdir in dataset_subdirs:

            shape_files = glob.glob(f'{dataset_subdir}/affordance*.npy') # point cloud with affordance score (Nx4), the first element is the contact point
            shape_list_tmp = []
            affordance_list_tmp = []
            center_list_tmp = []
            scale_list_tmp = [] 
            for shape_file in shape_files:
                pcd = np.load(shape_file).astype(np.float32)
                points = pcd[:,:3]

                centroid_points, center, scale = normalize_pc(points, copy_pts=True) # points will be in a unit sphere
                centroid_points = torch.from_numpy(centroid_points).unsqueeze(0).to('cuda').contiguous()
                input_pcid = furthest_point_sample(centroid_points, self.sample_num_points).long().reshape(-1)  # BN
                centroid_points = centroid_points[0, input_pcid, :].squeeze()

                center = torch.from_numpy(center).to('cuda')

                shape_list_tmp.append(centroid_points)
                center_list_tmp.append(center)
                scale_list_tmp.append(scale)

                if enable_affordance == True:
                    affordance = pcd[:,3]
                    affordance = torch.from_numpy(affordance).to('cuda')
                    affordance = affordance[input_pcid]
                    affordance_list_tmp.append(affordance)
            
            self.shape_list.append(shape_list_tmp)
            self.center_list.append(center_list_tmp)
            self.scale_list.append(scale_list_tmp)
            if enable_affordance == True:
                self.affordance_list.append(affordance_list_tmp)

            traj_list_tmp = []
            if enable_traj: 
                traj_files = glob.glob(f'{dataset_subdir}/*.json') # trajectory in 7d format
                for traj_file in traj_files:
                    
                    f_traj = open(traj_file, 'r')
                    traj_dict = json.load(f_traj)
                    waypoints = np.asarray(traj_dict['trajectory'])

                    if self.type == "residual":
                        waypoints[0, :3] = 0 # clear position of the first waypoint because this position will be given by affordance network

                    waypoints = torch.FloatTensor(waypoints).to('cuda')

                    traj_list_tmp.append(waypoints)

                self.traj_list.append(traj_list_tmp)

        if enable_traj and enable_affordance:
            assert len(self.shape_list) == len(self.traj_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance and trajectories'
        elif enable_affordance:
            assert len(self.shape_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance'
        elif enable_traj:
            assert len(self.shape_list) == len(self.traj_list), 'inconsistent length of shapes and trajectories'
        

        self.size = len(self.shape_list)

    def print_data_shape(self):
        print(f'sample_num_points : {self.sample_num_points}')
        print(f"shape : {len(self.shape_list)}")
        if self.enable_affordance:
            print(f"affordance : {len(self.affordance_list)}")
        if self.enable_traj:
            print(f"trajectory : {len(self.traj_list)}")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):

        # for point cloud processing
        num_pcd = len(self.shape_list[index])
        shape_id = np.random.randint(0, num_pcd)
        points = self.shape_list[index][shape_id]
        center = self.center_list[index][shape_id]
        scale = self.scale_list[index][shape_id]

        # for affordance processing if enabled
        affordance = None
        if self.enable_affordance:
            affordance = self.affordance_list[index][shape_id]

        # for waypoint preprocessing
        waypoints = None
        if self.enable_traj:
            num_traj = len(self.traj_list[index])
            traj_id = np.random.randint(0, num_traj)
            wpts = self.traj_list[index][traj_id]
            waypoints = wpts.clone()
            if self.type == "absolute":
                waypoints[:,:3] = (waypoints[:,:3] - center) / scale
            elif self.type == "residual":
                waypoints[1:,:3] = waypoints[1:,:3] / scale
            else :
                print(f"dataset type undefined : {self.type}")
                exit(-1)

        # ret value
        if self.enable_affordance and self.enable_traj:
            return points, affordance, waypoints
        if self.enable_traj:
            return points, waypoints
        if self.enable_affordance:
            return points, affordance
        return points
    
class KptrajReconAffordanceDataset(Dataset):
    def __init__(self, dataset_dir, num_steps=30, wpt_dim=6, sample_num_points=1000, enable_traj=True, enable_affordance=False):
        
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'

        dataset_subdirs = glob.glob(f'{dataset_dir}/*')
        self.enable_affordance = enable_affordance
        self.enable_traj = enable_traj

        self.type = "residual" if "residual" in dataset_dir else "absolute"
        self.traj_len = num_steps
        self.wpt_dim = wpt_dim
        self.sample_num_points = sample_num_points

        self.shape_list = [] # torch tensor
        self.center_list = []
        self.scale_list = [] 
        self.affordance_list = []
        self.traj_list = []
        for dataset_subdir in dataset_subdirs:

            shape_files = glob.glob(f'{dataset_subdir}/affordance*.npy') # point cloud with affordance score (Nx4), the first element is the contact point
            shape_list_tmp = []
            affordance_list_tmp = []
            center_list_tmp = []
            scale_list_tmp = [] 
            for shape_file in shape_files:
                pcd = np.load(shape_file).astype(np.float32)
                points = pcd[:,:3]

                centroid_points, center, scale = normalize_pc(points, copy_pts=True) # points will be in a unit sphere
                centroid_points = torch.from_numpy(centroid_points).unsqueeze(0).to('cuda').contiguous()
                input_pcid = furthest_point_sample(centroid_points, self.sample_num_points).long().reshape(-1)  # BN
                centroid_points = centroid_points[0, input_pcid, :].squeeze()

                center = torch.from_numpy(center).to('cuda')

                shape_list_tmp.append(centroid_points)
                center_list_tmp.append(center)
                scale_list_tmp.append(scale)

                if enable_affordance == True:
                    affordance = pcd[:,3]
                    affordance = torch.from_numpy(affordance).to('cuda')
                    affordance = affordance[input_pcid]
                    affordance_list_tmp.append(affordance)
            
            self.shape_list.append(shape_list_tmp)
            self.center_list.append(center_list_tmp)
            self.scale_list.append(scale_list_tmp)
            if enable_affordance == True:
                self.affordance_list.append(affordance_list_tmp)

            traj_list_tmp = []
            if enable_traj: 
                traj_files = glob.glob(f'{dataset_subdir}/*.json') # trajectory in 7d format
                for traj_file in traj_files:
                    
                    f_traj = open(traj_file, 'r')
                    traj_dict = json.load(f_traj)
                    waypoints = np.asarray(traj_dict['trajectory'])

                    if self.type == "residual":
                        first_rot_matrix = R.from_rotvec(waypoints[0, 3:]).as_matrix() # omit absolute position of the first waypoint
                        first_rot_matrix_xy = first_rot_matrix.T.reshape(-1)[:6] # the first, second column of the rotation matrix
                        waypoints[0] = first_rot_matrix_xy # rotation only (6d rotation representation)

                    waypoints = torch.FloatTensor(waypoints).to('cuda')

                    traj_list_tmp.append(waypoints)

                self.traj_list.append(traj_list_tmp)

        if enable_traj and enable_affordance:
            assert len(self.shape_list) == len(self.traj_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance and trajectories'
        elif enable_affordance:
            assert len(self.shape_list) == len(self.affordance_list), 'inconsistent length of shapes and affordance'
        elif enable_traj:
            assert len(self.shape_list) == len(self.traj_list), 'inconsistent length of shapes and trajectories'
        

        self.size = len(self.shape_list)

    def print_data_shape(self):
        print(f'sample_num_points : {self.sample_num_points}')
        print(f"shape : {len(self.shape_list)}")
        if self.enable_affordance:
            print(f"affordance : {len(self.affordance_list)}")
        if self.enable_traj:
            print(f"trajectory : {len(self.traj_list)}")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):

        # for point cloud processing
        num_pcd = len(self.shape_list[index])
        shape_id = np.random.randint(0, num_pcd)
        points = self.shape_list[index][shape_id]
        center = self.center_list[index][shape_id]
        scale = self.scale_list[index][shape_id]

        # for affordance processing if enabled
        affordance = None
        if self.enable_affordance:
            affordance = self.affordance_list[index][shape_id]

        # for waypoint preprocessing
        waypoints = None
        if self.enable_traj:
            num_traj = len(self.traj_list[index])
            traj_id = np.random.randint(0, num_traj)
            wpts = self.traj_list[index][traj_id]
            waypoints = wpts.clone()
            if self.type == "absolute":
                waypoints[:,:3] = (waypoints[:,:3] - center) / scale
            elif self.type == "residual":
                waypoints[1:,:3] = waypoints[1:,:3] / scale
            else :
                print(f"dataset type undefined : {self.type}")
                exit(-1)

        # ret value
        if self.enable_affordance and self.enable_traj:
            return points, affordance, waypoints
        if self.enable_traj:
            return points, waypoints
        if self.enable_affordance:
            return points, affordance
        return points

if __name__=="__main__":
    
    # dataset_dir = "../dataset/traj_recon/hook-kptraj_1104_aug-absolute-30/01.16.13.53/train"
    # dataset = KptrajReconDataset(dataset_dir)
    # dataset.print_data_shape()
    # dataset_dir = "../dataset/traj_recon/hook-kptraj_1104_aug-residual-30/01.16.13.55/train"
    # dataset = KptrajReconDataset(dataset_dir)
    # dataset.print_data_shape()
    # dataset_dir = "../dataset/traj_recon/hook-kptraj_1104-absolute-30/01.16.13.41/train"
    # dataset = KptrajReconDataset(dataset_dir)
    # dataset.print_data_shape()
    # dataset_dir = "../dataset/traj_recon/hook-kptraj_1104-residual-30/01.16.13.41/train"
    # dataset = KptrajReconDataset(dataset_dir)
    # dataset.print_data_shape()
    
    # dataset_dir = "../dataset/traj_recon_shape/hook-kptraj_1104_aug-absolute-30/01.16.13.56/train"
    # dataset = KptrajReconShapeDataset(dataset_dir)
    # dataset.print_data_shape()
    # dataset_dir = "../dataset/traj_recon_shape/hook-kptraj_1104_aug-residual-30/01.16.13.57/train"
    # dataset = KptrajReconShapeDataset(dataset_dir)
    # dataset.print_data_shape()
    # dataset_dir = "../dataset/traj_recon_shape/hook-kptraj_1104-absolute-30/01.16.13.41/train"
    # dataset = KptrajReconShapeDataset(dataset_dir)
    # dataset.print_data_shape()
    # dataset_dir = "../dataset/traj_recon_shape/hook-kptraj_1104-residual-30/01.16.13.42/train"
    # dataset = KptrajReconShapeDataset(dataset_dir)
    # dataset.print_data_shape()
    
    dataset_dir = "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-absolute-30/02.03.13.28/train"
    dataset = KptrajReconAffordanceDataset(dataset_dir)
    dataset.print_data_shape()
    # dataset_dir = "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-residual-30/02.03.13.29/train"
    # dataset = KptrajReconAffordanceDataset(dataset_dir)
    # dataset.print_data_shape()
    # dataset_dir = "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-absolute-30/02.11.13.38/train"
    # dataset = KptrajReconAffordanceDataset(dataset_dir)
    # dataset.print_data_shape()
    # dataset_dir = "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-residual-30/02.11.13.39/train"
    # dataset = KptrajReconAffordanceDataset(dataset_dir)
    # dataset.print_data_shape()

