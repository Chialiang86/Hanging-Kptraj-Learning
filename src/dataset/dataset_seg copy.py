import os, glob, json, sys
sys.path.append('../')

import numpy as np
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from utils.training_utils import get_model_module, optimizer_to_device, normalize_pc

class KptrajReconAffordanceDataset(Dataset):
    def __init__(self, dataset_dir, num_steps=30, wpt_dim=6, sample_num_points=1000, enable_traj=True, affordance_type=0, device='cuda', with_noise=False):
        
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
        assert wpt_dim == 9  or wpt_dim == 6 or wpt_dim == 3, f'wpt_dim should be 3 or 6'
        
        self.with_noise = with_noise
        self.noise_pos_scale = 0.0005 # unit: meter
        self.noise_rot_scale = 0.5 * torch.pi / 180 # unit: meter

        self.device = device
        self.affordance_type = affordance_type
        self.affordance_name = {
            2:'partseg',
            4:'fusion',
        }[affordance_type]

        dataset_subdirs = glob.glob(f'{dataset_dir}/*')
        self.enable_traj = enable_traj

        self.type = "residual" if "residual" in dataset_dir else "absolute"
        self.traj_len = num_steps
        self.wpt_dim = wpt_dim
        self.sample_num_points = sample_num_points
        
        self.shape_list = [] # torch tensor
        self.center_list = []
        self.scale_list = [] 
        # self.partseg_list = []
        # self.fusion_list = []
        self.traj_list = []
        for i, dataset_subdir in enumerate(tqdm(dataset_subdirs)):

            shape_files = glob.glob(f'{dataset_subdir}/affordance*.npy') # point cloud with affordance score (Nx4), the first element is the contact point
            shape_list_tmp = []
            # fusion_list_tmp = []
            # partseg_list_tmp = []
            center_list_tmp = []
            scale_list_tmp = [] 
            for shape_file in shape_files:
                pcd = np.load(shape_file).astype(np.float32)
                
                fusion = (pcd[:,3] + pcd[:,4]) / 2 # just average it
                fusion_norm = (fusion - np.min(fusion)) / (np.max(fusion) - np.min(fusion))
                segmented_ind = np.where(fusion_norm > 0.2)[0] # only high response region selected
                segmented_pcd = pcd[segmented_ind]

                points = segmented_pcd[:,:3]
                centroid_points, center, scale = normalize_pc(points, copy_pts=True) # points will be in a unit sphere
                centroid_points = torch.from_numpy(centroid_points).unsqueeze(0).to(device).contiguous()
                
                input_pcid = None
                point_num = centroid_points.shape[1]
                if point_num >= self.sample_num_points:
                    input_pcid = furthest_point_sample(centroid_points, self.sample_num_points).long().reshape(-1)  # BN
                    centroid_points = centroid_points[0, input_pcid, :].squeeze()
                else :
                    mod_num = self.sample_num_points % point_num
                    repeat_num = int(self.sample_num_points // point_num)
                    input_pcid = furthest_point_sample(centroid_points, mod_num).long().reshape(-1)  # BN
                    centroid_points = torch.cat([centroid_points.repeat(1, repeat_num, 1), centroid_points[:, input_pcid]], dim=1).squeeze()
                    input_pcid = torch.cat([torch.arange(0, point_num).int().repeat(repeat_num).to(self.device), input_pcid])
                
                center = torch.from_numpy(center).to(device)

                shape_list_tmp.append(centroid_points)
                center_list_tmp.append(center)
                scale_list_tmp.append(scale)

                # if self.affordance_name == 'partseg':
                #     partseg = segmented_pcd[:,4]
                #     partseg = torch.from_numpy(partseg).to(device)
                #     partseg = partseg[input_pcid]
                #     partseg_list_tmp.append(partseg)

                # if self.affordance_name == 'fusion':
                #     partseg = segmented_pcd[:,4]
                #     fusion = (segmented_pcd[:,3] + segmented_pcd[:,4]) / 2 # just average it
                #     fusion = torch.from_numpy(fusion).to(device)
                #     fusion = fusion[input_pcid]
                #     fusion_list_tmp.append(fusion)
            
            self.shape_list.append(shape_list_tmp)
            self.center_list.append(center_list_tmp)
            self.scale_list.append(scale_list_tmp)
            # if self.affordance_name == 'partseg':
            #     self.partseg_list.append(partseg_list_tmp)
            # if self.affordance_name == 'fusion':
            #     self.fusion_list.append(fusion_list_tmp)

            traj_list_tmp = []
            if enable_traj: 
                traj_files = glob.glob(f'{dataset_subdir}/*.json')[:1] # trajectory in 7d format
                    
                for traj_file in traj_files:
                    
                    f_traj = open(traj_file, 'r')
                    traj_dict = json.load(f_traj)

                    waypoints = np.asarray(traj_dict['trajectory'])
                    if self.type == "residual":
                        
                        if self.wpt_dim == 6:
                            first_rot_matrix = R.from_rotvec(waypoints[0, 3:]).as_matrix() # omit absolute position of the first waypoint
                            first_rot_matrix_xy = (first_rot_matrix.T).reshape(-1)[:6] # the first, second column of the rotation matrix
                            waypoints[0] = first_rot_matrix_xy # rotation only (6d rotation representation)

                    if self.wpt_dim == 9:
                        waypoints_9d = np.zeros((waypoints.shape[0], 9))
                        waypoints_9d[:, :3] = waypoints[:, :3]
                        rot_matrix = R.from_rotvec(waypoints[:, 3:]).as_matrix() # omit absolute position of the first waypoint
                        rot_matrix_xy = np.transpose(rot_matrix, (0, 2, 1)).reshape((waypoints.shape[0], -1))[:, :6] # the first, second column of the rotation matrix
                        waypoints_9d[:, 3:] = rot_matrix_xy # rotation only (6d rotation representation)
                        waypoints = waypoints_9d

                    waypoints = torch.FloatTensor(waypoints).to(device)
                    traj_list_tmp.append(waypoints)

                self.traj_list.append(traj_list_tmp)

        if enable_traj and self.affordance_name == "partseg":
            assert len(self.shape_list) == len(self.traj_list) == len(self.partseg_list), 'inconsistent length of shapes and partseg and trajectories'
        elif enable_traj and self.affordance_name == "fusion":
            assert len(self.shape_list) == len(self.traj_list) == len(self.fusion_list), 'inconsistent length of shapes and affordance and trajectories'
        elif self.affordance_name == 'partseg':
            assert len(self.shape_list) == len(self.partseg_list), 'inconsistent length of shapes and partseg'
        elif self.affordance_name == "fusion":
            assert len(self.shape_list) == len(self.fusion_list), 'inconsistent length of shapes and affordance'
        elif enable_traj: # 100
            assert len(self.shape_list) == len(self.traj_list), 'inconsistent length of shapes and trajectories'
        
        self.size = len(self.shape_list)

    def print_data_shape(self):
        print(f'dataset size : {self.size}')
        print(f'sample_num_points : {self.sample_num_points}')
        if self.enable_traj:
            print(f"trajectory : {len(self.traj_list)}")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):

        # for point cloud processing
        num_pcd = len(self.shape_list[index])
        shape_id = np.random.randint(0, num_pcd)
        points = self.shape_list[index][shape_id].clone()
        center = self.center_list[index][shape_id].clone()
        scale = self.scale_list[index][shape_id]

        # noise to point cloud
        if self.with_noise:
            points_cp = points[0].clone()
            point_noises = torch.randn(points.shape).to(self.device) * self.noise_pos_scale / scale
            points += point_noises
            points[0] = points_cp

        # # for partseg processing if enabled
        # partseg = None
        # if self.affordance_name == 'partseg':
        #     partseg = self.partseg_list[index][shape_id]

        # # for fusion processing if enabled
        # fusion = None
        # if self.affordance_name == 'fusion':
        #     fusion = self.fusion_list[index][shape_id]

        # for waypoint preprocessing
        waypoints = None
        if self.enable_traj:
            num_traj = len(self.traj_list[index])
            traj_id = np.random.randint(0, num_traj)
            wpts = self.traj_list[index][traj_id]
            waypoints = wpts.clone()
            
            # noise to waypoints
            if self.with_noise:
                pos_noises = torch.randn(waypoints[:, :3].shape).to(self.device) * self.noise_pos_scale * 0.1
                waypoints[:, :3] += pos_noises
                rot_noises = torch.randn(waypoints[:, 3:].shape).to(self.device) * self.noise_rot_scale * 0.1
                waypoints[:, 3:] += rot_noises

            if self.type == "absolute":
                waypoints[:, :3] = (waypoints[:, :3] - center) / scale
            elif self.type == "residual":
                waypoints[1:, :3] = waypoints[1:, :3] / scale
            else :
                print(f"dataset type undefined : {self.type}")
                exit(-1)

        # ret value
        return points, waypoints[:,:self.wpt_dim]
        # if self.enable_traj and self.affordance_name == 'partseg':
        #     return points, partseg, waypoints[:,:self.wpt_dim]
        # elif self.enable_traj and self.affordance_name == 'fusion':
        #     return points, fusion, waypoints[:,:self.wpt_dim]
        # elif self.enable_traj:
        #     return points, waypoints[:,:self.wpt_dim]
        # elif self.affordance_name == 'partseg':
        #     return points, partseg
        # elif self.affordance_name == 'fusion':
        #     return points, fusion
        # return points

if __name__=="__main__":
    
    dataset_dir = "../../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
    dataset = KptrajReconAffordanceDataset(dataset_dir)
    dataset.print_data_shape()
