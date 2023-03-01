import argparse, json, yaml, os, time, glob
import open3d as o3d
import numpy as np
from datetime import datetime

from tqdm import tqdm

from pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.training_utils import get_model_module, get_dataset_module, optimizer_to_device, normalize_pc

import torch
from torch.utils.data import DataLoader, Subset

import pybullet as p
from PIL import Image
from scipy.spatial.transform import Rotation as R
from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose, \
                               pose_6d_to_7d, pose_7d_to_6d, draw_coordinate
from utils.testing_utils import trajectory_scoring, refine_waypoint_rotation


def recover_trajectory(traj : torch.Tensor or np.ndarray, hook_pose : list or np.ndarray, 
                        center : torch.Tensor or np.ndarray, scale : float, dataset_mode : int=0):
    # traj : dim = batch x num_steps x 6
    # dataset_mode : 0 for abosute, 1 for residual 

    if type(traj) == torch.Tensor:
        traj = traj.cpu().detach().numpy()
    if type(center) == torch.Tensor:
        center = center.cpu().detach().numpy()

    base_trans = get_matrix_from_pose(hook_pose)

    waypoints = []

    if dataset_mode == 0: # "absolute"
        for i in range(traj.shape[0]): # batches
            waypoints.append([])
            for j in range(1, traj[i].shape[0]): # waypoints
                traj[i, j, :3] = traj[i, j, :3] * scale + center
                current_trans = base_trans @ get_matrix_from_pose(traj[i, j])
                waypoints[-1].append(get_pose_from_matrix(current_trans, pose_size=6))
    
    if dataset_mode == 1: # "residual"
        for i in range(traj.shape[0]):
            traj[i, 0, :3] = traj[i, 0, :3] * scale + center
            current_trans = base_trans @ get_matrix_from_pose(traj[i, 0])
            current_pose = get_pose_from_matrix(current_trans, pose_size=6)
            waypoints.append([current_pose])
            # waypoints.append([get_pose_from_matrix(current_trans, pose_size=6)])
            for j in range(1, traj[i].shape[0]):
                tmp_pose = np.zeros(6)
                traj[i, j, :3] *= scale
                tmp_pose[:3] = current_pose[:3] + traj[i, j, :3]
                tmp_pose[3:] = R.from_matrix(
                                        R.from_rotvec(
                                            traj[i, j, 3:]
                                        ).as_matrix() @ R.from_rotvec(
                                            current_pose[3:]
                                        ).as_matrix()
                                    ).as_rotvec()
                current_pose = tmp_pose

                waypoints[-1].append(current_pose)
    
    return waypoints

def main(args):
    
    inference_dir = args.inference_dir
    assert os.path.exists(inference_dir), f'{inference_dir} not exists'

    affordance_weight_path = f'{args.affordance_checkpoint_dir}/{args.affordance_weight_subpath}'
    trajectory_weight_path = f'{args.trajectory_checkpoint_dir}/{args.trajectory_weight_subpath}'
    assert os.path.exists(affordance_weight_path), f'{affordance_weight_path} not exists'
    assert os.path.exists(trajectory_weight_path), f'{trajectory_weight_path} not exists'

    affordance_config_file = args.affordance_config 
    trajectory_config_file = args.trajectory_config 
    assert os.path.exists(affordance_config_file), f'{affordance_config_file} not exists'
    assert os.path.exists(trajectory_config_file), f'{trajectory_config_file} not exists'

    # ================== Config ==================

    device = args.device
    visualize = args.visualize
    dataset_mode = 0 if 'absolute' in trajectory_weight_path else 1 # 0: absolute, 1: residual

    # ================== Model ==================

    affordance_config = None
    with open(affordance_config_file, 'r') as f:
        affordance_config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for network
    affordance_module_name = affordance_config['module']
    affordance_model_name = affordance_config['model']
    affordance_model_inputs = affordance_config['inputs']

    # load model
    affordance_network_class = get_model_module(affordance_module_name, affordance_model_name)
    affordance_network = affordance_network_class({'model.use_xyz': affordance_model_inputs['model.use_xyz']}).to(device)
    affordance_network.load_state_dict(torch.load(affordance_weight_path))

    trajectory_config = None
    with open(trajectory_config_file, 'r') as f:
        trajectory_config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for network
    trajectory_module_name = trajectory_config['module']
    trajectory_model_name = trajectory_config['model']
    trajectory_model_inputs = trajectory_config['inputs']

    # load model
    trajectory_network_class = get_model_module(trajectory_module_name, trajectory_model_name)
    trajectory_network = trajectory_network_class(**trajectory_model_inputs, dataset_type=dataset_mode).to(device)
    trajectory_network.load_state_dict(torch.load(trajectory_weight_path))
    
    # ================== Inference ==================

    # inference
    inference_obj_dir = args.obj_shape_root
    assert os.path.exists(inference_obj_dir), f'{inference_obj_dir} not exists'
    inference_obj_whole_dirs = glob.glob(f'{inference_obj_dir}/*')

    inference_hook_shape_root = args.hook_shape_root
    assert os.path.exists(inference_hook_shape_root), f'{inference_hook_shape_root} not exists'

    inference_hook_dir = args.inference_dir # for hook shapes
    inference_hook_whole_dirs = glob.glob(f'{inference_hook_dir}/*')

    inference_obj_paths = []
    inference_hook_paths = []

    for inference_obj_path in inference_obj_whole_dirs:
        paths = glob.glob(f'{inference_obj_path}/*.json')
        assert len(paths) == 1, f'multiple object contact informations : {paths}'
        inference_obj_paths.extend(paths) 

    for inference_hook_path in inference_hook_whole_dirs:
        # if 'Hook' in inference_hook_path:
        paths = glob.glob(f'{inference_hook_path}/affordance-0.npy')
        inference_hook_paths.extend(paths) 

    hook_pcds = []
    hook_urdfs = []
    for inference_hook_path in inference_hook_paths:
        # pcd = o3d.io.read_point_cloud(inference_shape_path)
        # points = np.asarray(pcd.points, dtype=np.float32)
        hook_name = inference_hook_path.split('/')[-2]
        urdf_prefix = os.path.split(inference_hook_path)[0]
        points = np.load(inference_hook_path)[:, :3].astype(np.float32)
        affords = np.load(inference_hook_path)[:, 3].astype(np.float32)

        hook_urdf = f'{inference_hook_shape_root}/{hook_name}/base.urdf'
        assert os.path.exists(hook_urdf), f'{hook_urdf} not exists'
        hook_urdfs.append(hook_urdf) 
        hook_pcds.append(points)

    obj_contact_poses = []
    obj_urdfs = []
    for inference_obj_path in inference_obj_paths:
        obj_contact_info = json.load(open(inference_obj_path, 'r'))
        obj_contact_poses.append(obj_contact_info['contact_pose'])

        obj_urdf = '{}/base.urdf'.format(os.path.split(inference_obj_path)[0])
        assert os.path.exists(obj_urdf), f'{obj_urdf} not exists'
        obj_urdfs.append(obj_urdf)

    # ================== Simulator ==================

    # Create pybullet GUI
    # physicsClientId = p.connect(p.DIRECT)
    physicsClientId = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.1,
        cameraYaw=90,
        cameraPitch=-10,
        cameraTargetPosition=[0.0, 0.0, 0.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    sim_timestep = 1.0 / 240
    p.setTimeStep(sim_timestep)
    p.setGravity(0, 0, 0)

    hook_pose = [
        0.0,
        0.0,
        0.0,
        4.329780281177466e-17,
        0.7071067811865475,
        0.7071067811865476,
        4.329780281177467e-17
    ]
    
    # ================== Inference ==================

    sample_num_points = 1000
    batch_size = 1
    for sid, pcd in enumerate(hook_pcds):
        
        # urdf file
        hook_urdf = hook_urdfs[sid]
        hook_id = p.loadURDF(hook_urdf, hook_pose[:3], hook_pose[3:])

        # sample trajectories
        centroid_pcd, centroid, scale = normalize_pc(pcd, copy_pts=True) # points will be in a unit sphere
        contact_point = centroid_pcd[0]

        points_batch = torch.from_numpy(centroid_pcd).unsqueeze(0).to(device=device).contiguous()
        input_pcid = furthest_point_sample(points_batch, sample_num_points).long().reshape(-1)  # BN
        points_batch = points_batch[0, input_pcid, :].squeeze()
        points_batch = points_batch.repeat(batch_size, 1, 1)

        # affordance network
        affords = affordance_network.inference(points_batch)
        affords = (affords - torch.min(affords)) / (torch.max(affords) - torch.min(affords))
        affords = affords.squeeze().cpu().detach().numpy()

        contact_point_cond = np.where(affords == np.max(affords))[0]
        contact_point_inference = points[contact_point_cond][0]

        print(contact_point_inference)

        # contact_point_batch = torch.from_numpy(contact_point).to(device=device).repeat(batch_size, 1)

        # # trajectory network
        # recon_traj = trajectory_network.sample(points_batch, contact_point_batch)
        # recon_traj = recon_traj.cpu().detach().numpy()

        # recovered_trajs = recover_trajectory(recon_traj, hook_pose, centroid, scale, dataset_mode)

        # # conting inference score using object and object contact information
        # for recovered_traj in recovered_trajs:

        #     for i in range(len(obj_contact_poses)):

        #         obj_id = p.loadURDF(obj_urdfs[i])
        #         reversed_recovered_traj = recovered_traj[::-1][:-10]
        #         reversed_recovered_traj = refine_waypoint_rotation(reversed_recovered_traj)
        #         score = trajectory_scoring(reversed_recovered_traj, hook_id, obj_id, [0, 0, 0, 0, 0, 0, 1], obj_contact_pose=obj_contact_poses[i])
        #         print(f'{obj_urdfs[i]}, score : {score}')
        #         p.removeBody(obj_id)

        # if visualize:
        #     wpt_ids = []
        #     for i, recovered_traj in enumerate(recovered_trajs):
        #         colors = list(np.random.rand(3)) + [1]
        #         for wpt_i, wpt in enumerate(recovered_traj):
        #             wpt_id = p.createMultiBody(
        #                 baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, 0.001), 
        #                 baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, 0.001, rgbaColor=colors), 
        #                 basePosition=wpt[:3]
        #             )
        #             wpt_ids.append(wpt_id)

        #     # capture a list of images and save as gif
        #     delta = 10
        #     delta_sum = 0
        #     cameraYaw = 90
        #     rgbs = []
        #     while True:
        #         keys = p.getKeyboardEvents()
        #         p.resetDebugVisualizerCamera(
        #             cameraDistance=0.08,
        #             cameraYaw=cameraYaw,
        #             cameraPitch=-10,
        #             cameraTargetPosition=[0.0, 0.0, 0.0]
        #         )

        #         cam_info = p.getDebugVisualizerCamera()
        #         width = cam_info[0]
        #         height = cam_info[1]
        #         view_mat = cam_info[2]
        #         proj_mat = cam_info[3]
        #         img_info = p.getCameraImage(width, height, viewMatrix=view_mat, projectionMatrix=proj_mat)
        #         rgb = img_info[2]
        #         rgbs.append(Image.fromarray(rgb))

        #         cameraYaw += delta 
        #         delta_sum += delta 
        #         cameraYaw = cameraYaw % 360
        #         if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
        #             break
        #         if delta_sum >= 360:
        #             break

        #     for wpt_id in wpt_ids:
        #         p.removeBody(wpt_id)

        p.removeBody(hook_id)


if __name__=="__main__":
    parser = argparse.ArgumentParser() # about dataset

    # weight
    parser.add_argument('--inference_dir', '-id', type=str, default='../shapes/hook_all')
    parser.add_argument('--obj_shape_root', '-osr', type=str, default='../shapes/inference_objs')
    parser.add_argument('--hook_shape_root', '-hsr', type=str, default='../shapes/hook_all_new_0')
    parser.add_argument('--affordance_checkpoint_dir', '-acd', type=str, default='checkpoints', help="'training_mode=test' only")
    parser.add_argument('--affordance_weight_subpath', '-awp', type=str, default='', help="subpath of saved weight")
    parser.add_argument('--trajectory_checkpoint_dir', '-tcd', type=str, default='', help="subpath of saved weight")
    parser.add_argument('--trajectory_weight_subpath', '-twp', type=str, default='', help="subpath of saved weight")
    parser.add_argument('--visualize', '-v', action='store_true')
    
    # other info
    parser.add_argument('--device', '-dv', type=str, default="cuda")
    parser.add_argument('--affordance_config', '-acfg', type=str, default='../config/affordance.yaml')
    parser.add_argument('--trajectory_config', '-tcfg', type=str, default='../config/affordance.yaml')
    parser.add_argument('--verbose', '-vb', action='store_true')
    args = parser.parse_args()

    main(args)