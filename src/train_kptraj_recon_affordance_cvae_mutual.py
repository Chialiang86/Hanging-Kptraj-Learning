import argparse, sys, json, yaml, cv2, imageio, os, time, glob
import open3d as o3d
import numpy as np

from datetime import datetime

from tqdm import tqdm
from time import strftime

from sklearn.model_selection import train_test_split
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.training_utils import get_model_module, get_dataset_module, optimizer_to_device, normalize_pc, kl_annealing

import torch
from torch.utils.data import DataLoader, Subset
from torchsummary import summary

from scipy.spatial.transform import Rotation as R
from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose, \
                               pose_6d_to_7d, pose_7d_to_6d, draw_coordinate
from utils.testing_utils import trajectory_scoring, refine_waypoint_rotation, robot_kptraj_hanging

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    return train_set, val_set

def train(args):

    time_stamp = datetime.today().strftime('%m.%d.%H.%M')
    training_tag = time_stamp if args.training_tag == '' else f'{args.training_tag}'
    dataset_dir = args.dataset_dir
    dataset_root = args.dataset_dir.split('/')[-2] # dataset category
    dataset_subroot = args.dataset_dir.split('/')[-1] # time stamp
    config_file = args.config
    verbose = args.verbose
    device = args.device
    dataset_mode = 0 if 'absolute' in dataset_dir else 1 # 0: absolute, 1: residual

    config_file_id = config_file.split('/')[-1][:-5] # remove '.yaml'
    checkpoint_dir = f'{args.checkpoint_dir}/{config_file_id}-{training_tag}/{dataset_root}-{dataset_subroot}'
    print(f'checkpoint_dir: {checkpoint_dir}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for training
    dataset_name = config['dataset_module']
    dataset_class_name = config['dataset_class']
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['model_inputs']
    dataset_inputs = config['dataset_inputs']
    train_traj_start = config['model_inputs']['train_traj_start']
    
    # training batch and iters
    batch_size = config['batch_size']
    start_epoch = config['start_epoch']
    stop_epoch = config['stop_epoch']

    # training scheduling params
    lr = config['lr']
    lr_decay_rate = config['lr_decay_rate']
    lr_decay_epoch = config['lr_decay_epoch']
    weight_decay = config['weight_decay']
    save_freq = config['save_freq']

    # TODO: checkpoint dir to load and train
    # TODO: config file should add a new key : point number
    
    dataset_class = get_dataset_module(dataset_name, dataset_class_name)
    train_set = dataset_class(dataset_dir=f'{dataset_dir}/train', **dataset_inputs)
    val_set = dataset_class(dataset_dir=f'{dataset_dir}/val', **dataset_inputs)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    sample_num_points = train_set.sample_num_points
    print(f'dataset: {dataset_dir}')
    print(f'checkpoint_dir: {checkpoint_dir}')
    print(f'num of points in point cloud: {sample_num_points}')

    network_class = get_model_module(module_name, model_name)
    network = network_class(**model_inputs, dataset_type=dataset_mode).to(device)
    
    if verbose:
        summary(network)

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=lr_decay_epoch, gamma=lr_decay_rate)
    optimizer_to_device(network_opt, device)

    if verbose:
        print(f'training batches: {len(train_loader)}')
        print(f'validation batches: {len(val_loader)}')

    # start training
    start_time = time.time()

    # will only work if 'kl_annealing' = 1 in "model_inputs"
    kl_weight = kl_annealing(kl_anneal_cyclical=True, 
                niter=stop_epoch, 
                start=0.0, stop=0.1, kl_anneal_cycle=5, kl_anneal_ratio=0.5)

    # train for every epoch
    for epoch in range(start_epoch, stop_epoch + 1):

        train_batches = enumerate(train_loader, 0)
        val_batches = enumerate(val_loader, 0)

        # training
        train_dist_losses = []
        train_afford_losses = []
        train_nn_losses = []
        train_dir_losses = []
        train_kl_losses = []
        train_recon_losses = []
        train_total_losses = []
        for i_batch, (sample_pcds, sample_affords, sample_trajs) in tqdm(train_batches, total=len(train_loader)):

            # set models to training mode
            network.train()

            sample_pcds = sample_pcds.to(device).contiguous() 
            sample_trajs = sample_trajs.to(device).contiguous()
            sample_cp = sample_pcds[:, 0]

            # forward pass
            losses = network.get_loss(epoch, sample_pcds, sample_trajs, sample_cp, sample_affords, lbd_kl=kl_weight.get_beta())  # B x 2, B x F x N
            total_loss = losses['total']

            if 'afford' in losses.keys():
                train_afford_losses.append(losses['afford'].item())
            if 'dist' in losses.keys():
                train_dist_losses.append(losses['dist'].item())
            if 'nn' in losses.keys():
                train_nn_losses.append(losses['nn'].item())
            if 'dir' in losses.keys():
                train_dir_losses.append(losses['dir'].item())
            train_kl_losses.append(losses['kl'].item())
            train_recon_losses.append(losses['recon'].item())
            train_total_losses.append(losses['total'].item())

            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()

        network_lr_scheduler.step()
        
        if 'afford' in losses.keys():
            train_afford_avg_loss = np.mean(np.asarray(train_afford_losses))
        if 'dist' in losses.keys():
            train_dist_avg_loss = np.mean(np.asarray(train_dist_losses))
        if 'nn' in losses.keys():
            train_nn_avg_loss = np.mean(np.asarray(train_nn_losses))
        if 'dir' in losses.keys():
            train_dir_avg_loss = np.mean(np.asarray(train_dir_losses))
        train_kl_avg_loss = np.mean(np.asarray(train_kl_losses))
        train_recon_avg_loss = np.mean(np.asarray(train_recon_losses))
        train_total_avg_loss = np.mean(np.asarray(train_total_losses))
        print(
                f'''---------------------------------------------\n'''
                f'''[ training stage ]\n'''
                f''' - time : {strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} \n'''
                f''' - epoch : {epoch:>5.0f}/{stop_epoch:<5.0f} \n'''
                f''' - lr : {network_opt.param_groups[0]['lr']:>5.2E} \n'''
                f''' - train_afford_avg_loss : {train_afford_avg_loss if 'afford' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - train_dist_avg_loss : {train_dist_avg_loss if 'dist' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - train_nn_avg_loss : {train_nn_avg_loss if 'nn' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - train_dir_avg_loss : {train_dir_avg_loss if 'dir' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - train_kl_avg_loss : {train_kl_avg_loss:>10.5f}\n'''
                f''' - train_recon_avg_loss : {train_recon_avg_loss:>10.5f}\n'''
                f''' - train_total_avg_loss : {train_total_avg_loss:>10.5f}\n'''
                f'''---------------------------------------------\n'''
            )
        
        # save checkpoint
        if (epoch - start_epoch) % save_freq == 0 and (epoch - start_epoch) > 0 and epoch > train_traj_start:
            with torch.no_grad():
                print('Saving checkpoint ...... ')
                # torch.save(network, os.path.join(checkpoint_dir, f'{sample_num_points}_points-network_epoch-{epoch}.pth'))
                torch.save(network.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-network_epoch-{epoch}.pth'))
                # torch.save(network_opt.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-optimizer_epoch-{epoch}.pth'))
                # torch.save(network_lr_scheduler.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-scheduler_epoch-{epoch}.pth'))

        # validation
        val_dist_losses = []
        val_afford_losses = []
        val_nn_losses = []
        val_dir_losses = []
        val_kl_losses = []
        val_recon_losses = []
        val_total_losses = []
        # total_loss, total_precision, total_recall, total_Fscore, total_accu = 0, 0, 0, 0, 0
        for i_batch, (sample_pcds, sample_affords, sample_trajs) in tqdm(val_batches, total=len(val_loader)):

            # set models to evaluation mode
            network.eval()

            sample_pcds = sample_pcds.to(device).contiguous() 
            sample_trajs = sample_trajs.to(device).contiguous()
            sample_cp = sample_pcds[:, 0]

            with torch.no_grad():
                losses = network.get_loss(epoch, sample_pcds, sample_trajs, sample_cp, sample_affords, lbd_kl=kl_weight.get_beta())  # B x 2, B x F x N
                
                if 'afford' in losses.keys():
                    val_afford_losses.append(losses['afford'].item())
                if 'dist' in losses.keys():
                    val_dist_losses.append(losses['dist'].item())
                if 'nn' in losses.keys():
                    val_nn_losses.append(losses['nn'].item())
                if 'dir' in losses.keys():
                    val_dir_losses.append(losses['dir'].item())
                val_kl_losses.append(losses['kl'].item())
                val_recon_losses.append(losses['recon'].item())
                val_total_losses.append(losses['total'].item())

        if 'afford' in losses.keys():
            val_afford_avg_loss = np.mean(np.asarray(val_afford_losses))
        if 'dist' in losses.keys():
            val_dist_avg_loss = np.mean(np.asarray(val_dist_losses))
        if 'nn' in losses.keys():
            val_nn_avg_loss = np.mean(np.asarray(val_nn_losses))
        if 'dir' in losses.keys():
            val_dir_avg_loss = np.mean(np.asarray(val_dir_losses))
        val_kl_avg_loss = np.mean(np.asarray(val_kl_losses))
        val_recon_avg_loss = np.mean(np.asarray(val_recon_losses))
        val_total_avg_loss = np.mean(np.asarray(val_total_losses))
        print(
                f'''---------------------------------------------\n'''
                f'''[ validation stage ]\n'''
                f''' - time : {strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} \n'''
                f''' - epoch : {epoch:>5.0f}/{stop_epoch:<5.0f} \n'''
                f''' - lr : {network_opt.param_groups[0]['lr']:>5.2E} \n'''
                f''' - val_afford_avg_loss : {val_afford_avg_loss if 'afford' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - val_dist_avg_loss : {val_dist_avg_loss if 'dist' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - val_nn_avg_loss : {val_nn_avg_loss if 'nn' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - val_dir_avg_loss : {val_dir_avg_loss if 'dir' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - val_kl_avg_loss : {val_kl_avg_loss:>10.5f}\n'''
                f''' - val_recon_avg_loss : {val_recon_avg_loss:>10.5f}\n'''
                f''' - val_total_avg_loss : {val_total_avg_loss:>10.5f}\n'''
                f'''---------------------------------------------\n'''
            )

        kl_weight.update()

def capture_from_viewer(geometries):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    for geometry in geometries:
        vis.add_geometry(geometry)

    # Updates
    for geometry in geometries:
        vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()

    o3d_screenshot_mat = vis.capture_screen_float_buffer(do_render=True) # need to be true to capture the image
    o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
    o3d_screenshot_mat = cv2.cvtColor(o3d_screenshot_mat,cv2.COLOR_BGR2RGB)
    o3d_screenshot_mat = cv2.resize(o3d_screenshot_mat, (o3d_screenshot_mat.shape[1] // 6, o3d_screenshot_mat.shape[0] // 6))
    vis.destroy_window()

    return o3d_screenshot_mat

def recover_trajectory(traj_src : torch.Tensor or np.ndarray, hook_poses : torch.Tensor or np.ndarray, 
                        centers : torch.Tensor or np.ndarray, scales : torch.Tensor or np.ndarray, dataset_mode : int=0, wpt_dim : int=6):
    # traj : dim = batch x num_steps x 6
    # dataset_mode : 0 for abosute, 1 for residual 

    traj = None
    if type(traj_src) == torch.Tensor:
        traj = traj_src.clone().cpu().detach().numpy()
        centers = centers.clone().cpu().detach().numpy()
        scales = scales.clone().cpu().detach().numpy()
        hook_poses = hook_poses.clone().cpu().detach().numpy()
    elif type(traj_src) == np.ndarray:
        traj = np.copy(traj_src)

    # base_trans = get_matrix_from_pose(hook_pose)

    waypoints = []

    if dataset_mode == 0: # "absolute"

        for traj_id in range(traj.shape[0]):
            traj[traj_id, :, :3] = traj[traj_id, :, :3] * scales[traj_id] + centers[traj_id]

        for traj_id in range(traj.shape[0]): # batches
            waypoints.append([])
            for wpt_id in range(0, traj[traj_id].shape[0]): # waypoints

                wpt = np.zeros(6)
                if wpt_dim == 6:
                    wpt = traj[traj_id, wpt_id]
                    current_trans = get_matrix_from_pose(hook_poses[traj_id]) @ get_matrix_from_pose(wpt)
                else :
                    # contact pose rotation
                    wpt[:3] = traj[traj_id, wpt_id]

                    # transform to world coordinate first
                    current_trans = np.identity(4)
                    current_trans[:3, 3] = traj[traj_id, wpt_id]
                    current_trans = get_matrix_from_pose(hook_poses[traj_id]) @ current_trans

                    if wpt_id < traj[traj_id].shape[0] - 1:
                        # transform to world coordinate first
                        next_trans = np.identity(4)
                        next_trans[:3, 3] = traj[traj_id, wpt_id+1]
                        next_trans =  get_matrix_from_pose(hook_poses[traj_id]) @ next_trans

                        x_direction = np.asarray(next_trans[:3, 3]) - np.asarray(current_trans[:3, 3])
                        x_direction /= np.linalg.norm(-x_direction, ord=2)
                        y_direction = np.cross(x_direction, [0, 0, -1])
                        y_direction /= np.linalg.norm(y_direction, ord=2)
                        z_direction = np.cross(x_direction, y_direction)
                        rotation_mat = np.vstack((x_direction, y_direction, z_direction)).T
                        current_trans[:3, :3] = rotation_mat
                        
                    else :
                        current_trans[:3, :3] = R.from_rotvec(waypoints[-1][-1][3:]).as_matrix() # use the last waypoint's rotation as current rotation
                
                waypoints[-1].append(get_pose_from_matrix(current_trans, pose_size=6))
    
    if dataset_mode == 1: # "residual"

        for traj_id in range(traj.shape[0]):
            traj[traj_id, 0, :3] = (traj[traj_id, 0, :3] * scales[traj_id]) + centers[traj_id]
            traj[traj_id, 1:, :3] = (traj[traj_id, 1:, :3] * scales[traj_id])

        for traj_id in range(traj.shape[0]):
            waypoints.append([])
            tmp_pos = np.array([0.0, 0.0, 0.0])
            tmp_rot = np.array([0.0, 0.0, 0.0])
            for wpt_id in range(0, traj[traj_id].shape[0]):
                
                wpt = np.zeros(6)
                if wpt_dim == 6:

                    if wpt_id == 0 :
                        wpt = traj[traj_id, wpt_id]
                        tmp_pos = wpt[:3]
                        tmp_rot = wpt[3:]
                    else :
                        tmp_pos = tmp_pos + np.asarray(traj[traj_id, wpt_id, :3])
                        tmp_rot = R.from_matrix(
                                            R.from_rotvec(
                                                traj[traj_id, wpt_id, 3:]
                                            ).as_matrix() @ R.from_rotvec(
                                                tmp_rot
                                            ).as_matrix()
                                        ).as_rotvec()
                        wpt[:3] = tmp_pos
                        wpt[3:] = tmp_rot
                        
                    current_trans = get_matrix_from_pose(hook_poses[traj_id]) @ get_matrix_from_pose(wpt)
                
                else :
                    
                    # transform to world coordinate first
                    current_trans = np.identity(4)
                    current_trans[:3, 3] = tmp_pos + traj[traj_id, wpt_id]
                    tmp_pos += traj[traj_id, wpt_id]
                    current_trans = get_matrix_from_pose(hook_poses[traj_id]) @ current_trans

                    if wpt_id < traj[traj_id].shape[0] - 1:
                        # transform to world coordinate first
                        next_trans = np.identity(4)
                        next_trans[:3, 3] = traj[traj_id, wpt_id+1]
                        next_trans = get_matrix_from_pose(hook_poses[traj_id]) @ next_trans

                        x_direction = np.asarray(next_trans[:3, 3]) - np.asarray(current_trans[:3, 3])
                        x_direction /= np.linalg.norm(-x_direction, ord=2)
                        y_direction = np.cross(x_direction, [0, 0, -1])
                        y_direction /= np.linalg.norm(y_direction, ord=2)
                        z_direction = np.cross(x_direction, y_direction)
                        rotation_mat = np.vstack((x_direction, y_direction, z_direction)).T
                        current_trans[:3, :3] = rotation_mat
                    else :
                        current_trans[:3, :3] = R.from_rotvec(waypoints[-1][-1][3:]).as_matrix() # use the last waypoint's rotation as current rotation

                waypoints[-1].append(get_pose_from_matrix(current_trans, pose_size=6))
    
    return waypoints


def val(args):

    import pybullet as p
    import pybullet_data
    from pybullet_robot_envs.envs.panda_envs.panda_env import pandaEnv
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    # ================== config ==================

    checkpoint_dir = f'{args.checkpoint_dir}'
    config_file = args.config
    device = args.device
    dataset_mode = 0 if 'absolute' in checkpoint_dir else 1 # 0: absolute, 1: residual
    weight_subpath = args.weight_subpath
    weight_path = f'{checkpoint_dir}/{weight_subpath}'

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # sample_num_points = int(weight_subpath.split('_')[0])

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for network
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['model_inputs']
    dataset_inputs = config['dataset_inputs']
    batch_size = config['batch_size']

    wpt_dim = config['dataset_inputs']['wpt_dim']

    # params for training
    dataset_name = config['dataset_module']
    dataset_class_name = config['dataset_class']
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['model_inputs']

    dataset_dir = args.dataset_dir
    
    network_class = get_model_module(module_name, model_name)
    network = network_class(**model_inputs, dataset_type=dataset_mode).to(device)
    network.load_state_dict(torch.load(weight_path))

    dataset_class = get_dataset_module(dataset_name, dataset_class_name)
    val_set = dataset_class(dataset_dir=f'{dataset_dir}/train', **dataset_inputs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

     # validation
    val_dir_losses = []
    val_kl_losses = []
    val_recon_losses = []
    val_total_losses = []
    val_afford_losses = []
    val_dist_losses = []
    val_nn_losses = []

    # # Create pybullet GUI
    # p.connect(p.GUI)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    # p.resetDebugVisualizerCamera(
    #     cameraDistance=0.1,
    #     cameraYaw=80,
    #     cameraPitch=-10,
    #     cameraTargetPosition=[0.0, 0.0, 0.0]
    # )
    # p.resetSimulation()
    # p.setPhysicsEngineParameter(numSolverIterations=150)
    # sim_timestep = 1.0 / 240
    # p.setTimeStep(sim_timestep)
    # p.setGravity(0, 0, 0)

    whole_fs = None
    val_batches = enumerate(val_loader, 0)
    # total_loss, total_precision, total_recall, total_Fscore, total_accu = 0, 0, 0, 0, 0
    for i_batch, (sample_pcds, sample_affords, sample_trajs)  in tqdm(val_batches, total=len(val_loader)):

        # set models to evaluation mode
        network.eval()

        sample_pcds = sample_pcds.to(device).contiguous() 
        sample_trajs = sample_trajs.to(device).contiguous()
        sample_cp = sample_pcds[:, 0]

        with torch.no_grad():

            # f_s, losses = network.get_loss(30000, sample_pcds, sample_trajs, sample_cp, sample_affords)  # B x 2, B x F x N
            losses = network.get_loss(30000, sample_pcds, sample_trajs, sample_cp, sample_affords)  # B x 2, B x F x N

            # whole_fs = f_s if whole_fs is None else torch.vstack((whole_fs, f_s))

            if 'afford' in losses.keys():
                val_afford_losses.append(losses['afford'].item())
            if 'dist' in losses.keys():
                val_dist_losses.append(losses['dist'].item())
            if 'nn' in losses.keys():
                val_nn_losses.append(losses['nn'].item())
            if 'dir' in losses.keys():
                val_dir_losses.append(losses['dir'].item())
            val_kl_losses.append(losses['kl'].item())
            val_recon_losses.append(losses['recon'].item())
            val_total_losses.append(losses['total'].item())
    
    # whole_fs = whole_fs.detach().cpu().numpy()
    # pca = PCA(n_components=3)
    # pca.fit(whole_fs)
    # X_pca = pca.transform(whole_fs)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    if 'afford' in losses.keys():
        val_afford_avg_loss = np.mean(np.asarray(val_afford_losses))
    if 'dist' in losses.keys():
        val_dist_avg_loss = np.mean(np.asarray(val_dist_losses))
    if 'nn' in losses.keys():
        val_nn_avg_loss = np.mean(np.asarray(val_nn_losses))
    if 'dir' in losses.keys():
        val_dir_avg_loss = np.mean(np.asarray(val_dir_losses))
    val_kl_avg_loss = np.mean(np.asarray(val_kl_losses))
    val_recon_avg_loss = np.mean(np.asarray(val_recon_losses))
    val_total_avg_loss = np.mean(np.asarray(val_total_losses))
    print(
            f'''---------------------------------------------\n'''
            f'''[ validation stage ]\n'''
            f''' - val_afford_avg_loss : {val_afford_avg_loss if 'afford' in losses.keys() else 0.0:>10.5f}\n'''
            f''' - val_dist_avg_loss : {val_dist_avg_loss if 'dist' in losses.keys() else 0.0:>10.5f}\n'''
            f''' - val_nn_avg_loss : {val_nn_avg_loss if 'nn' in losses.keys() else 0.0:>10.5f}\n'''
            f''' - val_dir_avg_loss : {val_dir_avg_loss if 'dir' in losses.keys() else 0.0:>10.5f}\n'''
            f''' - val_kl_avg_loss : {val_kl_avg_loss:>10.5f}\n'''
            f''' - val_recon_avg_loss : {val_recon_avg_loss:>10.5f}\n'''
            f''' - val_total_avg_loss : {val_total_avg_loss:>10.5f}\n'''
            f'''---------------------------------------------\n'''
        )

def test(args):

    from PIL import Image
    import pybullet as p
    import pybullet_data
    from pybullet_robot_envs.envs.panda_envs.panda_env import pandaEnv

    # ================== config ==================

    checkpoint_dir = f'{args.checkpoint_dir}'
    config_file = args.config
    verbose = args.verbose
    visualize = args.visualize
    evaluate = args.evaluate
    device = args.device
    dataset_mode = 0 if 'absolute' in checkpoint_dir else 1 # 0: absolute, 1: residual
    weight_subpath = args.weight_subpath
    weight_path = f'{checkpoint_dir}/{weight_subpath}'

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'

    checkpoint_subdir = checkpoint_dir.split('/')[1]
    checkpoint_subsubdir = checkpoint_dir.split('/')[2]

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # sample_num_points = int(weight_subpath.split('_')[0])
    sample_num_points = 1000
    print(f'num of points = {sample_num_points}')

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'
    print(f'checkpoint: {weight_path}')

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for network
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['model_inputs']
    dataset_inputs = config['dataset_inputs']
    batch_size = config['batch_size']

    wpt_dim = config['dataset_inputs']['wpt_dim']
    print(f'wpt_dim: {wpt_dim}')

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

    obj_contact_poses = []
    obj_grasping_infos = []
    obj_urdfs = []
    for inference_obj_path in inference_obj_paths:
        obj_contact_info = json.load(open(inference_obj_path, 'r'))
        obj_contact_poses.append(obj_contact_info['contact_pose'])
        obj_grasping_infos.append(obj_contact_info['initial_pose'][8]) # bottom position

        obj_urdf = '{}/base.urdf'.format(os.path.split(inference_obj_path)[0])
        assert os.path.exists(obj_urdf), f'{obj_urdf} not exists'
        obj_urdfs.append(obj_urdf)

    hook_pcds = []
    hook_affords = []
    hook_urdfs = []
    for inference_hook_path in inference_hook_paths:
        hook_name = inference_hook_path.split('/')[-2]
        points = np.load(inference_hook_path)[:, :3].astype(np.float32)
        affords = np.load(inference_hook_path)[:, 3].astype(np.float32)

        hook_urdf = f'{inference_hook_shape_root}/{hook_name}/base.urdf'
        assert os.path.exists(hook_urdf), f'{hook_urdf} not exists'
        hook_urdfs.append(hook_urdf) 
        hook_pcds.append(points)

    inference_subdir = os.path.split(inference_hook_dir)[-1]
    output_dir = f'inference/inference_trajs/{checkpoint_subdir}/{checkpoint_subsubdir}/{inference_subdir}'
    os.makedirs(output_dir, exist_ok=True)
    
    # ================== Model ==================

    # load model
    network_class = get_model_module(module_name, model_name)
    network = network_class(**model_inputs, dataset_type=dataset_mode).to(device)
    network.load_state_dict(torch.load(weight_path))

    # ================== Simulator ==================

    # Create pybullet GUI
    physics_client_id = None
    physics_client_id = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    # if visualize:
    #     physics_client_id = p.connect(p.GUI)
    #     p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    # else:
    #     physics_client_id = p.connect(p.DIRECT)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.2,
        cameraYaw=90,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0.0, 1.3]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    sim_timestep = 1.0 / 240
    p.setTimeStep(sim_timestep)
    p.setGravity(0, 0, 0)

    # ------------------- #
    # --- Setup robot --- #
    # ------------------- #

    # Load plane contained in pybullet_data
    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"))
    robot = pandaEnv(physics_client_id, use_IK=1)

    # -------------------------- #
    # --- Load other objects --- #
    # -------------------------- #

    p.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"), [1, 0.0, 0.0])

    hook_pose = [
        0.5,
        -0.1,
        1.3,
        4.329780281177466e-17,
        0.7071067811865475,
        0.7071067811865476,
        4.329780281177467e-17
    ]

    # ================== Inference ==================

    batch_size = 1
    all_scores = []
    for sid, pcd in enumerate(hook_pcds):

        if sid > 20:
            break
        
        # urdf file
        hook_urdf = hook_urdfs[sid]
        hook_id = p.loadURDF(hook_urdf, hook_pose[:3], hook_pose[3:])
        # p.resetBasePositionAndOrientation(hook_id, hook_pose[:3], hook_pose[3:])

        # sample trajectories
        centroid_pcd, centroid, scale = normalize_pc(pcd, copy_pts=True) # points will be in a unit sphere
        contact_point = centroid_pcd[0]
        # centroid_pcd = 1.0 * (np.random.rand(pcd.shape[0], pcd.shape[1]) - 0.5).astype(np.float32) # random noise

        points_batch = torch.from_numpy(centroid_pcd).unsqueeze(0).to(device=device).contiguous()
        input_pcid = furthest_point_sample(points_batch, sample_num_points).long().reshape(-1)  # BN
        points_batch = points_batch[0, input_pcid, :].squeeze()
        points_batch = points_batch.repeat(batch_size, 1, 1)

        # contact_point_batch = torch.from_numpy(contact_point).to(device=device).repeat(batch_size, 1)
        # affordance, recon_trajs = network.sample(points_batch, contact_point_batch)

        # generate trajectory using predicted contact points
        affordance, recon_trajs = network.sample(points_batch)

        ###############################################
        # =========== for affordance head =========== #
        ###############################################

        points = points_batch[0].cpu().detach().squeeze().numpy()
        affordance = affordance[0].cpu().detach().squeeze().numpy()
        affordance = (affordance - np.min(affordance)) / (np.max(affordance) - np.min(affordance))
        colors = cv2.applyColorMap((255 * affordance).astype(np.uint8), colormap=cv2.COLORMAP_JET).squeeze()

        # contact_point_cond = np.where(affordance == np.max(affordance))[0]
        # contact_point = points[contact_point_cond][0]

        contact_point_coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        contact_point_coor.translate(contact_point.reshape((3, 1)))

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        if visualize:
            img_list = []
            frames = 36
            rotate_per_frame = np.pi * 2 / frames
            for _ in range(frames):
                r = point_cloud.get_rotation_matrix_from_xyz((0, rotate_per_frame, 0)) # (rx, ry, rz) = (right, up, inner)
                point_cloud.rotate(r, center=(0, 0, 0))
                contact_point_coor.rotate(r, center=(0, 0, 0))
                geometries = [point_cloud, contact_point_coor]

                img = capture_from_viewer(geometries)
                img_list.append(img)
            
            save_path = f"{output_dir}/{weight_subpath[:-4]}-affor-{sid}.gif"
            imageio.mimsave(save_path, img_list, fps=10)

        ##############################################################
        # =========== for trajectory reconstruction head =========== #
        ##############################################################

        hook_poses = torch.Tensor(hook_pose).repeat(batch_size, 1).to(device)
        scales = torch.Tensor([scale]).repeat(batch_size).to(device)
        centroids = torch.from_numpy(centroid).repeat(batch_size, 1).to(device)
        recovered_trajs = recover_trajectory(recon_trajs, hook_poses, centroids, scales, dataset_mode, wpt_dim)

        draw_coordinate(recovered_trajs[0][0], size=0.02)

        # conting inference score using object and object contact information
        if evaluate:
            max_obj_success_cnt = 0
            wpt_ids = []
            for traj_id, recovered_traj in enumerate(recovered_trajs):

                obj_success_cnt = 0
                for i, (obj_urdf, obj_contact_pose, obj_grasping_info) in enumerate(zip(obj_urdfs, obj_contact_poses, obj_grasping_infos)):
                    reversed_recovered_traj = recovered_traj[::-1]
                    reversed_recovered_traj = refine_waypoint_rotation(reversed_recovered_traj)

                    obj_id = p.loadURDF(obj_urdf)
                    rgbs, success = robot_kptraj_hanging(robot, reversed_recovered_traj, obj_id, hook_id, obj_contact_pose, obj_grasping_info, visualize=False)
                    res = 'success' if success else 'failed'
                    obj_success_cnt += 1 if success else 0
                    p.removeBody(obj_id)

                    if len(rgbs) > 0 and traj_id == 0: # only when visualize=True
                        rgbs[0].save(f"{output_dir}/{weight_subpath[:-4]}-{sid}-{i}-{res}.gif", save_all=True, append_images=rgbs, duration=80, loop=0)

                max_obj_success_cnt = max(obj_success_cnt, max_obj_success_cnt)

            all_scores.append(max_obj_success_cnt / len(obj_contact_poses))
        
        p.removeAllUserDebugItems()
               
        if visualize:
            wpt_ids = []
            for i, recovered_traj in enumerate(recovered_trajs):
                colors = list(np.random.rand(3)) + [1]
                for wpt_i, wpt in enumerate(recovered_traj):

                    wpt_id = p.createMultiBody(
                        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, 0.001), 
                        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, 0.001, rgbaColor=colors), 
                        basePosition=wpt[:3]
                    )
                    wpt_ids.append(wpt_id)

            # capture a list of images and save as gif
            delta = 10
            delta_sum = 0
            cameraYaw = 90
            rgbs = []
            while True:
                keys = p.getKeyboardEvents()
                p.resetDebugVisualizerCamera(
                    cameraDistance=0.08,
                    cameraYaw=cameraYaw,
                    cameraPitch=-30,
                    cameraTargetPosition=[0.5, -0.1, 1.3]
                )

                cam_info = p.getDebugVisualizerCamera()
                width = cam_info[0]
                height = cam_info[1]
                view_mat = cam_info[2]
                proj_mat = cam_info[3]
                img_info = p.getCameraImage(width, height, viewMatrix=view_mat, projectionMatrix=proj_mat)
                rgb = img_info[2]
                rgbs.append(Image.fromarray(rgb))

                cameraYaw += delta 
                delta_sum += delta 
                cameraYaw = cameraYaw % 360
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    break
                if delta_sum >= 360:
                    break

            for wpt_id in wpt_ids:
                p.removeBody(wpt_id)

            rgbs[0].save(f"{output_dir}/{weight_subpath[:-4]}-traj-{sid}.gif", save_all=True, append_images=rgbs, duration=80, loop=0)

        p.removeBody(hook_id)

    if evaluate:
        all_scores = np.asarray(all_scores)
        print("===============================================================================================")
        print(f'checkpoint: {weight_path}')
        print(f'inference_dir: {args.inference_dir}')
        print(f'[summary] all success rate: {np.mean(all_scores)}')
        print("===============================================================================================")
        
def main(args):
    dataset_dir = args.dataset_dir
    checkpoint_dir = args.checkpoint_dir
    config_file = args.config

    assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
    assert os.path.exists(checkpoint_dir), f'{checkpoint_dir} not exists'
    assert os.path.exists(config_file), f'{config_file} not exists'

    if args.training_mode == "train":
        train(args)

    if args.training_mode == "val":
        val(args)

    if args.training_mode == "test":
        test(args)


if __name__=="__main__":

    default_dataset = [
        # "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000"
        "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-residual-40/02.27.10.32-1000"
        # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-absolute-30/02.03.13.28",
        # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-residual-30/02.03.13.29",
        # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-absolute-30/02.03.13.30",
        # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-residual-30/02.03.13.34"
    ]

    parser = argparse.ArgumentParser()
    # about dataset
    parser.add_argument('--dataset_dir', '-dd', type=str, default=default_dataset[0])

    # training mode
    parser.add_argument('--training_mode', '-tm', type=str, default='train', help="training mode : [train, test]")
    parser.add_argument('--training_tag', '-tt', type=str, default='', help="training_tag")
    
    # testing
    parser.add_argument('--weight_subpath', '-wp', type=str, default='5000_points-network_epoch-150.pth', help="subpath of saved weight")
    parser.add_argument('--checkpoint_dir', '-cd', type=str, default='checkpoints', help="'training_mode=test' only")
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--evaluate', '-e', action='store_true')
    parser.add_argument('--inference_dir', '-id', type=str, default='')
    parser.add_argument('--obj_shape_root', '-osr', type=str, default='../shapes/inference_objs')
    parser.add_argument('--hook_shape_root', '-hsr', type=str, default='../shapes/hook_all_new_0')
    
    # other info
    parser.add_argument('--device', '-dv', type=str, default="cuda")
    parser.add_argument('--config', '-cfg', type=str, default='../config/traj_recon_affordance_cvae_kl_large.yaml')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.2)
    parser.add_argument('--verbose', '-vb', action='store_true')
    args = parser.parse_args()

    main(args)