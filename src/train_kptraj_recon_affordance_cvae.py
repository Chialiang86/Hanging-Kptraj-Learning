import argparse, json, yaml, os, time, glob
from tqdm import tqdm
from time import strftime
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.training_utils import get_model_module, get_dataset_module, optimizer_to_device, normalize_pc, kl_annealing

import torch
from torch.utils.data import DataLoader, Subset
from torchsummary import summary

from scipy.spatial.transform import Rotation as R
from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose, draw_coordinate
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
        train_nn_losses = []
        train_dir_losses = []
        train_kl_losses = []
        train_recon_losses = []
        train_total_losses = []
        for i_batch, (sample_pcds, sample_trajs) in tqdm(train_batches, total=len(train_loader)):

            # set models to training mode
            network.train()

            sample_pcds = sample_pcds.to(device).contiguous() 
            sample_trajs = sample_trajs.to(device).contiguous()
            sample_cp = sample_pcds[:, 0]
            
            # forward pass
            losses = network.get_loss(sample_pcds, sample_trajs, sample_cp, lbd_kl=kl_weight.get_beta())  # B x 2, B x F x N
            total_loss = losses['total']
            
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
                f''' - train_dist_avg_loss : {train_dist_avg_loss if 'dist' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - train_nn_avg_loss : {train_nn_avg_loss if 'nn' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - train_dir_avg_loss : {train_dir_avg_loss if 'dir' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - train_kl_avg_loss : {train_kl_avg_loss:>10.5f}\n'''
                f''' - train_recon_avg_loss : {train_recon_avg_loss:>10.5f}\n'''
                f''' - train_total_avg_loss : {train_total_avg_loss:>10.5f}\n'''
                f'''---------------------------------------------\n'''
            )
        
        # save checkpoint
        if (epoch - start_epoch) % save_freq == 0 and (epoch - start_epoch) > 0:
            with torch.no_grad():
                print('Saving checkpoint ...... ')
                # torch.save(network, os.path.join(checkpoint_dir, f'{sample_num_points}_points-network_epoch-{epoch}.pth'))
                torch.save(network.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-network_epoch-{epoch}.pth'))
                # torch.save(network_opt.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-optimizer_epoch-{epoch}.pth'))
                # torch.save(network_lr_scheduler.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-scheduler_epoch-{epoch}.pth'))

        # validation
        val_dist_losses = []
        val_nn_losses = []
        val_dir_losses = []
        val_kl_losses = []
        val_recon_losses = []
        val_total_losses = []
        # total_loss, total_precision, total_recall, total_Fscore, total_accu = 0, 0, 0, 0, 0
        for i_batch, (sample_pcds, sample_trajs) in tqdm(val_batches, total=len(val_loader)):

            # set models to evaluation mode
            network.eval()

            sample_pcds = sample_pcds.to(device).contiguous() 
            sample_trajs = sample_trajs.to(device).contiguous()
            sample_cp = sample_pcds[:, 0]

            with torch.no_grad():
                losses = network.get_loss(sample_pcds, sample_trajs, sample_cp, lbd_kl=kl_weight.get_beta())  # B x 2, B x F x N

                if 'dist' in losses.keys():
                    val_dist_losses.append(losses['dist'].item())
                if 'nn' in losses.keys():
                    val_nn_losses.append(losses['nn'].item())
                if 'dir' in losses.keys():
                    val_dir_losses.append(losses['dir'].item())
                val_kl_losses.append(losses['kl'].item())
                val_recon_losses.append(losses['recon'].item())
                val_total_losses.append(losses['total'].item())

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
                f''' - val_dist_avg_loss : {val_dist_avg_loss if 'dist' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - val_nn_avg_loss : {val_nn_avg_loss if 'nn' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - val_dir_avg_loss : {val_dir_avg_loss if 'dir' in losses.keys() else 0.0:>10.5f}\n'''
                f''' - val_kl_avg_loss : {val_kl_avg_loss:>10.5f}\n'''
                f''' - val_recon_avg_loss : {val_recon_avg_loss:>10.5f}\n'''
                f''' - val_total_avg_loss : {val_total_avg_loss:>10.5f}\n'''
                f'''---------------------------------------------\n'''
            )

        kl_weight.update()

def recover_trajectory(traj_src : torch.Tensor or np.ndarray, hook_poses : torch.Tensor or np.ndarray, 
                        centers : torch.Tensor or np.ndarray, scales : torch.Tensor or np.ndarray, dataset_mode : int=0, wpt_dim : int=6):
    # traj : dim = batch x num_steps x 6
    # dataset_mode : 0 for abosute, 1 for residual 

    traj = None
    if type(traj_src) == torch.Tensor:
        traj = traj_src.clone().cpu().detach().numpy()
    if type(centers) == torch.Tensor:
        centers = centers.clone().cpu().detach().numpy()
    if type(scales) == torch.Tensor:
        scales = scales.clone().cpu().detach().numpy()
    if type(hook_poses) == torch.Tensor:
        hook_poses = hook_poses.clone().cpu().detach().numpy()
    
    if type(traj_src) == np.ndarray:
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
                        wpt_0 = traj[traj_id, wpt_id]
                        tmp_pos = wpt_0[:3]
                        tmp_rot = wpt_0[3:]
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
    val_set = dataset_class(dataset_dir=f'{dataset_dir}/val', **dataset_inputs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

     # validation
    val_dir_losses = []
    val_kl_losses = []
    val_recon_losses = []
    val_total_losses = []

    hook_pose = [
        0.0,
        0.0,
        0.0,
        4.329780281177466e-17,
        0.7071067811865475,
        0.7071067811865476,
        4.329780281177467e-17
    ]

    contact_pose = [
        -0.0008948207340292377,
        -0.001648854540121203,
        0.040645438498105646,
        0.08101362598725576,
        0.019759708787220154,
        0.06648792925106443,
        0.9942965863246975
    ]

    # Create pybullet GUI
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.1,
        cameraYaw=80,
        cameraPitch=-10,
        cameraTargetPosition=[0.0, 0.0, 0.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    sim_timestep = 1.0 / 240
    p.setTimeStep(sim_timestep)
    p.setGravity(0, 0, 0)

    hook_urdf = "../shapes/hook_all_new_0/Hook_hcu_303_normal/base.urdf"
    obj_urdf = "../shapes/inference_objs_1/daily_5/base.urdf"
    
    hook_id = p.loadURDF(hook_urdf, hook_pose[:3], hook_pose[3:])
    obj_id = p.loadURDF(obj_urdf)

    val_batches = enumerate(val_loader, 0)
    # total_loss, total_precision, total_recall, total_Fscore, total_accu = 0, 0, 0, 0, 0
    for i_batch, (sample_pcds, sample_trajs, centers, scales) in tqdm(val_batches, total=len(val_loader)):

        # set models to evaluation mode
        network.eval()

        sample_pcds = sample_pcds.to(device).contiguous() 
        sample_trajs = sample_trajs.to(device).contiguous()
        sample_cp = sample_pcds[:, 0]

        with torch.no_grad():
            losses = network.get_loss(sample_pcds, sample_trajs, sample_cp, lbd_kl=1.0)  # B x 2, B x F x N

            # traj_gts = losses['traj_gt']
            # traj_preds = losses['traj_pred']

            # traj_preds_reshape = traj_preds[:, 0].reshape(-1, 2, 3).permute(0, 2, 1)
            # bsz = traj_preds_reshape.shape[0]
            # b1 = torch.nn.functional.normalize(traj_preds_reshape[:, :, 0], p=2, dim=1)
            # a2 = traj_preds_reshape[:, :, 1]
            # b2 = torch.nn.functional.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
            # b3 = torch.cross(b1, b2, dim=1)
            # rotmats = torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

            # traj_preds_recover = traj_preds.clone()
            # traj_preds_recover[:, 0, :3] = centers
            # traj_preds_recover[:, 0, 3:] = torch.Tensor(R.from_matrix(rotmats.cpu().numpy()).as_rotvec()).to(device)

            # hook_poses = torch.Tensor(hook_pose).repeat(batch_size, 1).to(device)
            # recovered_trajs = recover_trajectory(traj_preds_recover, hook_poses, centers, scales, dataset_mode=dataset_mode, wpt_dim=wpt_dim)
                
            # for traj_id, recovered_traj in enumerate(recovered_trajs):

            #     reversed_recovered_traj = recovered_traj[::-1]
            #     reversed_recovered_traj = refine_waypoint_rotation(reversed_recovered_traj)
            #     score, rgbs = trajectory_scoring(reversed_recovered_traj, hook_id, obj_id, [0, 0, 0, 0, 0, 0, 1], obj_contact_pose=contact_pose, visualize=True)

            if losses['dir'] is not None:
                val_dir_losses.append(losses['dir'].item())
            else:
                val_dir_losses.append(0)
            val_kl_losses.append(losses['kl'].item())
            val_recon_losses.append(losses['recon'].item())
            val_total_losses.append(losses['total'].item())

    val_dir_avg_loss = np.mean(np.asarray(val_dir_losses))
    val_kl_avg_loss = np.mean(np.asarray(val_kl_losses))
    val_recon_avg_loss = np.mean(np.asarray(val_recon_losses))
    val_total_avg_loss = np.mean(np.asarray(val_total_losses))
    print(
            f'''---------------------------------------------\n'''
            f'''[ validation results ]\n'''
            f''' - checkpoint: {weight_path}\n'''
            f''' - dataset_dir: {dataset_dir}\n'''
            f''' - wpt_dim: {wpt_dim}\n'''
            f''' - val_dir_avg_loss : {val_dir_avg_loss:>10.5f}\n'''
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

    verbose = args.verbose
    visualize = args.visualize
    evaluate = args.evaluate
    device = args.device
    
    # for trajectory generation checkpoint
    checkpoint_dir = f'{args.checkpoint_dir}'
    dataset_mode = 0 if 'absolute' in checkpoint_dir else 1 # 0: absolute, 1: residual
    weight_subpath = args.weight_subpath
    weight_path = f'{checkpoint_dir}/{weight_subpath}'

    # for affordance checkpoint
    affordance_weight_path = args.affordance_weight

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'
    assert os.path.exists(affordance_weight_path), f'affordance weight file : {affordance_weight_path} not exists'

    checkpoint_subdir = checkpoint_dir.split('/')[1]
    checkpoint_subsubdir = checkpoint_dir.split('/')[2]

    sample_num_points = 1000
    print(f'num of points = {sample_num_points}')

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'
    print('=================================================')
    print(f'affordance checkpoint: {affordance_weight_path}')
    print(f'trajectory generation checkpoint: {weight_path}')
    print('=================================================')

    # ================== config ================== #
    config_file = args.config

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

    afford_config_file = args.afford_config
    with open(afford_config_file, 'r') as f:
        afford_config = yaml.load(f, Loader=yaml.Loader) # dictionary

    afford_module_name = afford_config['module']
    afford_model_name = afford_config['model']
    afford_model_inputs = afford_config['model_inputs']

    # ================== Model ================== #

    # load trajectory generation model
    network_class = get_model_module(module_name, model_name)
    network = network_class(**model_inputs, dataset_type=dataset_mode).to(device)
    network.load_state_dict(torch.load(weight_path))

    # load affordance generation model
    afford_network_class = get_model_module(afford_module_name, afford_model_name)
    afford_network = afford_network_class({'model.use_xyz': afford_model_inputs['model.use_xyz']}).to(device)
    afford_network.load_state_dict(torch.load(affordance_weight_path))

    # ================== Load Inference Shape ================== #

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
        obj_grasping_infos.append(obj_contact_info['initial_pose'][8])

        obj_urdf = '{}/base.urdf'.format(os.path.split(inference_obj_path)[0])
        assert os.path.exists(obj_urdf), f'{obj_urdf} not exists'
        obj_urdfs.append(obj_urdf)

    hook_pcds = []
    hook_affords = []
    hook_urdfs = []

    class_num = 15
    easy_cnt = 0
    normal_cnt = 0
    hard_cnt = 0
    devil_cnt = 0
    for inference_hook_path in inference_hook_paths:

        hook_name = inference_hook_path.split('/')[-2]
        points = np.load(inference_hook_path)[:, :3].astype(np.float32)
        affords = np.load(inference_hook_path)[:, 3].astype(np.float32)
        
        easy_cnt += 1 if 'easy' in hook_name else 0
        normal_cnt += 1 if 'normal' in hook_name else 0
        hard_cnt += 1 if 'hard' in hook_name else 0
        devil_cnt += 1 if 'devil' in hook_name else 0

        if 'easy' in hook_name and easy_cnt > class_num:
            continue
        if 'normal' in hook_name and normal_cnt > class_num:
            continue
        if 'hard' in hook_name and hard_cnt > class_num:
            continue
        if 'devil' in hook_name and devil_cnt > class_num:
            continue

        hook_urdf = f'{inference_hook_shape_root}/{hook_name}/base.urdf'
        assert os.path.exists(hook_urdf), f'{hook_urdf} not exists'
        hook_urdfs.append(hook_urdf) 
        hook_pcds.append(points)

    inference_subdir = os.path.split(inference_hook_dir)[-1]
    output_dir = f'inference/inference_trajs/{checkpoint_subdir}/{checkpoint_subsubdir}/{inference_subdir}'
    os.makedirs(output_dir, exist_ok=True)

    # ================== Simulator ================== #

    # Create pybullet GUI
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
    # p.stepSimulation()

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
    all_scores = {
        'easy': [],
        'normal': [],
        'hard': [],
        'devil': [],
        'all': []
    }
    for sid, pcd in enumerate(hook_pcds):

        # hook urdf file
        hook_urdf = hook_urdfs[sid]
        hook_id = p.loadURDF(hook_urdf, hook_pose[:3], hook_pose[3:])

        # hook name
        hook_name = hook_urdf.split('/')[-2]
        difficulty = 'easy' if 'easy' in hook_name else \
                     'normal' if 'normal' in hook_name else \
                     'hard' if 'hard' in hook_name else  \
                     'devil'

        # sample trajectories
        centroid_pcd, centroid, scale = normalize_pc(pcd, copy_pts=True) # points will be in a unit sphere
        # centroid_pcd = 1.0 * (np.random.rand(pcd.shape[0], pcd.shape[1]) - 0.5).astype(np.float32) # random noise

        points_batch = torch.from_numpy(centroid_pcd).unsqueeze(0).to(device=device).contiguous()
        input_pcid = furthest_point_sample(points_batch, sample_num_points).long().reshape(-1)  # BN
        points_batch = points_batch[0, input_pcid, :].squeeze()
        points_batch = points_batch.repeat(batch_size, 1, 1)

        # contact_point = centroid_pcd[0]
        # contact_point_batch = torch.from_numpy(contact_point).to(device=device).repeat(batch_size, 1)

        # contact point inference
        affordance = afford_network.inference(points_batch)
        affordance_min = torch.unsqueeze(torch.min(affordance, dim=2).values, 1)
        affordance_max = torch.unsqueeze(torch.max(affordance, dim=2).values, 1)
        affordance = (affordance - affordance_min) / (affordance_max - affordance_min)
        contact_cond = torch.where(affordance == torch.max(affordance)) # only high response region selected
        contact_cond0 = contact_cond[0].to(torch.long) # point cloud id
        contact_cond2 = contact_cond[2].to(torch.long) # contact point ind for the point cloud
        contact_point_batch = points_batch[contact_cond0, contact_cond2]

        # contact point inference
        recon_trajs = network.sample(points_batch, contact_point_batch)

        hook_poses = torch.Tensor(hook_pose).repeat(batch_size, 1)
        scales = torch.Tensor([scale]).repeat(batch_size)
        centroids = torch.from_numpy(centroid).repeat(batch_size, 1)
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

            all_scores[difficulty].append(max_obj_success_cnt / len(obj_contact_poses))
            all_scores['all'].append(max_obj_success_cnt / len(obj_contact_poses))
        
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

            p.removeAllUserDebugItems()

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
                    cameraPitch=0,
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

            rgbs[0].save(f"{output_dir}/{weight_subpath[:-4]}-{sid}.gif", save_all=True, append_images=rgbs, duration=80, loop=0)
               
        p.removeBody(hook_id)

    if evaluate:
        easy_mean = np.asarray(all_scores['easy'])
        normal_mean = np.asarray(all_scores['normal'])
        hard_mean = np.asarray(all_scores['hard'])
        devil_mean = np.asarray(all_scores['devil'])
        all_mean = np.asarray(all_scores['all'])
        print("===============================================================================================")
        print('checkpoint: {}'.format(weight_path))
        print('inference_dir: {}'.format(args.inference_dir))
        print('[easy] success rate: {:00.03f}%'.format(np.mean(easy_mean) * 100))
        print('[normal] success rate: {:00.03f}%'.format(np.mean(normal_mean) * 100))
        print('[hard] success rate: {:00.03f}%'.format(np.mean(hard_mean) * 100))
        print('[devil] success rate: {:00.03f}%'.format(np.mean(devil_mean) * 100))
        print('[all] success rate: {:00.03f}%'.format(np.mean(all_mean) * 100))
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
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000"
    ]

    parser = argparse.ArgumentParser()
    # about dataset
    parser.add_argument('--dataset_dir', '-dd', type=str, default=default_dataset[0])

    # training mode
    parser.add_argument('--training_mode', '-tm', type=str, default='train', help="training mode : [train, val, test]")
    parser.add_argument('--training_tag', '-tt', type=str, default='', help="training_tag")
    
    # testing
    parser.add_argument('--weight_subpath', '-wp', type=str, default='1000_points-network_epoch-20000.pth', help="subpath of saved weight")
    parser.add_argument('--checkpoint_dir', '-cd', type=str, default='checkpoints', help="'training_mode=test' only")
    parser.add_argument('--affordance_weight', '-aw', type=str,
                            default='checkpoints/af_msg-03.05.13.45/hook_all_new-kptraj_all_new-absolute-40_03.05.12.50-1000/1000_points-network_epoch-20000.pth', 
                            help="'training_mode=test' only"
                        )
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--evaluate', '-e', action='store_true')
    parser.add_argument('--inference_dir', '-id', type=str, default='')
    parser.add_argument('--obj_shape_root', '-osr', type=str, default='../shapes/inference_objs')
    parser.add_argument('--hook_shape_root', '-hsr', type=str, default='../shapes/hook_all_new_0')
    
    # other info
    parser.add_argument('--device', '-dv', type=str, default="cuda")
    parser.add_argument('--config', '-cfg', type=str, default='../config/traj_af_10.yaml')
    parser.add_argument('--afford_config', '-acfg', type=str, default='../config/af_msg.yaml')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.2)
    parser.add_argument('--verbose', '-vb', action='store_true')
    args = parser.parse_args()

    main(args)