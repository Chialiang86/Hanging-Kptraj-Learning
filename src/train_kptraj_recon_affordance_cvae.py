import argparse, yaml, os, time, glob
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

import pybullet as p
from PIL import Image
from scipy.spatial.transform import Rotation as R
from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose, \
                               pose_6d_to_7d, pose_7d_to_6d, draw_coordinate

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    return train_set, val_set

def train(args):

    time_stamp = datetime.today().strftime('%m.%d.%H.%M')
    training_tag = time_stamp if args.training_tag == '' else args.training_tag
    dataset_dir = args.dataset_dir
    dataset_root = args.dataset_dir.split('/')[-2] # dataset category
    dataset_subroot = args.dataset_dir.split('/')[-1] # time stamp
    config_file = args.config
    verbose = args.verbose
    device = args.device
    dataset_mode = 0 if 'absolute' in dataset_dir else 1 # 0: absolute, 1: residual

    config_file_id = config_file.split('/')[-1][:-5] # remove '.yaml'
    checkpoint_dir = f'{args.checkpoint_dir}/{config_file_id}_{training_tag}/{dataset_root}_{dataset_subroot}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for training
    dataset_name = config['dataset_module']
    dataset_class_name = config['dataset_class']
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['inputs']
    
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

    dataset_class = get_dataset_module(dataset_name, dataset_class_name)
    train_set = dataset_class(dataset_dir=f'{dataset_dir}/train', enable_traj=1, enable_affordance=0)
    val_set = dataset_class(dataset_dir=f'{dataset_dir}/val', enable_traj=1, enable_affordance=0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    sample_num_points = train_set.sample_num_points
    print(f'num of points = {sample_num_points}')

    '''
    model_inputs:
        pcd_feat_dim: 256
        traj_feat_dim: 256
        cp_feat_dim: 32
        hidden_dim: 256
        z_feat_dim: 128 # not used
        num_steps: 30
        wpt_dim: 6
        lbd_kl: 1.0 # no use
        lbd_recon: 1.0
        lbd_dir: 1.0
        kl_annealing: 0
    '''
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
        train_kl_losses = []
        train_recon_losses = []
        train_total_losses = []
        for i_batch, (sample_pcds, sample_trajs) in tqdm(train_batches, total=len(train_loader)):

            # set models to training mode
            network.train()

            sample_pcds = sample_pcds.to(device).contiguous() 
            sample_trajs = sample_trajs.to(device).contiguous()
            sample_cp = sample_pcds[:, 0]

            # input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, 4000).long().reshape(-1)  # BN
            # input_pcid2 = furthest_point_sample(sample_pcds, 4000).long().reshape(-1)  # BN
            # input_pcs = sample_pcds[input_pcid1, input_pcid2, :].reshape(batch_size, 4000, -1)
            
            # forward pass
            losses = network.get_loss(sample_pcds, sample_trajs, sample_cp, lbd_kl=kl_weight.get_beta())  # B x 2, B x F x N
            total_loss = losses['total']

            train_kl_losses.append(losses['kl'].item())
            train_recon_losses.append(losses['recon'].item())
            train_total_losses.append(losses['total'].item())

            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()

        network_lr_scheduler.step()
        
        train_kl_avg_loss = np.mean(np.asarray(train_kl_losses))
        train_recon_avg_loss = np.mean(np.asarray(train_recon_losses))
        train_total_avg_loss = np.mean(np.asarray(train_total_losses))
        print(
                f'''---------------------------------------------\n'''
                f'''[ training stage ]\n'''
                f''' - time : {strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} \n'''
                f''' - epoch : {epoch:>5.0f}/{stop_epoch:<5.0f} \n'''
                f''' - lr : {network_opt.param_groups[0]['lr']:>5.2E} \n'''
                f''' - train_kl_avg_loss : {train_kl_avg_loss:>10.5f}\n'''
                f''' - train_recon_avg_loss : {train_recon_avg_loss:>10.5f}\n'''
                f''' - train_total_avg_loss : {train_total_avg_loss:>10.5f}\n'''
                f'''---------------------------------------------\n'''
            )
        
        # save checkpoint
        if (epoch - start_epoch) % save_freq == 0 and (epoch - start_epoch) > 0:
            with torch.no_grad():
                print('Saving checkpoint ...... ')
                torch.save(network.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-network_epoch-{epoch}.pth'))
                torch.save(network_opt.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-optimizer_epoch-{epoch}.pth'))
                torch.save(network_lr_scheduler.state_dict(), os.path.join(checkpoint_dir, f'{sample_num_points}_points-scheduler_epoch-{epoch}.pth'))

        # validation
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
                val_kl_losses.append(losses['kl'].item())
                val_recon_losses.append(losses['recon'].item())
                val_total_losses.append(losses['total'].item())

        val_kl_avg_loss = np.mean(np.asarray(val_kl_losses))
        val_recon_avg_loss = np.mean(np.asarray(val_recon_losses))
        val_total_avg_loss = np.mean(np.asarray(val_total_losses))
        print(
                f'''---------------------------------------------\n'''
                f'''[ validation stage ]\n'''
                f''' - time : {strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} \n'''
                f''' - epoch : {epoch:>5.0f}/{stop_epoch:<5.0f} \n'''
                f''' - lr : {network_opt.param_groups[0]['lr']:>5.2E} \n'''
                f''' - val_kl_avg_loss : {val_kl_avg_loss:>10.5f}\n'''
                f''' - val_recon_avg_loss : {val_recon_avg_loss:>10.5f}\n'''
                f''' - val_total_avg_loss : {val_total_avg_loss:>10.5f}\n'''
                f'''---------------------------------------------\n'''
            )

        kl_weight.update()

def recovery_trajectory(traj : torch.Tensor or np.ndarray, hook_pose : list or np.ndarray, 
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

def test(args):

    # ================== config ==================

    checkpoint_dir = f'{args.checkpoint_dir}'
    config_file = args.config
    verbose = args.verbose
    device = args.device
    dataset_mode = 0 if 'absolute' in checkpoint_dir else 1 # 0: absolute, 1: residual
    weight_subpath = args.weight_subpath
    weight_path = f'{checkpoint_dir}/{weight_subpath}'

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'

    checkpoint_subdir = checkpoint_dir.split('/')[1]
    checkpoint_subsubdir = checkpoint_dir.split('/')[2]
    output_dir = f'inference_trajs/{checkpoint_subdir}/{checkpoint_subsubdir}'
    os.makedirs(output_dir, exist_ok=True)

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # sample_num_points = int(weight_subpath.split('_')[0])
    sample_num_points = 2000
    print(f'num of points = {sample_num_points}')

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for training
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['inputs']

    # ================== Load Inference Shape ==================

    # inference
    inference_shape_dir = '../shapes/hook_validation_small'
    inference_shape_paths = glob.glob(f'{inference_shape_dir}/*/affordance*.npy')
    pcds = []
    affords = []
    urdfs = []
    for inference_shape_path in inference_shape_paths:
        # pcd = o3d.io.read_point_cloud(inference_shape_path)
        # points = np.asarray(pcd.points, dtype=np.float32)
        urdf_prefix = os.path.split(inference_shape_path)[0]
        points = np.load(inference_shape_path)[:, :3].astype(np.float32)
        affords = np.load(inference_shape_path)[:, 3].astype(np.float32)
        urdfs.append(f'{urdf_prefix}/base.urdf') 
        pcds.append(points)
    
    # ================== Model ==================

    # load model
    network_class = get_model_module(module_name, model_name)
    network = network_class(**model_inputs, dataset_type=dataset_mode).to(device)
    network.load_state_dict(torch.load(weight_path))

    if verbose:
        summary(network)

    # ================== Simulator ==================

    # Create pybullet GUI
    # physicsClientId = p.connect(p.DIRECT)
    physicsClientId = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.2,
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

    batch_size = 8
    for sid, pcd in enumerate(pcds):
        
        # urdf file
        hook_urdf = urdfs[sid]
        hook_id = p.loadURDF(hook_urdf, hook_pose[:3], hook_pose[3:])
        # p.resetBasePositionAndOrientation(hook_id, hook_pose[:3], hook_pose[3:])

        # sample trajectories
        centroid_pcd, centroid, scale = normalize_pc(pcd) # points will be in a unit sphere
        contact_point = centroid_pcd[0]

        points_batch = torch.from_numpy(centroid_pcd).unsqueeze(0).to(device=device).contiguous()
        input_pcid = furthest_point_sample(points_batch, sample_num_points).long().reshape(-1)  # BN
        points_batch = points_batch[0, input_pcid, :].squeeze()
        points_batch = points_batch.repeat(batch_size, 1, 1)

        contact_point_batch = torch.from_numpy(contact_point).to(device=device).repeat(batch_size, 1)

        recon_traj = network.sample(points_batch, contact_point_batch)
        recon_traj = recon_traj.cpu().detach().numpy()

        recovered_trajs = recovery_trajectory(recon_traj, hook_pose, centroid, scale, dataset_mode)

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
                cameraDistance=0.12,
                cameraYaw=cameraYaw,
                cameraPitch=-10,
                cameraTargetPosition=[0.0, 0.0, 0.0]
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

        rgbs[0].save(f"{output_dir}/{weight_subpath[:-4]}-{sid}.gif", save_all=True, append_images=rgbs, duration=80, loop=0)
        p.removeBody(hook_id)
        for wpt_id in wpt_ids:
            p.removeBody(wpt_id)
        
def main(args):
    dataset_dir = args.dataset_dir
    checkpoint_dir = args.checkpoint_dir
    config_file = args.config

    assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
    assert os.path.exists(checkpoint_dir), f'{checkpoint_dir} not exists'
    assert os.path.exists(config_file), f'{config_file} not exists'

    if args.training_mode == "train":
        train(args)

    if args.training_mode == "test":
        test(args)


if __name__=="__main__":

    default_dataset = [
        "../data/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-absolute-30/02.03.13.28",
        "../data/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-residual-30/02.03.13.29",
        "../data/traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-absolute-30/02.03.13.30",
        "../data/traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-residual-30/02.03.13.34"
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
    
    # other info
    parser.add_argument('--device', '-dv', type=str, default="cuda")
    parser.add_argument('--config', '-cfg', type=str, default='../config/traj_recon_affordance_cvae_kl_large.yaml')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.2)
    parser.add_argument('--verbose', '-vb', action='store_true')
    args = parser.parse_args()

    main(args)