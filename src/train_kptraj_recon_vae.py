import argparse, yaml, os, glob, time
import numpy as np
from tqdm import tqdm
from time import strftime
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchsummary import summary
from utils.training_utils import get_model_module, get_dataset_module, optimizer_to_device, kl_annealing

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

    config_file_id = config_file.split('/')[-1][:-5]
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
    train_set = dataset_class(dataset_dir=f'{dataset_dir}/train')
    val_set = dataset_class(dataset_dir=f'{dataset_dir}/val')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

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
                start=0.001, stop=0.01, kl_anneal_cycle=10, kl_anneal_ratio=1)

    # train for every epoch
    for epoch in range(start_epoch, stop_epoch + 1):

        train_batches = enumerate(train_loader, 0)
        val_batches = enumerate(val_loader, 0)

        # training
        train_kl_losses = []
        train_recon_losses = []
        train_total_losses = []
        for i_batch, sample_trajs in tqdm(train_batches):

            # set models to training mode
            network.train()

            sample_trajs = sample_trajs.to(device).contiguous()

            # forward pass
            losses = network.get_loss(sample_trajs, lbd_kl=kl_weight.get_beta())  # B x 2, B x F x N
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
                f'''kl weight: {kl_weight.get_beta()}\n'''
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
                torch.save(network.state_dict(), os.path.join(checkpoint_dir, f'network_epoch-{epoch}.pth'))
                torch.save(network_opt.state_dict(), os.path.join(checkpoint_dir, f'optimizer_epoch-{epoch}.pth'))
                torch.save(network_lr_scheduler.state_dict(), os.path.join(checkpoint_dir, f'scheduler_epoch-{epoch}.pth'))

        # validation
        val_kl_losses = []
        val_recon_losses = []
        val_total_losses = []
        # total_loss, total_precision, total_recall, total_Fscore, total_accu = 0, 0, 0, 0, 0
        for i_batch, sample_trajs in tqdm(val_batches):

            # set models to evaluation mode
            network.eval()

            with torch.no_grad():
                # losses = network.get_loss(sample_trajs.to(device))  # B x 2, B x F x N
                losses = network.get_loss(sample_trajs.to(device), lbd_kl=kl_weight.get_beta())  # B x 2, B x F x N
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

def recovery_trajectory(traj : torch.Tensor or np.ndarray, hook_pose : list or np.ndarray,  dataset_mode : int=0):
    # traj : dim = batch x num_steps x 6
    # dataset_mode : 0 for abosute, 1 for residual 

    if type(traj) == torch.Tensor:
        traj = traj.cpu().detach().numpy()

    base_trans = get_matrix_from_pose(hook_pose)

    waypoints = []

    if dataset_mode == 0: # "absolute"
        for i in range(traj.shape[0]):
            waypoints.append([])
            for j in range(1, traj[i].shape[0]):
                current_trans = base_trans @ get_matrix_from_pose(traj[i, j])
                waypoints[-1].append(get_pose_from_matrix(current_trans, pose_size=6))
    

    if dataset_mode == 1: # "residual"
        for i in range(traj.shape[0]):
            current_trans = base_trans @ get_matrix_from_pose(traj[i, 0])
            current_pose = get_pose_from_matrix(current_trans, pose_size=6)
            waypoints.append([current_pose])
            # waypoints.append([get_pose_from_matrix(current_trans, pose_size=6)])
            for j in range(1, traj[i].shape[0]):
                tmp_pose = np.zeros(6)
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
                # residual_tran = get_matrix_from_pose(traj[i, j])
                # current_trans = current_trans @ residual_tran
                # waypoints[-1].append(get_pose_from_matrix(current_trans, pose_size=6))
    
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

    # params for training
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['inputs']

    # ================== Load Inference Shape ==================

    inference_urdf_dir = '../shapes/hook_validation_small'
    inference_urdf_paths = glob.glob(f'{inference_urdf_dir}/*/base.urdf')
    urdfs = []
    for inference_urdf_path in inference_urdf_paths:
        urdfs.append(inference_urdf_path)
    
    # ================== Model ==================

    # load model

    network_class = get_model_module(module_name, model_name)
    network = network_class(**model_inputs, dataset_type=dataset_mode).to(device)
    network.load_state_dict(torch.load(weight_path))

    if verbose:
        summary(network)

    # ================== Simulator ==================

    # Create pybullet GUI
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
    for uid, urdf in enumerate(urdfs):
        
        # urdf file
        hook_id = p.loadURDF(urdf)
        p.resetBasePositionAndOrientation(hook_id, hook_pose[:3], hook_pose[3:])    

        # inference
        traj = network.sample(batch_size)

        recovered_trajs = recovery_trajectory(traj, hook_pose, dataset_mode)

        wpt_ids = []
        for i, recovered_traj in enumerate(recovered_trajs):
            colors = list(np.random.rand(3)) + [1]
            for wpt_i, wpt in enumerate(recovered_traj):
                wpt_id = p.createMultiBody(
                    baseCollisionShapeIndex=-1, 
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, 0.001, rgbaColor=colors), 
                    basePosition=wpt[:3]
                )
                wpt_ids.append(wpt_id)

        # capture a list of images and save as gif
        delta = 2
        delta_sum = 0
        cameraYaw = 90
        rgbs = []
        while True:
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

            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                break
            if delta_sum >= 360:
                break

        rgbs[0].save(f"{output_dir}/{weight_subpath[:-4]}.gif", save_all=True, append_images=rgbs, duration=40, loop=0)
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
    parser = argparse.ArgumentParser()

    default_dataset = [
        "../data/traj_recon/hook-kptraj_1104_aug-absolute-30/01.16.13.53",
        "../data/traj_recon/hook-kptraj_1104_aug-residual-30/01.16.13.55",
        "../data/traj_recon/hook-kptraj_1104-absolute-30/01.16.13.41",
        "../data/traj_recon/hook-kptraj_1104-residual-30/01.16.13.41"
    ]

    # about dataset
    parser.add_argument('--dataset_dir', '-dd', type=str, default=default_dataset[0])

    # training mode
    parser.add_argument('--training_mode', '-tm', type=str, default='train', help="training mode : [train, test]")
    parser.add_argument('--training_tag', '-tt', type=str, default='', help="training_tag")
    
    # testing
    parser.add_argument('--weight_subpath', '-wp', type=str, default='network_epoch-200.pth', help="subpath of saved weight")
    parser.add_argument('--checkpoint_dir', '-cd', type=str, default='checkpoints', help="'training_mode=test' only")
    
    # other info
    parser.add_argument('--device', '-dv', type=str, default="cuda")
    parser.add_argument('--config', '-cfg', type=str, default='../config/traj_recon_vae_kl_small.yaml')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.2)
    parser.add_argument('--verbose', '-vb', action='store_true')

    args = parser.parse_args()

    main(args)