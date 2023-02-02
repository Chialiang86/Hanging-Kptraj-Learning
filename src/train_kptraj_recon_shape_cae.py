import argparse, yaml, os, time
import numpy as np

from tqdm import tqdm
from time import strftime

from sklearn.model_selection import train_test_split
from pointnet2_ops.pointnet2_utils import furthest_point_sample

from utils.training_utils import get_model_module, get_dataset_module, optimizer_to_device, normalize_pc

import torch
from torch.utils.data import DataLoader, Subset
from torchsummary import summary

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    return train_set, val_set

def train(args):

    dataset_dir = args.dataset_dir
    dataset_subdir = os.path.split(args.dataset_dir)[1]
    checkpoint_dir = f'{args.checkpoint_dir}/{dataset_subdir}'
    config_file = args.config
    split_ratio = args.split_ratio
    verbose = args.verbose
    device = args.device

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

    dataset_class = get_dataset_module(dataset_name, dataset_class_name)
    all_data = dataset_class(dataset_dir=dataset_dir)
    train_set, val_set = train_val_dataset(all_data, split_ratio) # split_ratio is the ratio validation set
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    network_class = get_model_module(module_name, model_name)
    network = network_class(**model_inputs).to(device)
    
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


    # train for every epoch
    for epoch in range(start_epoch, stop_epoch + 1):

        train_batches = enumerate(train_loader, 0)
        val_batches = enumerate(val_loader, 0)

        # training
        train_kl_losses = []
        train_recon_losses = []
        train_total_losses = []
        for i_batch, (sample_pcds, sample_trajs) in tqdm(train_batches):

            # set models to training mode
            network.train()

            sample_pcds = sample_pcds.to(device).contiguous() 
            sample_trajs = sample_trajs.to(device).contiguous()

            # input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, 4000).long().reshape(-1)  # BN
            # input_pcid2 = furthest_point_sample(sample_pcds, 4000).long().reshape(-1)  # BN
            # input_pcs = sample_pcds[input_pcid1, input_pcid2, :].reshape(batch_size, 4000, -1)
            
            # forward pass
            losses = network.get_loss_test_rotation(sample_pcds, sample_trajs)  # B x 2, B x F x N
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
                f''' - lr : {lr:>5.2E} \n'''
                f''' - train_kl_avg_loss : {train_kl_avg_loss:>10.5f}\n'''
                f''' - train_recon_avg_loss : {train_recon_avg_loss:>10.5f}\n'''
                f''' - train_total_avg_loss : {train_total_avg_loss:>10.5f}\n'''
                f'''---------------------------------------------\n'''
            )
        
        # save checkpoint
        if (epoch - start_epoch) % 5 == 0 and (epoch - start_epoch) > 0:
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
        for i_batch, (sample_pcds, sample_trajs) in tqdm(val_batches):

            # set models to evaluation mode
            network.eval()

            with torch.no_grad():
                losses = network.get_loss(sample_pcds.to(device), sample_trajs.to(device))  # B x 2, B x F x N
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
                f''' - lr : {lr:>5.2E} \n'''
                f''' - val_kl_avg_loss : {val_kl_avg_loss:>10.5f}\n'''
                f''' - val_recon_avg_loss : {val_recon_avg_loss:>10.5f}\n'''
                f''' - val_total_avg_loss : {val_total_avg_loss:>10.5f}\n'''
                f'''---------------------------------------------\n'''
            )


# def network_forward(batch, data_features, network, conf, \
#             is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
#             log_console=False, log_tb=False, tb_writer=None, lr=None):
#     # prepare input
#     input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(conf.device)  # B x 3N x 3   # point cloud
#     batch_size = input_pcs.shape[0]

#     input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
#     input_pcid2 = furthest_point_sample(input_pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
#     input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)

#     input_dirs1 = torch.cat(batch[data_features.index('gripper_direction_world')], dim=0).to(conf.device)  # B x 3 # up作为feature
#     input_dirs2 = torch.cat(batch[data_features.index('gripper_forward_direction_world')], dim=0).to(conf.device)  # B x 3   # forward
#     actual_motion = torch.Tensor(batch[data_features.index('gt_motion')]).to(conf.device)  # B  # 度数
#     task_waypoints = torch.Tensor(batch[data_features.index('task_waypoints')]).to(conf.device)     # 取 waypoint, 4*3 (初始一定是(0,0,0), gripper坐标系)
#     task_traj = torch.cat([torch.cat([input_dirs1, input_dirs2], dim=0).view(conf.batch_size, 2, 3), task_waypoints], dim=1).view(conf.batch_size, conf.num_steps + 1, 3)  # up和forward两个方向拼起来 + waypoints
#     contact_point = torch.Tensor(batch[data_features.index('position_world')]).to(conf.device)

#     losses = network.get_loss(input_pcs, actual_motion, task_traj, contact_point)  # B x 2, B x F x N
#     kl_loss = losses['kl']
#     total_loss = losses['tot']
#     recon_loss = losses['recon']
#     dir_loss = losses['dir']

#     # display information
#     data_split = 'train'
#     if is_val:
#         data_split = 'val'

#     with torch.no_grad():

#         # log to tensorboard
#         if log_tb and tb_writer is not None:
#             tb_writer.add_scalar('total_loss', total_loss.item(), step)
#             tb_writer.add_scalar('kl_loss', kl_loss.item(), step)
#             tb_writer.add_scalar('dir_loss', dir_loss.item(), step)
#             tb_writer.add_scalar('recon_loss', recon_loss.item(), step)
#             tb_writer.add_scalar('lr', lr, step)

#         return losses
        
def main(args):
    dataset_dir = args.dataset_dir
    checkpoint_dir = args.checkpoint_dir
    config_file = args.config

    dataset_subdir = os.path.split(dataset_dir)[1]

    assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
    assert os.path.exists(checkpoint_dir), f'{checkpoint_dir} not exists'
    assert os.path.exists(config_file), f'{config_file} not exists'

    os.makedirs(f'{checkpoint_dir}/{dataset_subdir}', exist_ok=True)

    train(args)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', '-dd', type=str, default='../data/traj_recon/hook-keypoint_trajectory_1104-absolute-2-202301031301')
    parser.add_argument('--checkpoint_dir', '-cd', type=str, default='checkpoints')
    parser.add_argument('--device', '-dv', type=str, default="cuda")
    parser.add_argument('--config', '-cfg', type=str, default='../config/traj_recon.yaml')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.2)
    parser.add_argument('--verbose', '-vb', action='store_true')
    args = parser.parse_args()

    main(args)