import argparse, json, yaml, os, time, glob, cv2, imageio, copy
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from tqdm import tqdm
from time import strftime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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

    time_stamp = datetime.today().strftime('%m.%d.%H.%M')
    training_tag = time_stamp if args.training_tag == '' else args.training_tag
    dataset_dir = args.dataset_dir
    dataset_root = args.dataset_dir.split('/')[-2] # dataset category
    dataset_subroot = args.dataset_dir.split('/')[-1] # time stamp
    config_file = args.config
    verbose = args.verbose
    device = args.device

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

    dataset_class = get_dataset_module(dataset_name, dataset_class_name)
    train_set = dataset_class(dataset_dir=f'{dataset_dir}/train', **dataset_inputs)
    val_set = dataset_class(dataset_dir=f'{dataset_dir}/val', **dataset_inputs)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    sample_num_points = train_set.sample_num_points
    print(f'num of points = {sample_num_points}')

    '''
    model_inputs:
        model.use_xyz: 256
    '''
    network_class = get_model_module(module_name, model_name)
    # network = network_class({'model.use_xyz': model_inputs['model.use_xyz']}).to(device)
    network = network_class(model_inputs).to(device)
    
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

        # set models to training mode
        network.train()
        # training
        train_total_losses = []
        for i_batch, (sample_pcds, sample_affords) in tqdm(train_batches, total=len(train_loader)):

            sample_pcds = sample_pcds.to(device).contiguous() 
            sample_affords = sample_affords.to(device).contiguous()

            # forward pass
            losses = network.get_loss(sample_pcds, sample_affords)  # B x 2, B x F x N
            train_total_losses.append(losses.item())

            # optimize one step
            network_opt.zero_grad()
            losses.backward()
            network_opt.step()

        network_lr_scheduler.step()
        
        train_total_avg_loss = np.mean(np.asarray(train_total_losses))
        print(
                f'''---------------------------------------------\n'''
                f'''[ training stage ]\n'''
                f''' - time : {strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} \n'''
                f''' - epoch : {epoch:>5.0f}/{stop_epoch:<5.0f} \n'''
                f''' - lr : {network_opt.param_groups[0]['lr']:>5.2E} \n'''
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

        # set models to evaluation mode
        network.eval()

        # validation
        val_total_losses = []
        # total_loss, total_precision, total_recall, total_Fscore, total_accu = 0, 0, 0, 0, 0
        for i_batch, (sample_pcds, sample_affords) in tqdm(val_batches, total=len(val_loader)):

            sample_pcds = sample_pcds.to(device).contiguous() 
            sample_affords = sample_affords.to(device).contiguous()

            with torch.no_grad():
                losses = network.get_loss(sample_pcds, sample_affords)  # B x 2, B x F x N
                val_total_losses.append(losses.item())

        val_total_avg_loss = np.mean(np.asarray(val_total_losses))
        print(
                f'''---------------------------------------------\n'''
                f'''[ validation stage ]\n'''
                f''' - time : {strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} \n'''
                f''' - epoch : {epoch:>5.0f}/{stop_epoch:<5.0f} \n'''
                f''' - lr : {network_opt.param_groups[0]['lr']:>5.2E} \n'''
                f''' - val_total_avg_loss : {val_total_avg_loss:>10.5f}\n'''
                f'''---------------------------------------------\n'''
            )

def val(args):

    dataset_dir = args.dataset_dir
    checkpoint_dir = args.checkpoint_dir
    config_file = args.config
    device = args.device
    weight_subpath = args.weight_subpath
    weight_path = f'{checkpoint_dir}/{weight_subpath}'

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'

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

    dataset_class = get_dataset_module(dataset_name, dataset_class_name)
    val_set = dataset_class(dataset_dir=f'{dataset_dir}/val', **dataset_inputs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    val_batches = enumerate(val_loader, 0)

    # ================== Model ==================

    # load model
    network_class = get_model_module(module_name, model_name)
    network = network_class({'model.use_xyz': model_inputs['model.use_xyz']}).to(device)
    network.load_state_dict(torch.load(weight_path))

    # ================ Validation ===============

    val_total_losses = []
    # set models to evaluation mode
    network.eval()
    # total_loss, total_precision, total_recall, total_Fscore, total_accu = 0, 0, 0, 0, 0
    for i_batch, (sample_pcds, sample_affords) in tqdm(val_batches, total=len(val_loader)):


        sample_pcds = sample_pcds.to(device).contiguous() 
        sample_affords = sample_affords.to(device).contiguous()

        with torch.no_grad():
            losses = network.get_loss(sample_pcds, sample_affords)  # B x 2, B x F x N
            val_total_losses.append(losses.item())

    val_total_avg_loss = np.mean(np.asarray(val_total_losses))
    print(
            f'''---------------------------------------------\n'''
            f'''[ validation stage ]\n'''
            f''' - val_total_avg_loss : {val_total_avg_loss:>10.5f}\n'''
            f'''---------------------------------------------\n'''
        )

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

def test(args):

    # ================== config ==================

    checkpoint_dir = args.checkpoint_dir
    inference_dir = args.inference_dir
    config_file = args.config
    device = args.device
    weight_subpath = args.weight_subpath
    weight_path = f'{checkpoint_dir}/{weight_subpath}'
    evaluate = args.evaluate

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'
    assert evaluate == 0 or evaluate == 1, 'evaluate can noly be 0: affordance, or 1: part segmentation'

    checkpoint_subdir = checkpoint_dir.split('/')[1]
    checkpoint_subsubdir = checkpoint_dir.split('/')[2]

    inference_subdir = os.path.split(inference_dir)[-1]
    output_dir = f'inference/inference_affordances/{checkpoint_subdir}/{checkpoint_subsubdir}/{inference_subdir}'
    os.makedirs(output_dir, exist_ok=True)

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for training
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['model_inputs']

    # ================== Load Inference Shape ==================

    # inference
    realworld = True if 'realworld_hook' in inference_dir else False
    inference_shape_paths = glob.glob(f'{inference_dir}/*/affordance-0.npy') \
                                if args.visualize else (glob.glob(f'{inference_dir}/*/*.npy') \
                                    if args.evaluate else glob.glob(f'{inference_dir}/*/affordance-0.npy'))
    pcds = []
    hook_names = []
    affordances = []
    urdfs = []
    for inference_shape_path in inference_shape_paths:
        # pcd = o3d.io.read_point_cloud(inference_shape_path)
        # points = np.asarray(pcd.points, dtype=np.float32)
        hook_name = inference_shape_path.split('/')[-2]
        urdf_prefix = os.path.split(inference_shape_path)[0]
        data = np.load(inference_shape_path)
        points = data[:, :3].astype(np.float32)
        urdfs.append(f'{urdf_prefix}/base.urdf') 
        pcds.append(points)
        hook_names.append(hook_name)
        if data.shape[1] > 3: # affordance only
            affor_dim = 3 if evaluate == 1 else 4
            affordance = data[:, affor_dim].astype(np.float32)
            affordances.append(affordance)
    
    # ================== Model ==================

    # load model
    network_class = get_model_module(module_name, model_name)
    network = network_class({'model.use_xyz': model_inputs['model.use_xyz']}).to(device)
    network.load_state_dict(torch.load(weight_path))

    # ================== Inference ==================
    frames = 20
    rotate_per_frame = (2 * np.pi) / frames

    batch_size = 1
    within_5mm_cnt = 0
    within_10mm_cnt = 0
    differences = []
    for sid, pcd in enumerate(tqdm(pcds)):
        
        # sample trajectories
        pcd_copy = copy.deepcopy(pcd)
        centroid_pcd, centroid, scale = normalize_pc(pcd_copy) # points will be in a unit sphere

        points_batch = torch.from_numpy(centroid_pcd).unsqueeze(0).to(device=device).contiguous()
        points_batch = points_batch.repeat(batch_size, 1, 1)

        # pcd_feat, affords = network.inference(points_batch)
        affords = network.inference_sigmoid(points_batch)
        affords = (affords - torch.min(affords)) / (torch.max(affords) - torch.min(affords))
        part_cond = torch.where(affords > 0.25) # only high response region selected
        print(f'hook_name:{hook_names[sid]}, sid:{sid}, partsize:{len(part_cond[0])}')
        affords = affords.squeeze().cpu().detach().numpy()

        points = pcd
        colors = cv2.applyColorMap((255 * affords).astype(np.uint8), colormap=cv2.COLORMAP_JET).squeeze()

        contact_point_cond = np.where(affords == np.max(affords))[0]
        contact_point = points[contact_point_cond][0]

        # save_path = f"{output_dir}/{weight_subpath[:-4]}-cpfeat-{sid}.jpg"
        # plt.plot(pcd_feat.cpu().detach().numpy()[0, :, 0])
        # plt.title('The feature of contact point')
        # plt.savefig(save_path)
        # plt.clf()

        # for part segmentation
        # part_cond = np.where(affords > 0.5)[0]
        # part_feat = pcd_feat[0, :, part_cond].cpu().detach().numpy()
        # mean_feat = np.mean(part_feat, axis=1)
        # max_feat = np.max(part_feat, axis=1)
        # plt.plot(mean_feat)
        # plt.title('The feature of contact point')
        # plt.show()
        # plt.clf()
        # save_path = f"{output_dir}/{weight_subpath[:-4]}-cpfeat-{sid}.jpg"
        # plt.plot(max_feat)
        # plt.title('The feature of contact point')
        # plt.savefig(save_path)
        # plt.clf()

        # print(pcd_feat[part_cond].shape)
        # centroid_pcd_part = centroid_pcd[part_cond]
        # centroid_pcd_part = torch.from_numpy(centroid_pcd_part).unsqueeze(0).to('cuda').contiguous()
        # ind = furthest_point_sample(centroid_pcd_part, 200).long().reshape(-1).cpu().detach().numpy()  # BN
        # input_pcid = part_cond[ind]

        # plt.plot()
        # plt.title('The feature of contact point')
        # plt.savefig(save_path)
        # plt.clf()

        # part_colors = np.zeros(points.shape)
        # part_colors[input_pcid] = colors[input_pcid]
        # print(input_pcid.shape)
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(points)
        # point_cloud.colors = o3d.utility.Vector3dVector(part_colors)
        # o3d.visualization.draw_geometries([point_cloud])

        if not realworld and evaluate:
            contact_point_gt = pcd[0]
            difference = np.linalg.norm(contact_point - contact_point_gt, ord=2)
            differences.append(difference)
            if difference < 0.005:
                within_5mm_cnt += 1
            if difference < 0.01:
                within_10mm_cnt += 1

        contact_point_coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
        contact_point_coor.translate(contact_point.reshape((3, 1)))

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255)

        if args.visualize:
            img_list = []
            for _ in range(frames):
                r = point_cloud.get_rotation_matrix_from_xyz((0, rotate_per_frame, 0)) # (rx, ry, rz) = (right, up, inner)
                point_cloud.rotate(r, center=(0, 0, 0))
                contact_point_coor.rotate(r, center=(0, 0, 0))
                geometries = [point_cloud, contact_point_coor]

                img = capture_from_viewer(geometries)
                img_list.append(img)
            save_path = f"{output_dir}/{weight_subpath[:-4]}-{sid}.gif"
            imageio.mimsave(save_path, img_list, fps=10)
            print(f'{save_path} saved')
            
            # r = point_cloud.get_rotation_matrix_from_xyz((0, (2 * np.pi) / 2, 0)) # (rx, ry, rz) = (right, up, inner)
            # point_cloud.rotate(r, center=(0, 0, 0))
            # contact_point_coor.rotate(r, center=(0, 0, 0))
            # geometries = [point_cloud, contact_point_coor]
            # img = capture_from_viewer(geometries)
            # save_path = f"{output_dir}/{weight_subpath[:-4]}-{sid}.png"
            # imageio.imwrite(save_path, img)
            # print(f'{save_path} saved')

    if evaluate == 1:
        # differences = np.asarray(differences)
        # interval = 0.001
        # low = np.floor(np.min(differences) / interval) * interval
        # high = np.ceil(np.max(differences) / interval) * interval
        # cnts = []
        # for inter in np.arange(low, high, interval):
        #     cond = np.where((differences >= inter) & (differences < inter + interval))
        #     cnt = len(cond[0])
        #     cnts.append(cnt)
        
        # xticks = [f'{np.round(num * 100000) / 100000}' for num in np.arange(low, high, interval)]

        # plt.figure(figsize=(8, 12))
        # plt.ylabel('count')
        # plt.xlabel('num of points')
        # plt.xticks(range(len(cnts)), xticks, rotation='vertical')
        # plt.bar(range(len(cnts)), cnts)
        # plt.title('Distance Distribution')
        # plt.show()

        print('======================================')
        print('inference_dir: {}'.format(inference_dir))
        print('weight_path: {}'.format(weight_path))
        print('mean distance: {:.4f}'.format(np.mean(differences)))
        print('within 5mm rate: {:.4f} ({}/{})'.format(within_5mm_cnt / len(pcds), within_5mm_cnt, len(pcds)))
        print('within 10mm rate: {:.4f} ({}/{})'.format(within_10mm_cnt / len(pcds), within_10mm_cnt, len(pcds)))
        print('======================================')


def analysis(args):

    import matplotlib.pyplot as plt

    # ================== config ==================

    checkpoint_dir = args.checkpoint_dir
    inference_dir = args.inference_dir
    config_file = args.config
    device = args.device
    weight_subpath = args.weight_subpath
    weight_path = f'{checkpoint_dir}/{weight_subpath}'

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'

    checkpoint_subdir = checkpoint_dir.split('/')[1]
    checkpoint_subsubdir = checkpoint_dir.split('/')[2]

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    assert os.path.exists(weight_path), f'weight file : {weight_path} not exists'

    config = None
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader) # dictionary

    # params for training
    module_name = config['module']
    model_name = config['model']
    model_inputs = config['model_inputs']
    
    # ================== Model ==================

    # load model
    network_class = get_model_module(module_name, model_name)
    network = network_class({'model.use_xyz': model_inputs['model.use_xyz']}).to(device)
    network.load_state_dict(torch.load(weight_path))

    # ================== Load Inference Shape ==================

    inference_hook_dir = args.inference_dir # for hook shapes
    inference_hook_whole_dirs = glob.glob(f'{inference_hook_dir}/*')

    inference_hook_paths = []
    for inference_hook_path in inference_hook_whole_dirs:
        # if 'Hook' in inference_hook_path:
        paths = glob.glob(f'{inference_hook_path}/affordance-*.npy')
        paths.sort(key=lambda x : int(x.split('/')[-1].split('-')[-1].split('.')[0])) # sort by trajectory id : [parent_dir]/traj-8.json => 8
        paths = paths[:10]
        inference_hook_paths.extend(paths) 
    
    hook_pcds = []
    hook_names = []
    hook_trajectories = []
    for inference_hook_path in inference_hook_paths:

        hook_name = inference_hook_path.split('/')[-2]
        points = np.load(inference_hook_path).astype(np.float32)
        
        hook_pcds.append(points)
        hook_names.append(hook_name)

        traj_path = f'{os.path.split(inference_hook_path)[0]}/traj-0.json'
        wpts = json.load(open(traj_path, 'r'))['trajectory']
        hook_trajectories.append(np.asarray(wpts))

    inference_subdir = os.path.split(inference_hook_dir)[-1]
    output_dir = f'inference/analysis/{checkpoint_subdir}/{checkpoint_subsubdir}/{inference_subdir}'
    os.makedirs(output_dir, exist_ok=True)
    
    # ================== Inference ==================

    interval = 500
    point_nums = {interval * n: 0 for n in range(15)}
    
    batch_size = 1
    difficulties = []

    whole_feats_parts = None
    for sid, (hook_name, pcd) in enumerate(tqdm(zip(hook_names, hook_pcds), total=len(hook_pcds))):

        # hook name
        difficulty = 'easy' if 'easy' in hook_name else \
                     'normal' if 'normal' in hook_name else \
                     'hard' if 'hard' in hook_name else  \
                     'devil'
        difficulties.append(difficulty)
        
        # sample trajectories
        pcd_copy = copy.deepcopy(pcd[:, :3])
        centroid_pcd, centroid, scale = normalize_pc(pcd_copy, copy_pts=True) # points will be in a unit sphere
        contact_point = centroid_pcd[0]

        # points_batch = torch.from_numpy(centroid_pcd).unsqueeze(0).to(device=device).contiguous()
        # input_pcid = furthest_point_sample(points_batch, sample_num_points).long().reshape(-1)  # BN
        # points_batch = points_batch[0, input_pcid, :].squeeze()
        # points_batch = points_batch.repeat(batch_size, 1, 1)

        # contact_point_batch = torch.from_numpy(contact_point).to(device=device).repeat(batch_size, 1)
        # affordance, recon_trajs = network.sample(points_batch, contact_point_batch)

        points_batch = torch.from_numpy(centroid_pcd).unsqueeze(0).to(device=device).contiguous()
        points_batch = points_batch.repeat(batch_size, 1, 1)

        # generate trajectory using predicted contact points
        affords, whole_feat = network.inference_sigmoid(points_batch, return_feat=True)

        # normalize affordance
        affords = (affords - torch.min(affords)) / (torch.max(affords) - torch.min(affords))

        ##############################################################
        # =========== for whole shape feature extraction =========== #
        ##############################################################

        contact_point_cond = torch.where(affords == torch.max(affords))
        whole_feats_part = whole_feat[contact_point_cond[0], :, contact_point_cond[2]]

        #######################################################
        # =========== for part feature extraction =========== #
        #######################################################

        part_cond = torch.where(affords > 0.1) # only high response region selected
        part_point_num = len(part_cond[-1])
        point_nums[int((part_point_num // interval) * interval)] += 1

        # point_num = pcd.shape[0]
        # point_nums[int((point_num // interval) * interval)] += 1

        # part_cond0 = part_cond[0].to(torch.long)
        # part_cond2 = part_cond[2].to(torch.long)
        # whole_feats_part = whole_feat[:, :, 0].clone()
        # for i in range(batch_size):
            
        #     cond = torch.where(part_cond0 == i)[0] # choose the indexes for the i'th point cloud
        #     tmp_max = torch.max(whole_feat[i, :, part_cond2[cond]], dim=1).values # get max pooling feature using that 10 point features from the sub point cloud 
        #     whole_feats_part[i] = tmp_max
        
        ##############################################################
        # =========== for whole shape feature extraction =========== #
        ##############################################################
        
        # whole_feats_part = torch.max(whole_feat, dim=2).values

        whole_feats_part = whole_feats_part.cpu().detach()
        if whole_feats_parts is None:
            whole_feats_parts = whole_feats_part
        else :
            whole_feats_parts = torch.cat([whole_feats_parts, whole_feats_part], axis=0)
        
        # ###############################################
        # # =========== for affordance head =========== #
        # ###############################################

        # points = points_batch[0].cpu().detach().squeeze().numpy()
        # affordance = affordance[0].cpu().detach().squeeze().numpy()
        # affordance = (affordance - np.min(affordance)) / (np.max(affordance) - np.min(affordance))
        # colors = cv2.applyColorMap((255 * affordance).astype(np.uint8), colormap=cv2.COLORMAP_JET).squeeze()

        # # contact_point_cond = np.where(affordance == np.max(affordance))[0]
        # # contact_point = points[contact_point_cond][0]

        # contact_point_coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        # contact_point_coor.translate(contact_point.reshape((3, 1)))

        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(points)
        # point_cloud.colors = o3d.utility.Vector3dVector(colors / 255)

    
    x_keys = list(point_nums.keys())
    x_labels = [f'{x_keys[i]} ~ {x_keys[i+1]}' for i in range(len(x_keys) - 1)]
    x_labels.append(f'> {x_keys[-1]}')
    y_values = point_nums.values()
    subset = inference_dir.split('/')[-1]
    plt.figure(figsize=(8, 12))
    plt.title('part point distribution')
    plt.xticks(range(len(x_labels)), x_labels, rotation = 90)
    plt.bar(range(len(y_values)), y_values)
    plt.savefig(f'part_{weight_subpath}_{subset}.png')
    
    # whole_feats_parts = whole_feats_parts.numpy()
    # X_embedded = PCA(n_components=2).fit_transform(whole_feats_parts)
    # X_embedded = np.hstack((X_embedded, np.zeros((X_embedded.shape[0], 1)))).astype(np.float32)

    # centroid_points = torch.from_numpy(X_embedded).unsqueeze(0).to('cuda').contiguous()
    # input_pcid = furthest_point_sample(centroid_points, 10).cpu().long().reshape(-1)  # BN
    # selected_points = X_embedded[input_pcid]

    # num_nn = 5
    # nn = NearestNeighbors(n_neighbors=num_nn, algorithm='ball_tree').fit(X_embedded)
    # distances, indices = nn.kneighbors(selected_points)

    # fig = plt.figure(figsize=(16, 12))
    # ax = fig.add_subplot()
    # ax.scatter(X_embedded[:, 0],   X_embedded[:, 1], c=[[0.8, 0.8, 0.8]], s=10)

    # colors = cv2.applyColorMap((255 * np.linspace(0, 1, num_nn)).astype(np.uint8), colormap=cv2.COLORMAP_JET).squeeze() / 255.0
    # for cls_id, (indice, color) in enumerate(zip(indices, colors)):

    #     for ind in indice:
    #         pcd = hook_pcds[ind]
    #         pcd_points = pcd[:, :3]
    #         pcd_afford = pcd[:, 4]
    #         pcd_colors = cv2.applyColorMap((255 * pcd_afford).astype(np.uint8), colormap=cv2.COLORMAP_JET).squeeze()
    #         point_cloud = o3d.geometry.PointCloud()
    #         point_cloud.points = o3d.utility.Vector3dVector(pcd_points)
    #         point_cloud.colors = o3d.utility.Vector3dVector(pcd_colors / 255)
    #         r = point_cloud.get_rotation_matrix_from_xyz((0, np.pi / 2, 0)) # (rx, ry, rz) = (right, up, inner)
    #         point_cloud.rotate(r, center=(0, 0, 0))

    #         wpts = hook_trajectories[ind]
    #         geometries = []
    #         for wpt_raw in wpts[:20]:
    #             wpt = wpt_raw[:3]
    #             coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.003)
    #             coor.translate(wpt.reshape((3, 1)))
    #             coor.rotate(r, center=(0, 0, 0))
    #             geometries.append(coor)
    #         geometries.append(point_cloud)

    #         img = capture_from_viewer(geometries)
    #         save_path = f'{output_dir}/cls-{cls_id}-{ind}.png'
    #         imageio.imsave(save_path, img)
    #         print(f'{save_path} saved')

    #     ax.scatter(X_embedded[indice, 0],   X_embedded[indice, 1], c=color.reshape(1, -1), s=10)

    # out_path = f'{output_dir}/fps-clustering.png'
    # plt.savefig(out_path)
    # plt.clf()
    
    # difficulties = np.asarray(difficulties)
    # easy_ind = np.where(difficulties == 'easy')[0]
    # normal_ind = np.where(difficulties == 'normal')[0]
    # hard_ind = np.where(difficulties == 'hard')[0]
    # devil_ind = np.where(difficulties == 'devil')[0]
    # max_rgb = 255.0
    # plt.scatter(X_embedded[easy_ind, 0],   X_embedded[easy_ind, 1],   c=[[123/max_rgb, 234/max_rgb ,0/max_rgb]],   label='easy',   s=10)
    # plt.scatter(X_embedded[normal_ind, 0], X_embedded[normal_ind, 1], c=[[123/max_rgb, 0/max_rgb   ,234/max_rgb]], label='normal', s=10)
    # plt.scatter(X_embedded[hard_ind, 0],   X_embedded[hard_ind, 1],   c=[[234/max_rgb, 123/max_rgb ,0/max_rgb]],   label='hard',   s=10)
    # plt.scatter(X_embedded[devil_ind, 0],  X_embedded[devil_ind, 1],  c=[[234/max_rgb, 0/max_rgb   ,123/max_rgb]], label='devil',  s=10)
    # plt.legend()
    # plt.savefig(f'{output_dir}/pca_{checkpoint_subdir}_{weight_subpath}.png')

def main(args):
    dataset_dir = args.dataset_dir
    checkpoint_dir = args.checkpoint_dir
    config_file = args.config

    if dataset_dir != '':
        assert os.path.exists(dataset_dir), f'{dataset_dir} not exists'
    assert os.path.exists(checkpoint_dir), f'{checkpoint_dir} not exists'
    assert os.path.exists(config_file), f'{config_file} not exists'

    if args.training_mode == "train":
        train(args)

    if args.training_mode == "val":
        val(args)

    if args.training_mode == "test":
        test(args)

    if args.training_mode == "analysis":
        analysis(args)


if __name__=="__main__":

    default_dataset = [

        "../dataset/traj_recon_affordance/kptraj_all_one_0214-absolute-40/02.15.17.24",
        # "../dataset/traj_recon_affordance/hook_all-kptraj_all_one_0214-residual-40/02.15.17.24",
        # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-absolute-30/02.03.13.28",
        # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-residual-30/02.03.13.29",
        # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-absolute-30/02.03.13.30",
        # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-residual-30/02.03.13.34"
    ]

    parser = argparse.ArgumentParser()
    # about dataset
    parser.add_argument('--dataset_dir', '-dd', type=str, default='')

    # training mode
    parser.add_argument('--training_mode', '-tm', type=str, default='train', help="training mode : [train, test]")
    parser.add_argument('--training_tag', '-tt', type=str, default='', help="training_tag")
    
    # testing
    parser.add_argument('--weight_subpath', '-wp', type=str, default='', help="subpath of saved weight")
    parser.add_argument('--checkpoint_dir', '-cd', type=str, default='checkpoints', help="'training_mode=test' only")
    parser.add_argument('--inference_dir', '-id', type=str, default='../shapes/hook_all')
    
    # other info
    parser.add_argument('--device', '-dv', type=str, default="cuda")
    parser.add_argument('--config', '-cfg', type=str, default='../config/affordance.yaml')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.2)
    parser.add_argument('--verbose', '-vb', action='store_true')
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--evaluate', '-e', action='store_true')
    args = parser.parse_args()

    main(args)