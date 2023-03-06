import json, glob, argparse, random, os, shutil, time
import torch
import open3d as o3d
import numpy as np
import pybullet as p

from datetime import datetime
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose, \
                               pose_7d_to_6d, draw_coordinate

def refine_waypoint_rotation(wpts : np.ndarray or list):

    assert wpts is not None and len(wpts) > 1, f'the trajectory only contains one waypoint or is None'

    # test direction
    next_pos = wpts[1][:3]
    tmp_pos = wpts[0][:3]
    tmp_dir = np.asarray(next_pos) - np.asarray(tmp_pos) 
    tmp_quat = wpts[0][3:]
    tmp_rotmat = R.from_quat(tmp_quat).as_matrix()
    tmp_rot_dir = (tmp_rotmat @ np.asarray([[1], [0], [0]])).T

    # no need to refine
    if np.dot(tmp_rot_dir, tmp_dir) > 0: 
        return wpts
    
    refine_mat = R.from_rotvec([0, 0, np.pi]).as_matrix()

    refined_wpts = []
    for i in range(len(wpts) - 1):
        tmp_pos = wpts[i][:3]
        tmp_rot = wpts[i][3:]
        tmp_refined_rot = R.from_matrix(R.from_quat(tmp_rot).as_matrix() @ refine_mat).as_quat()
        tmp_refined_pose = list(tmp_pos) + list(tmp_refined_rot)
        refined_wpts.append(tmp_refined_pose)
    return refined_wpts

def shorten_kpt_trajectory(kpt_trajectory : np.ndarray or list, length=20):

    if (type(kpt_trajectory) == list):
        kpt_trajectory = np.asarray(kpt_trajectory)

    assert kpt_trajectory.shape[0] > 0, f'no waypoint in trajectory'
    assert kpt_trajectory.shape[1] == 7, f'waypoint should be in 7d (x, y, z, x, y, z, w) format'

    # tmp_length = 0.0
    # tmp_index = kpt_trajectory.shape[0] - 1

    # for l in range(length):
    #     tmp_length += np.linalg.norm(kpt_trajectory[tmp_index][:3] - kpt_trajectory[tmp_index - 1][:3], ord=2)
    #     tmp_index -= 1
    #     if tmp_index < 0:
    #         break

    # return kpt_trajectory[tmp_index:]
    return kpt_trajectory[-length:]

def dist(wpt1 : list or np.ndarray, wpt2 : list or np.ndarray):
    tmp_trans = get_matrix_from_pose(wpt1)
    next_trans = get_matrix_from_pose(wpt2)
    diff_trans = np.linalg.inv(tmp_trans) @ next_trans
    diff_6pose = get_pose_from_matrix(diff_trans, 6)
    diff_pos_sum = np.sum(diff_6pose[:3] ** 2)
    diff_rot_sum = np.sum((diff_6pose[:3] * 180.0 / np.pi * 0.001) ** 2)
    diff_pose_dist = (diff_pos_sum + diff_rot_sum) ** 0.5
    return diff_pose_dist

def mean_waypt_dist(traj : list or np.ndarray):

    if type(traj) == list:
        traj = np.asarray(traj)
    sum = 0.0
    traj_num = traj.shape[0]
    for i in range(traj.shape[0] - 1):
        tmp_trans = get_matrix_from_pose(traj[i])
        next_trans = get_matrix_from_pose(traj[i+1])
        diff_trans = np.linalg.inv(tmp_trans) @ next_trans
        diff_6pose = get_pose_from_matrix(diff_trans, 6)
        diff_pos_sum = np.sum(diff_6pose[:3] ** 2)
        diff_rot_sum = np.sum((diff_6pose[:3] * 180.0 / np.pi * 0.001) ** 2)
        diff_pose_dist = (diff_pos_sum + diff_rot_sum) ** 0.5
        sum += diff_pose_dist
    
    # print(f"average dist : {sum / traj_num}")
    return sum / traj_num
    

def main(args):

    # Create pybullet GUI
    if args.visualize:
        physicsClientId = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.2,
            cameraYaw=120,
            cameraPitch=-10,
            cameraTargetPosition=[0.5, -0.05, 1.3]
        )
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        sim_timestep = 1.0 / 240
        p.setTimeStep(sim_timestep)
        p.setGravity(0, 0, 0)

        hook_pose = [
            0.49995471253804147,
            -0.057410801277958314,
            1.2987927364900584,
            4.329780281177466e-17,
            0.7071067811865475,
            0.7071067811865476,
            4.329780281177467e-17
        ]

    # dataset attribute configs
    kptraj_type = {0:'absolute', 1:'residual'}[args.kptraj_type]
    dataset_category_id = args.dataset_category
    assert dataset_category_id >=0 and dataset_category_id <= 4, f'dataset_category_id should be in [0, 1, ,2, 3, 4], but got {dataset_category_id}'
    dataset_category = {
                            0:'traj_recon', 
                            1:'traj_recon_shape',
                            2:'traj_recon_affordance',
                            3:'traj_recon_scoring',
                            4:'traj_recon_affordance+scoring',
                            5:'traj_recon_partseg', 
                            6:'traj_recon_partseg+scoring', 
                            7:'traj_recon_affordance_partseg', 
                            8:'traj_recon_affordance_partseg+scoring'
                        }[dataset_category_id]

    # config path or directory information
    kptraj_root = args.kptraj_root
    kptraj_dir = os.path.join(kptraj_root, args.kptraj_dir)
    assert os.path.exists(kptraj_root), f'{kptraj_root} not exists'
    assert os.path.exists(kptraj_dir), f'{kptraj_dir} not exists'

    shape_root = args.shape_root
    src_shape_dir = os.path.join(shape_root, args.shape_dir)
    assert os.path.exists(shape_root), f'{shape_root} not exists'
    assert os.path.exists(src_shape_dir), f'{src_shape_dir} not exists'

    data_root = args.data_root
    assert os.path.exists(data_root), f'{data_root} not exists'

    # the distance threshold of the adjacent waypoints
    sample_dist = args.kptraj_sample_distance
    sample_dist = (2 * sample_dist ** 2) ** 0.5

    # extract hooks ids
    # because the trajectories from same hook family (contain the augmented hooks) 
    # cannot show up in both training/testing set
    dir_files = os.listdir(f'{kptraj_dir}') # all the json files
    kptraj_files = []
    for dir_file in dir_files:
        if 'Hook' in dir_file:
            kptraj_files.append(dir_file)

    # extract the hook ids without augmented hooks 
    # _aug: the trajectories are augmented
    #: the shapes are augmented)
    # IMPORTANT!!! get these hook_ids from keypoint trajectory folder (not from shape folder)
    hook_ids = np.asarray([kptraj_file.split('.')[0].split('_aug')[0].split('#')[0] for kptraj_file in kptraj_files])
    hook_ids = np.unique(hook_ids)

    # traning set and testing set split
    train_split_num = args.split_ratio[0] * len(hook_ids) # default 80%
    val_split_num = args.split_ratio[1] * len(hook_ids) # default 10%
    test_split_num = ((len(hook_ids) - train_split_num - val_split_num) if (len(hook_ids) - train_split_num - val_split_num > 0) else 0) # default 10%

    # at least one hook id is for these sets
    train_split_num = int(np.ceil(train_split_num))
    val_split_num = int(np.ceil(val_split_num))
    test_split_num = int(np.ceil(test_split_num))

    if train_split_num + val_split_num + test_split_num > len(hook_ids):
        train_split_num = len(hook_ids) - val_split_num - test_split_num
    
    random_inds = random.sample(range(len(hook_ids)), len(hook_ids))

    # assign these ids to three different data subsets
    train_hook_ids = hook_ids[random_inds[:train_split_num]]
    val_hook_ids = hook_ids[random_inds[train_split_num:-test_split_num]]
    test_hook_ids = hook_ids[random_inds[-test_split_num:]]

    # config output dataset dir
    time_stamp = datetime.today().strftime('%m.%d.%H.%M')
    dataset_id = f'{time_stamp}-{args.shape_num_pts}' if args.data_tag == '' else f'{args.data_tag}-{args.shape_num_pts}'

    data_subroot = os.path.join(data_root, dataset_category)
    traj_recon_data_train_dir = os.path.join(data_subroot, f'{args.shape_dir}-{args.kptraj_dir}-{kptraj_type}-{args.kptraj_length}', f'{dataset_id}', 'train')
    traj_recon_data_val_dir = os.path.join(data_subroot, f'{args.shape_dir}-{args.kptraj_dir}-{kptraj_type}-{args.kptraj_length}', f'{dataset_id}', 'val')
    traj_recon_data_test_dir = os.path.join(data_subroot, f'{args.shape_dir}-{args.kptraj_dir}-{kptraj_type}-{args.kptraj_length}', f'{dataset_id}', 'test')
    os.makedirs(traj_recon_data_train_dir, exist_ok=True)
    os.makedirs(traj_recon_data_val_dir, exist_ok=True)
    os.makedirs(traj_recon_data_test_dir, exist_ok=True)

    less_pts_num = 0
    short_traj_num = 0

    # log_file (now, this file is used for recording point cloud that contain too less points)
    log_file_path = f'dataprep_{dataset_id}.txt'
    f_out = open(log_file_path, 'w')
    # point_thresh = 1000 # TODO: maybe can write it in input_args

    # for shape_file in shape_files
    for kptraj_file in tqdm(kptraj_files):

        # the file is the information for object
        if 'Hook' not in kptraj_file:
            continue

        # ex : ../raw/keypoint_trajectory_1104/Hook_skew#3_aug.json -> Hook_skew
        shape_name = kptraj_file.split('/')[-1].split('.')[0].split('_aug')[0].split('#')[0]  
        # ex : ../raw/keypoint_trajectory_1104/Hook_skew#3_aug.json -> Hook_skew#3_aug
        shape_name_postfix = kptraj_file.split('/')[-1].split('.')[0]  
        # ex : ../raw/keypoint_trajectory_1104/Hook_skew#3_aug.json -> Hook_skew#3
        shape_name_postfix_without_aug = kptraj_file.split('/')[-1].split('.')[0].split('_aug')[0]

        if args.visualize:
            hook_id = p.loadURDF(f"../shapes/hook/{shape_name_postfix_without_aug}/base.urdf", hook_pose[:3], hook_pose[3:])
            p.resetBasePositionAndOrientation(hook_id,  hook_pose[:3], hook_pose[3:])

        # data_dir
        tgt_data_dir = traj_recon_data_train_dir if np.asarray([hook_id in kptraj_file for hook_id in train_hook_ids]).any() else (
                        traj_recon_data_val_dir if np.asarray([hook_id in kptraj_file for hook_id in val_hook_ids]).any() else traj_recon_data_test_dir
                    ) 

        tgt_data_whole_dir = f'{tgt_data_dir}/{shape_name_postfix}'
        
        # ================================== #
        # for point cloud and affordance map #
        # ================================== #

        # copy point cloud paths to dest directory
        src_shape_path_todo = []
        target_shape_path_todo = []
        if 'shape' in dataset_category:
            # copy point cloud to dest
            src_shape_paths = glob.glob(f'{src_shape_dir}/{shape_name}/base*.ply')
            for src_shape_path in src_shape_paths:

                pcd = o3d.io.read_point_cloud(src_shape_path)
                num_pts = np.asarray(pcd.points).shape[0]
                if num_pts >= args.shape_num_pts: # check num of points
                    shape_id = os.path.split(src_shape_path)[1].split('.')[0].split('-')[-1] # shape-0.ply -> 0
                    target_shape_path = f'{tgt_data_whole_dir}/shape-{shape_id}.ply'
                    src_shape_path_todo.append(src_shape_path)
                    target_shape_path_todo.append(target_shape_path)
            
            if len(target_shape_path_todo) == 0:
                f_out.write(f'{shape_name} doesn\'t contain enough points\n')
                less_pts_num += 1
                continue
        
        src_affordance_path_todo = []
        target_affordance_path_todo = []
        # copy affordance paths to dest directory
        if 'affordance' in dataset_category:
            # copy point cloud to dest
            src_affordance_paths = glob.glob(f'{src_shape_dir}/{shape_name}/affordance*.npy')
            for src_affordance_path in src_affordance_paths:

                src_affordance_map = np.load(src_affordance_path)
                num_pts = src_affordance_map.shape[0]
                if num_pts >= args.shape_num_pts: # check num of points
                    shape_id = os.path.split(src_affordance_path)[1].split('.')[0].split('-')[-1] # affordance-0.npy -> 0
                    target_affordance_path = f'{tgt_data_whole_dir}/affordance-{shape_id}.npy'
                    src_affordance_path_todo.append(src_affordance_path)
                    target_affordance_path_todo.append(target_affordance_path)

            if len(target_affordance_path_todo) == 0:
                f_out.write(f'{shape_name} doesn\'t contain enough points\n')
                less_pts_num += 1
                continue
        
        # ================================ #
        # for semantic keypoint trajectory #
        # ================================ #
        
        # load json
        whole_kptraj_file = os.path.join(kptraj_dir, kptraj_file)
        f_kptraj = open(whole_kptraj_file, 'r')
        json_kptraj = json.load(f_kptraj)
        f_kptraj.close()

        kptrajs = json_kptraj['trajectory'][:args.kptraj_num]
        kptrajs_shorten = []
        target_kptraj_path_todo = []
        
        # preprocess the trajectories in the json file
        for kptraj_id, kptraj in enumerate(kptrajs):
            
            # # decide sample frequency by waypoint intervals
            # mean_traj_wpt_dist = mean_waypt_dist(kptraj)
            # sample_freq = np.ceil(sample_dist / mean_traj_wpt_dist)

            # down sample the waypoints
            # wpts = []
            # for wpt_id, wpt in enumerate(kptraj):
                # if wpt_id % sample_freq == 0:
                #     # add first 6d pose
                #     wpts.append(list(kptraj[wpt_id]))
            
            # copy point cloud to dest
            wpts = [list(kptraj[0])]
            tmp_diff = 0
            tmp_wpt = kptraj[0]
            for wpt_id in range(1, len(kptraj)):
                
                tmp_diff += dist(tmp_wpt, kptraj[wpt_id])
                if tmp_diff > sample_dist:
                    # add first 6d pose
                    wpts.append(list(kptraj[wpt_id]))
                    tmp_diff = 0
                    tmp_wpt = kptraj[wpt_id]

            wpts.append(list(kptraj[-1]))

            # refine the trajectory rotation (roll may need to be rotated by 180 degree)
            wpts = refine_waypoint_rotation(wpts)

            # shorten the trajectory to the specific sizes (default 30)
            kptraj_shorten = shorten_kpt_trajectory(wpts, length=args.kptraj_length)
            if (len(kptraj_shorten) < args.kptraj_length):
                short_traj_num += 1
                continue

            # reverse the trajectory (the first waypoint is the closest waypoint to the contact point)
            kptraj_shorten = kptraj_shorten[::-1]
                
            wpts_shorten = []
            for wpt_id, wpt in enumerate(kptraj_shorten):

                if args.visualize:
                    wpt_trans = get_matrix_from_pose(hook_pose) @ get_matrix_from_pose(wpt)
                    draw_coordinate(wpt_trans)

                if kptraj_type == 'absolute':
                    # absolutely
                    wpts_shorten.append(list(pose_7d_to_6d(kptraj_shorten[wpt_id])))
                
                if kptraj_type == 'residual':
                    # relative
                    if wpt_id == 0:
                    
                        wpts_shorten.append(list(pose_7d_to_6d(kptraj_shorten[0])))
                    
                    else:

                        # add relative 6d pose
                        # relative @ W_i = W_i+1
                        wpt_relative_6d = np.zeros(6)
                        wpt_relative_6d[:3] = kptraj_shorten[wpt_id][:3] - kptraj_shorten[wpt_id - 1][:3] # new - tmp
                        wpt_relative_6d[3:] = R.from_matrix(
                                                    R.from_quat(
                                                        kptraj_shorten[wpt_id][3:]
                                                    ).as_matrix() @ np.linalg.inv(
                                                        R.from_quat(
                                                            kptraj_shorten[wpt_id - 1][3:]
                                                        ).as_matrix()
                                                    )
                                                ).as_rotvec()
                        wpts_shorten.append(list(wpt_relative_6d)) 
                        
                        # # add relative 6d pose
                        # wpt_prev_trans = get_matrix_from_pose(kptraj_shorten[wpt_id - 1])
                        # wpt_now_trans = get_matrix_from_pose(kptraj_shorten[wpt_id])
                        # wpt_relative_trans = wpt_prev_trans @ np.linalg.inv(wpt_now_trans)
                        # wpt_relative_6d = get_pose_from_matrix(wpt_relative_trans, pose_size=6)
                        # wpts_shorten.append(list(wpt_relative_6d)) # from 4x4d to 6d
            
            kptrajs_shorten.append(wpts_shorten)
            target_kptraj_path = f'{tgt_data_whole_dir}/traj-{kptraj_id}.json'
            target_kptraj_path_todo.append(target_kptraj_path)

            if args.visualize:
                p.removeAllUserDebugItems()

            # ========================== #
            # for saving to target paths #
            # ========================== #

        # output info
        os.makedirs(tgt_data_whole_dir, exist_ok=True)

        for traj_id in range(len(target_kptraj_path_todo)):
            kptraj_json = { "trajectory": kptrajs_shorten[traj_id] }
            target_kptraj_path = target_kptraj_path_todo[traj_id]
            f_kptraj = open(target_kptraj_path, 'w')
            json.dump(kptraj_json, f_kptraj, indent=4)
        
        if 'shape' in dataset_category:
            for src_shape_path, target_shape_path in zip(src_shape_path_todo, target_shape_path_todo):
                shutil.copyfile(src_shape_path, target_shape_path)
        
        if 'affordance' in dataset_category:
            for src_affordance_path, target_affordance_path in zip(src_affordance_path_todo, target_affordance_path_todo):
                shutil.copyfile(src_affordance_path, target_affordance_path)
        
        if args.visualize:
            p.removeBody(hook_id)

    print(f'less_pts_num : {less_pts_num}')
    print(f'short_traj_num : {short_traj_num}')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_category', '-dc', type=int, default=0, 
        help=
            '0:traj_recon, 1:traj_recon_shape, 2:traj_recon_affordance, 3:traj_recon_scoring, 4:traj_recon_affordance+scoring, '
    )
    parser.add_argument('--split_ratio', type=float, nargs='+', default=[0.8, 0.1], help='the first value is the ratio of training set, the second is for testing set')
    parser.add_argument('--kptraj_type', '-kt', type=int, default=0, help='0:absolute, 1:residual')
    parser.add_argument('--kptraj_root', '-kr', type=str, default='../raw')
    parser.add_argument('--kptraj_dir', '-kd', type=str, default='kptraj_all_one_0214')
    parser.add_argument('--kptraj_num', '-kn', type=int, default=1000)
    parser.add_argument('--shape_root', '-sr', type=str, default='../shapes')
    parser.add_argument('--shape_dir', '-sd', type=str, default='hook_all')
    parser.add_argument('--shape_num_pts', '-snp', type=int, default=1000, help='the number of points threshold, if the number of points larger than this threshold, then this script will save it')
    parser.add_argument('--data_root', '-dr', type=str, default='../dataset', help='the output dataset directory root')
    parser.add_argument('--data_tag', '-dt', type=str, default='', help='the output dataset tag')
    parser.add_argument('--kptraj_sample_distance', '-ksd', type=float, default=0.0028284) # # ((0.0028284 ** 2) / 2) ** 0.5 ~= 0.002 mm (for position error)
    parser.add_argument('--kptraj_length', '-kl', type=int, default=40)
    parser.add_argument('--visualize', '-v', action='store_true')

    args = parser.parse_args()
    main(args)
