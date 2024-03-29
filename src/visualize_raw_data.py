import json, argparse, glob, os, shutil, time
from scipy.spatial.transform import Rotation as R
from PIL import Image
import torch
import open3d as o3d
import numpy as np
import pybullet as p

from datetime import datetime
from tqdm import tqdm
from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose, \
                               pose_6d_to_7d, pose_7d_to_6d, draw_coordinate


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

def dist(wpt1 : list or np.ndarray, wpt2 : list or np.ndarray):
    # tmp_trans = get_matrix_from_pose(wpt1)
    # next_trans = get_matrix_from_pose(wpt2)
    # diff_trans = np.linalg.inv(tmp_trans) @ next_trans
    # diff_6pose = get_pose_from_matrix(diff_trans, 6)
    # diff_pos_sum = np.sum(diff_6pose[:3] ** 2)
    # diff_rot_sum = np.sum((diff_6pose[:3] * 180.0 / np.pi * 0.001) ** 2)
    # diff_pose_dist = (diff_pos_sum + diff_rot_sum) ** 0.5
    # return diff_pose_dist

    pos_diff = np.asarray(wpt2[:3]) - np.asarray(wpt1[:3])
    tmp_trans = get_matrix_from_pose(wpt1)
    next_trans = get_matrix_from_pose(wpt2)
    diff_trans = np.linalg.inv(tmp_trans) @ next_trans
    diff_6pose = get_pose_from_matrix(diff_trans, 6)
    diff_pos_sum = np.sum(pos_diff ** 2)
    diff_rot_sum = np.sum((diff_6pose[:3] * 180.0 / np.pi * 0.001) ** 2)
    diff_pose_dist = ((diff_pos_sum + diff_rot_sum)) ** 0.5
    diff_pose_dist = np.linalg.norm(pos_diff)
    return diff_pose_dist
    
def refine_waypoint_rotation(wpts : np.ndarray or list):

    assert wpts is not None and len(wpts) > 1, f'the trajectory only contains one waypoint or is None'

    # test direction
    next_pos = wpts[1][:3]
    tmp_pos = wpts[0][:3]
    tmp_dir = np.asarray(next_pos) - np.asarray(tmp_pos) 
    tmp_quat = wpts[0][3:]
    tmp_rotmat = R.from_rotvec(tmp_quat).as_matrix()
    # tmp_rotmat = R.from_quat(tmp_quat).as_matrix()
    tmp_rot_dir = (tmp_rotmat @ np.asarray([[1], [0], [0]])).T

    # no need to refine
    if np.dot(tmp_rot_dir, tmp_dir) > 0: 
        return wpts
    
    refine_mat = R.from_rotvec([0, 0, np.pi]).as_matrix()

    refined_wpts = []
    for i in range(len(wpts) - 1):
        tmp_pos = wpts[i][:3]
        tmp_rot = wpts[i][3:]
        tmp_refined_rot = R.from_matrix(R.from_rotvec(tmp_rot).as_matrix() @ refine_mat).as_quat()
        # tmp_refined_rot = R.from_matrix(R.from_quat(tmp_rot).as_matrix() @ refine_mat).as_quat()
        tmp_refined_pose = list(tmp_pos) + list(tmp_refined_rot)
        refined_wpts.append(tmp_refined_pose)
    return refined_wpts

def shorten_kpt_trajectory(kpt_trajectory : np.ndarray or list, length=20):

    if (type(kpt_trajectory) == list):
        kpt_trajectory = np.asarray(kpt_trajectory)

    assert kpt_trajectory.shape[0] > 0, f'no waypoint in trajectory'
    assert kpt_trajectory.shape[1] == 7 or kpt_trajectory.shape[1] == 6, f'waypoint should be in 7d (x, y, z, x, y, z, w) format'

    # tmp_length = 0.0
    # tmp_index = kpt_trajectory.shape[0] - 1

    # for l in range(length):
    #     tmp_length += np.linalg.norm(kpt_trajectory[tmp_index][:3] - kpt_trajectory[tmp_index - 1][:3], ord=2)
    #     tmp_index -= 1
    #     if tmp_index < 0:
    #         break

    # return kpt_trajectory[tmp_index:]
    return kpt_trajectory[-length:]

def main(args):

    # Create pybullet GUI
    physicsClientId = p.connect(p.GUI)
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

    hook_pose = [
        0.0,
        0.0,
        0.0,
        4.329780281177466e-17,
        0.7071067811865475,
        0.7071067811865476,
        4.329780281177467e-17
    ]

    kptraj_root = args.kptraj_root
    kptraj_dir = os.path.join(kptraj_root, args.kptraj_dir)
    assert os.path.exists(kptraj_root), f'{kptraj_root} not exists'
    assert os.path.exists(kptraj_dir), f'{kptraj_dir} not exists'

    shape_root = args.shape_root
    shape_dir = os.path.join(shape_root, args.shape_dir)
    assert os.path.exists(shape_root), f'{shape_root} not exists'
    assert os.path.exists(shape_dir), f'{shape_dir} not exists'

    sample_dist = args.kptraj_sample_distance
    
    kptraj_files = glob.glob(f'{kptraj_dir}/*devil/traj-0.json')
    kptraj_files.sort()
    # shape_files = glob.glob(f'{shape_dir}/*/*.ply')

    output_dir = f'inference_trajs/raw_data/{args.kptraj_dir}'
    os.makedirs(output_dir, exist_ok=True)

    # for shape_file in shape_files
    for i, kptraj_file in enumerate(tqdm(kptraj_files)):
        if 'Hook' not in kptraj_file:
            continue

        # ../raw/keypoint_trajectory_1104/Hook_skew#3_aug.json -> Hook_skew
        # shape_name = kptraj_file.split('/')[-1].split('.')[0].split('_aug')[0].split('#')[0]  
        # shape_name_postfix = kptraj_file.split('/')[-1].split('.')[0]
        # shape_name_hash = shape_name_postfix.split('_aug')[0]
        shape_name = kptraj_file.split('/')[-2].split('.')[0].split('_aug')[0].split('#')[0]  
        shape_name_postfix = kptraj_file.split('/')[-2].split('.')[0]
        shape_name_hash = shape_name_postfix.split('_aug')[0]

        # affordance_name_complete = f'{shape_dir}/{shape_name}/affordance.npy'
        # pcd_affordance = np.load(affordance_name_complete)
        # contact_point = pcd_affordance[0, :3]
        # print(pcd_affordance)

        urdf_path = f"../shapes/{args.shape_dir}/{shape_name_hash}/base.urdf"
        hook_id = p.loadURDF(urdf_path, hook_pose[:3], hook_pose[3:])
        # print(f'GT pose: {hook_pose}')
        # # get pose of loadURDF
        # pos, rot = p.getBasePositionAndOrientation(hook_id)
        # pose = list(pos) + list(rot)
        # pose1 = np.array(pose)
        # print(f'loadURDF pose: {pose}')
        # p.resetBasePositionAndOrientation(hook_id,  hook_pose[:3], hook_pose[3:])
        # pos, rot = p.getBasePositionAndOrientation(hook_id)
        # pose = list(pos) + list(rot)
        # pose2 = np.array(pose)
        # print(f'resetBasePositionAndOrientation pose: {pose}') #  => same as GT
        # print(pose2 - pose1)

        # load json
        f_kptraj = open(kptraj_file, 'r')
        json_kptraj = json.load(f_kptraj)
        f_kptraj.close()

        kptrajs = json_kptraj['trajectory']
        kptrajs = [kptrajs]

        wpt_ids = []
        for i, kptraj in enumerate(kptrajs):

            if i == 1:
                break
            
            # decide sample frequency by waypoint intervals
            # mean_traj_wpt_dist = mean_waypt_dist(kptraj)
            # sample_freq = np.ceil(sample_dist / mean_traj_wpt_dist)

            # copy point cloud to dest
            wpts = [list(kptraj[0])]
            tmp_diff = 0
            tmp_wpt = kptraj[0]
            for wpt_id in range(1, len(kptraj)):
                
                # tmp_diff += dist(tmp_wpt, kptraj[wpt_id])
                # if tmp_diff > sample_dist:
                #     # add first 6d pose
                wpts.append(list(kptraj[wpt_id]))
                tmp_diff = 0
                tmp_wpt = kptraj[wpt_id]

                # if wpt_id % sample_freq == 0:
                #     # add first 6d pose
                #     wpts.append(list(kptraj[wpt_id]))

            wpts.append(list(kptraj[-1]))
            colors = np.random.uniform(low=[0] * 3, high=[1] * 3).repeat(3).reshape((3,3)).T
            wpts = refine_waypoint_rotation(wpts)
            kptraj_shorten = shorten_kpt_trajectory(wpts, length=args.kptraj_length)
            # for wpt_id, wpt in enumerate(kptraj_shorten):

            #     wpt_trans = get_matrix_from_pose(hook_pose) @ get_matrix_from_pose(wpt)

            #     draw_coordinate(get_pose_from_matrix(wpt_trans), size=0.001, color=colors)
            #     # draw_coordinate(get_pose_from_matrix(wpt_trans), size=0.001)

            # contact_quat = kptraj_shorten[-1][3:]
            # contact_pose = list(contact_point) + list(contact_quat)
            # wpt_trans = get_matrix_from_pose(hook_pose) @ get_matrix_from_pose(contact_pose)
            # draw_coordinate(get_pose_from_matrix(wpt_trans), size=0.005)

            colors = list(np.random.rand(3)) + [1]
            for wpt_i, wpt in enumerate(kptraj_shorten):
                wpt_trans = get_matrix_from_pose(hook_pose) @ get_matrix_from_pose(wpt)
                wpt_world = get_pose_from_matrix(wpt_trans)
                wpt_id = p.createMultiBody(
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, 0.001), 
                    baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, 0.001, rgbaColor=colors), 
                    basePosition=wpt_world[:3]
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

        for wpt_id in wpt_ids:
            p.removeBody(wpt_id)

        # rgbs[0].save(f"{output_dir}/{shape_name}.gif", save_all=True, append_images=rgbs, duration=80, loop=0)
        # print(f"{output_dir}/{shape_name}.gif has been written")

        # while True:
        #     keys = p.getKeyboardEvents()
        #     if ord('q') in keys and keys[ord('q')]&p.KEY_WAS_TRIGGERED:
        #         break

        p.removeAllUserDebugItems()
        p.removeBody(hook_id)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--kptraj_root', '-kr', type=str, default='../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000')
    parser.add_argument('--kptraj_root', '-kr', type=str, default='../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/04.25.20.06-1000')
    parser.add_argument('--kptraj_dir', '-kd', type=str, default='train')
    parser.add_argument('--shape_root', '-sr', type=str, default='../shapes')
    parser.add_argument('--shape_dir', '-sd', type=str, default='hook_all_new')
    parser.add_argument('--kptraj_sample_distance', '-ksd', type=float, default=0.0028284) # ((0.0028284 ** 2) / 2) ** 0.5 ~= 0.002 mm (for position error)
    parser.add_argument('--kptraj_length', '-kl', type=int, default=40)

    args = parser.parse_args()
    main(args)

