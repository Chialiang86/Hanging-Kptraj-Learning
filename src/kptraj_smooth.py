import argparse, os, glob, json
import pybullet as p
import numpy as np

from tqdm import tqdm
from PIL import Image
from utils.bullet_utils import draw_coordinate, get_matrix_from_pose, get_pose_from_matrix, pose_6d_to_7d
from utils.testing_utils import trajectory_scoring
from utils.bezier_util import Bezier

def main(args):
    input_dir = args.input_dir
    assert os.path.exists(input_dir), f'{input_dir} not exists'

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create pybullet GUI
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    # p.connect(p.DIRECT)
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

    traj_dirs = glob.glob(f'{input_dir}/*')
    traj_dirs.sort()

    hook_pose = [
        0.0,
        0.0,
        0.0,
        4.329780281177466e-17,
        0.7071067811865475,
        0.7071067811865476,
        4.329780281177467e-17
    ]

    obj_urdf = '../shapes/inference_objs_1/daily_5/base.urdf'
    obj_json = f'../shapes/inference_objs_1/daily_5/hanging_exp_daily_5.json'
    obj_dict = json.load(open(obj_json, 'r'))
    obj_contact_pose = obj_dict['contact_pose']
    obj_id = p.loadURDF(obj_urdf)

    cam_info = p.getDebugVisualizerCamera()
    width = cam_info[0]
    height = cam_info[1]
    view_mat = cam_info[2]
    proj_mat = cam_info[3]

    for traj_dir in tqdm(traj_dirs):

        hook_name = traj_dir.split('/')[-1]
        hook_urdf = f'../shapes/hook_all_new/{hook_name}/base.urdf'
        hook_id = p.loadURDF(hook_urdf, hook_pose[:3], hook_pose[3:])

        traj_jsons = glob.glob(f'{traj_dir}/traj*.json')
        traj_jsons.sort()
        kptrajs = []
        for traj_json in traj_jsons:
            traj_dict = json.load(open(traj_json, 'r'))
            kptraj = traj_dict['trajectory']
            kptrajs.append(kptraj)
            break

        smooth_trajs = []
        for traj_id, kptraj in enumerate(tqdm(kptrajs)):

            kptraj_reverse = []
            for wpt in kptraj[::-1]:
                kptraj_reverse.append(list(pose_6d_to_7d(wpt)))
            
            num = len(kptraj_reverse)
            t_step = 1.0 / (num - 1)
            intersections = np.arange(0.0, 1 + 1e-10, t_step)
            
            trajectory3d = np.asarray(kptraj_reverse)[:, :3]
            trajectory_smooth3d = Bezier.Curve(intersections, trajectory3d)
            trajectory_smooth = np.asarray(kptraj_reverse)
            trajectory_smooth[:, :3] = trajectory_smooth3d

            score, _ = trajectory_scoring(trajectory_smooth, hook_id, obj_id, hook_pose, obj_contact_pose, visualize=False)
            if score > 0:
                smooth_trajs.append(trajectory_smooth.tolist())

        # out_traj_dict = {
        #     'trajectory': smooth_trajs
        # }

        # out_traj_json = f'{args.output_dir}/{hook_name}.json'
        # json.dump(out_traj_dict, open(out_traj_json, 'w'), indent=4)

        # for wpt in kptraj[::-1]:
            
        #     wpt_trans_world = get_matrix_from_pose(hook_pose) @ get_matrix_from_pose(wpt)
        #     wpt_world = get_pose_from_matrix(wpt_trans_world)
        #     draw_coordinate(wpt_world, size=0.001)

        #     obj_pose = get_pose_from_matrix(wpt_trans_world @ np.linalg.inv(get_matrix_from_pose(obj_contact_pose)))
        #     p.resetBasePositionAndOrientation(obj_id, obj_pose[:3], obj_pose[3:])

        p.removeBody(hook_id)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train')
    # parser.add_argument('--input_dir', '-id', type=str, default='../dataset/traj_recon_affordance/kptraj_all_new-absolute-40/alltraj-1000/train/')
    parser.add_argument('--output_dir', '-od', type=str, default='')
    args = parser.parse_args()
    main(args)