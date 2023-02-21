import argparse, glob, os, json
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from src.utils.bullet_utils import draw_coordinate, get_matrix_from_pos_rot, get_matrix_from_pose, get_pose_from_matrix
from scipy.spatial.transform import Rotation as R

def reconstruct_traj(traj : list or np.ndarray):
    
    if type(traj) == list:
        traj = np.asarray(traj)

    assert traj.shape[1] == 6, f'waypoint size must be 6, but got {traj.shape[1]}'

    first_pos = traj[0, :3]
    first_rotmat = R.from_rotvec(traj[0, 3:]).as_matrix()

    next_pos = first_pos
    next_rotmat = first_rotmat

    ret_traj = traj[0]
    for i in range(1, traj.shape[0]):
        tmp_wpt = traj[i]

        next_pos = next_pos + tmp_wpt[:3]
        next_rotmat = R.from_rotvec(tmp_wpt[3:]).as_matrix() @ next_rotmat
        next_wpt = np.hstack((next_pos, R.from_matrix(next_rotmat).as_rotvec()))

        ret_traj = np.vstack((ret_traj, next_wpt))
    
    return ret_traj

def main(args):

    input_dir = f'{args.input_root}/{args.input_dir}'
    assert os.path.exists(input_dir), f'{input_dir} not exists'

    traj_jsons = glob.glob(f'{input_dir}/*/*.json')
    traj_jsons.sort()

    trajs = []
    for traj_json in traj_jsons:

        traj_dict = json.load(open(traj_json, 'r'))
        trajs.append(traj_dict["trajectory"])
    
    trajs = np.asarray(trajs)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    c = trajs[:,1,2]
    ax.scatter(trajs[:, 1,0], trajs[:, 1,1], trajs[:,1,2], c=c, cmap='jet')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D Position Distribution')
    plt.savefig('euler_res_pos.png')
    c = trajs[:,1,5]
    ax.scatter(trajs[:, 1,3], trajs[:, 1,4], trajs[:,1,5], c=c, cmap='jet')
    ax.legend()
    ax.set_xlabel('roll')
    ax.set_ylabel('pitch')
    ax.set_zlabel('yaw')
    ax.set_title('3D Rotation Distribution')
    plt.savefig('euler_res_rot.png')


    # mean_traj = np.mean(trajs, axis=0)
    # ret_traj = reconstruct_traj(mean_traj)
    
    # # Create pybullet GUI
    # # physicsClientId = p.connect(p.DIRECT)
    # physicsClientId = p.connect(p.GUI)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    # p.resetDebugVisualizerCamera(
    #     cameraDistance=0.2,
    #     cameraYaw=90,
    #     cameraPitch=-1,
    #     cameraTargetPosition=[0.0, 0.0, 0.0]
    # )
    # p.resetSimulation()
    # p.setPhysicsEngineParameter(numSolverIterations=150)
    # sim_timestep = 1.0 / 240
    # p.setTimeStep(sim_timestep)
    # p.setGravity(0, 0, 0)

    # pos = np.asarray([0, 0, 0])
    # rot = p.getQuaternionFromEuler([np.pi/2, 0, np.pi])
    # p.loadURDF('shapes/hook/Hook_60/base.urdf', pos, rot)
    # base_trans = get_matrix_from_pos_rot(pos, rot)

    # draw_coordinate(np.asarray([0, 0, 0, 0, 0, 0]))

    # color = list(np.random.rand(3)) + [1]
    # for wpt_i, wpt in enumerate(ret_traj):
    #     wpt_trans = get_matrix_from_pose(wpt)
    #     abs_trans = base_trans @ wpt_trans
    #     abs_wpt = get_pose_from_matrix(abs_trans)
    #     wpt_id = p.createMultiBody(
    #         baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, 0.001), 
    #         baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, 0.001, rgbaColor=color), 
    #         basePosition=abs_wpt[:3]
    #     )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', '-ir', type=str, default='data')
    parser.add_argument('--input_dir', '-id', type=str, default='traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-residual-30/02.11.13.39/train')
    args = parser.parse_args()
    
    main(args)