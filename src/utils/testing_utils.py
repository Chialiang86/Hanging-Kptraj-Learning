
import time
import numpy as np
import pybullet as p
from PIL import Image
from scipy.spatial.transform import Rotation as R

from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose, \
                               pose_6d_to_7d, pose_7d_to_6d, draw_coordinate

PENETRATION_THRESHOLD = 0.001

def augment_next_waypoint(waypoint : list or np.ndarray,
                                direction_vec : list or np.ndarray,
                                length : float,
                                aug_num : int=10,
                                # in degree
                                noise_pos : float=0.2,
                                # in degree
                                noise_rot : float=1) -> np.ndarray:

    assert len(waypoint) == 7 and len(direction_vec) == 3, \
        f'length of waypoint should be 7 and direction_vec should be 3 but got {len(waypoint)} and {len(direction_vec)}'
    
    base_pos    = np.asarray(waypoint[:3]) # + np.asarray([0.0, 0.0, 0.02]) for testing
    base_rotvec = R.from_quat(waypoint[3:]).as_rotvec()

    deg_to_rad = np.pi / 180.0

    pos_low_limit  = np.full((3,), -noise_pos * deg_to_rad)
    pos_high_limit = np.full((3,),  noise_pos * deg_to_rad)
    rot_low_limit  = np.full((3,), -noise_rot * deg_to_rad)
    rot_high_limit = np.full((3,),  noise_rot * deg_to_rad)

    step_direction_vec = np.zeros((3, aug_num))
    random_rot = R.from_rotvec(np.random.uniform(pos_low_limit, pos_high_limit, (aug_num, 3))).as_matrix()
    for i in range(aug_num):
        step_direction_vec[:, i] = (random_rot[i] @ direction_vec.reshape(3, 1)).reshape(3,)
    step_direction_vec = step_direction_vec.T
    # step_direction_vec = direction_vec + np.random.uniform(pos_low_limit, pos_high_limit, (aug_num, 3))
    # step_direction_vec /= np.linalg.norm(step_direction_vec, axis=1, ord=2)

    step_pos = base_pos + length * step_direction_vec
    step_rotvec = base_rotvec + np.random.uniform(rot_low_limit, rot_high_limit, (aug_num, 3)) \
                    if (base_rotvec <  np.pi - noise_rot * deg_to_rad).all() and \
                       (base_rotvec > -np.pi + noise_rot * deg_to_rad).all() \
                    else np.full((aug_num, 3),base_rotvec)
    step_quat = R.from_rotvec(step_rotvec).as_quat()
    step_pose = np.hstack((step_pos, step_quat))

    return step_pose

def waypoint_score(hook_id : int, obj_id : int):

    # thresh = 0.001
    p.performCollisionDetection()

    contact_points = p.getContactPoints(bodyA=hook_id, bodyB=obj_id)
    # closest_points = p.getClosestPoints(bodyA=hook_id, bodyB=obj_id, distance=thresh)
    # within_thresh = 1 if len(closest_points) > 0 else 0

    penetration = 0.0
    for contact_point in contact_points:
        # contact distance, positive for separation, negative for penetration
        contact_distance = contact_point[8] 
        penetration += contact_distance if contact_distance < 0 else 0.0
    
    if len(contact_points) > 0:
        penetration /= len(contact_points)
    # return penetration, within_thresh
    return penetration

def refine_waypoint_rotation(wpts : np.ndarray or list):

    assert wpts is not None and len(wpts) > 1, f'the trajectory only contains one waypoint or is None'

    rot_format = None
    if len(wpts[0]) == 6:
        rot_format = 'rotvec'
    elif len(wpts[0]) == 7:
        rot_format = 'quat'
    else:
        print('wrong waypoint format')
        exit(-1)

    # test direction
    next_pos = wpts[1][:3]
    tmp_pos = wpts[0][:3]
    tmp_dir = np.asarray(next_pos) - np.asarray(tmp_pos) 
    tmp_rot = wpts[0][3:]
    if rot_format == 'rotvec':
        tmp_rotmat = R.from_rotvec(tmp_rot).as_matrix()
    else :
        tmp_rotmat = R.from_quat(tmp_rot).as_matrix()
    tmp_rot_dir = (tmp_rotmat @ np.asarray([[1], [0], [0]])).T

    # no need to refine
    if np.dot(tmp_rot_dir, tmp_dir) > 0: 
        return wpts
    
    refine_mat = R.from_rotvec([0, 0, np.pi]).as_matrix()

    refined_wpts = []
    for i in range(len(wpts) - 1):
        tmp_pos = wpts[i][:3]
        tmp_rot = wpts[i][3:]
        if rot_format == 'rotvec':
            tmp_refined_rot = R.from_matrix(R.from_rotvec(tmp_rot).as_matrix() @ refine_mat).as_rotvec()
        else: 
            tmp_refined_rot = R.from_matrix(R.from_quat(tmp_rot).as_matrix() @ refine_mat).as_quat()
        tmp_refined_pose = list(tmp_pos) + list(tmp_refined_rot)
        refined_wpts.append(tmp_refined_pose)
        
    return refined_wpts

def trajectory_scoring(src_traj : list or np.ndarray, hook_id : int, obj_id : int, hook_pose : list or np.ndarray, obj_contact_pose : list or np.ndarray, visualize=False):

    if type(src_traj) == list:
        src_traj = np.asarray(src_traj)
    if type(obj_contact_pose) == list:
        obj_contact_pose = np.asarray(obj_contact_pose)

    assert obj_contact_pose.shape == (4, 4) or obj_contact_pose.shape == (7,), \
             f'the shape of obj_contact_pose must be (4, 4) or (7,), but got {obj_contact_pose.shape}'
    
    if obj_contact_pose.shape == (7,):
        obj_contact_trans = get_matrix_from_pose(obj_contact_pose)
    else :
        obj_contact_trans = obj_contact_pose
    
    hook_trans = get_matrix_from_pose(list(hook_pose))

    # score = PENETRATION_THRESHOLD
    penetration_cost = 0.0
    score = len(src_traj) * 0.5
    color = np.random.rand(1, 3)
    color = np.repeat(color, 3, axis=0)
    rgbs = []
    cam_info = p.getDebugVisualizerCamera()
    for i, waypoint in enumerate(src_traj):

        relative_trans = get_matrix_from_pose(waypoint)
        world_trans = hook_trans @ relative_trans
        obj_trans = world_trans @ np.linalg.inv(obj_contact_trans)
        obj_pose = get_pose_from_matrix(obj_trans, pose_size=7)
        p.resetBasePositionAndOrientation(obj_id, obj_pose[:3], obj_pose[3:])

        # draw_coordinate(world_trans, size=0.002)

        time.sleep(0.05)
        
        penetration = waypoint_score(hook_id=hook_id, obj_id=obj_id)
        score += penetration / PENETRATION_THRESHOLD
        penetration_cost += penetration

        if visualize:
            width = cam_info[0]
            height = cam_info[1]
            view_mat = cam_info[2]
            proj_mat = cam_info[3]
            img_info = p.getCameraImage(width, height, viewMatrix=view_mat, projectionMatrix=proj_mat)
            rgb = img_info[2]
            rgbs.append(Image.fromarray(rgb))

        # score -= waypoint_penetration
        # within_thresh_cnt += within_thresh

    # score = score - penetration_cost
    
    # score /= within_thresh_cnt if within_thresh_cnt != 0 else 1.0
    # score += PENETRATION_THRESHOLD # hyper param, < 0 : not good
    # ratio = score / PENETRATION_THRESHOLD

    return score, rgbs