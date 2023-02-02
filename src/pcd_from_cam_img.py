import glob
import os
import time
import argparse
from  tqdm import tqdm
import pybullet as p
import numpy as np
import pybullet_data
import scipy.io as sio
import open3d as o3d
import xml.etree.ElementTree as ET

from PIL import Image
from scipy.spatial.transform import Rotation as R

def load_obj_urdf(urdf_path, pos=[0, 0, 0], rot=[0, 0, 0]):

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    center = np.array(
      [
        float(i) for i in root[0].find(
          "inertial"
        ).find(
          "origin"
        ).attrib['xyz'].split(' ')
      ]
    )
    scale = np.array(
      [
        float(i) for i in root[0].find(
          "visual"
        ).find(
          "geometry"
        ).find(
          "mesh"
        ).attrib["scale"].split(" ")
      ]
    )
    assert scale[0] == scale[1] and scale[0] == scale[2] and scale[1] == scale[2], f"scale is not uniformed : {scale}"
    obj_id = p.loadURDF(urdf_path, pos, p.getQuaternionFromEuler(rot))
    return obj_id, center, scale[0]



def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

def get_projmat_and_intrinsic(width, height, fx, fy, far, near):

  cx = width / 2
  cy = height / 2
  fov = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi

  project_matrix = p.computeProjectionMatrixFOV(
                      fov=fov,
                      aspect=1.0,
                      nearVal=near,
                      farVal=far
                    )
  
  intrinsic = np.array([
                [ fx, 0.0,  cx],
                [0.0,  fy,  cy],
                [0.0, 0.0, 1.0],
              ])
  
  return project_matrix, intrinsic

def get_viewmat_and_extrinsic(cameraEyePosition, cameraTargetPosition, cameraUpVector):


    view_matrix = p.computeViewMatrix(
                    cameraEyePosition=cameraEyePosition,
                    cameraTargetPosition=cameraTargetPosition,
                    cameraUpVector=cameraUpVector
                  )

    # rotation vector extrinsic
    z = np.asarray(cameraTargetPosition) - np.asarray(cameraEyePosition)
    norm = np.linalg.norm(z, ord=2)
    assert norm > 0, f'cameraTargetPosition and cameraEyePosition is at same location'
    z /= norm
   
    y = -np.asarray(cameraUpVector)
    y -= (np.dot(z, y)) * z
    norm = np.linalg.norm(y, ord=2)
    assert norm > 0, f'cameraUpVector is parallel to z axis'
    y /= norm
    
    x = cross(y, z)

    # extrinsic
    extrinsic = np.identity(4)
    extrinsic[:3, 0] = x
    extrinsic[:3, 1] = y
    extrinsic[:3, 2] = z
    extrinsic[:3, 3] = np.asarray(cameraEyePosition)

    return view_matrix, extrinsic

def get_matrix_from_pos_rot(pos : list or tuple, rot : list or tuple):
    assert (len(pos) == 3 and len(rot) == 4) or (len(pos) == 3 and len(rot) == 3)
    pos_m = np.asarray(pos)
    if len(rot) == 3:
        rot_m = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rot))).reshape((3, 3))
    elif len(rot) == 4:
        rot_m = np.asarray(p.getMatrixFromQuaternion(rot)).reshape((3, 3))
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m
    return ret_m

def create_rgbd(rgb, depth, intr, extr, dscale, depth_threshold=2.0):
    assert rgb.shape[:2] == depth.shape
    (h, w) = depth.shape
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]

    ix, iy =  np.meshgrid(range(w), range(h))

    x_ratio = (ix.ravel() - cx) / fx
    y_ratio = (iy.ravel() - cy) / fy

    z = depth.ravel() / dscale
    x = z * x_ratio
    y = z * y_ratio

    cond = np.where(z < depth_threshold)
    print(f'There are {len(cond[0])} points in point clouds')
    x = x[cond]
    y = y[cond]
    z = z[cond]

    points = np.vstack((x, y, z)).T
    center = np.mean(points, axis=0)
    colors = np.reshape(rgb,(depth.shape[0] * depth.shape[1], 3))
    colors = np.array([colors[:,2], colors[:,1], colors[:,0]]).T / 255.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    pcd.transform(extr)

    return pcd


def pcd_adjustment(pcd : o3d.geometry.PointCloud, pitch : float, yaw : float):
  deg2euler = 1 / 180 * np.pi

  if np.abs(pitch) >= 90:
    extrinsic_rot = R.from_rotvec([-(pitch + 90) * deg2euler, 0, 0]).as_matrix()
    extrinsic = np.identity(4)
    extrinsic[:3, :3] = extrinsic_rot
    pcd.transform(extrinsic)
    extrinsic_rot = R.from_rotvec([0, np.pi, 0]).as_matrix()
    extrinsic = np.identity(4)
    extrinsic[:3, :3] = extrinsic_rot
    pcd.transform(extrinsic)
  else :
    extrinsic_rot = R.from_rotvec([(pitch - 90) * deg2euler, 0, 0]).as_matrix()
    extrinsic = np.identity(4)
    extrinsic[:3, :3] = extrinsic_rot
    pcd.transform(extrinsic)

  extrinsic_rot = R.from_rotvec([0, 0, yaw * deg2euler]).as_matrix()
  extrinsic = np.identity(4)
  extrinsic[:3, :3] = extrinsic_rot
  pcd.transform(extrinsic)

  return pcd

def reset_camera(yaw : float, pitch : float, cameraDistance : float):
  p.resetDebugVisualizerCamera(
      cameraDistance=cameraDistance,
      cameraYaw=yaw,
      cameraPitch=pitch,
      cameraTargetPosition=[0.0, 0.0, 0.0]
  )

def adjust_normals(pcd : o3d.geometry.PointCloud()):
  normals = np.asarray(pcd.normals)
  points = np.asarray(pcd.points)
  mass_center = np.mean(points, axis=0)

  pos_relative_to_center = points - mass_center
  inner_product_with_normals = np.sum(pos_relative_to_center * normals, axis=1)
  print(inner_product_with_normals.shape)

  cond = np.where(inner_product_with_normals < 0)
  normals[cond] *= -1

def main(args):

  input_dir = args.input_dir
  file_token = args.file_token
  if not os.path.exists(input_dir):
    print(f'{input_dir} not exists')
    return 
  if not os.path.exists(f'{input_dir}/report'):
    os.mkdir(f'{input_dir}/report')

  p.connect(p.GUI)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())

  p.resetDebugVisualizerCamera(
      cameraDistance=0.3,
      cameraYaw=60,
      cameraPitch=-30,
      cameraTargetPosition=[0.0, 0.0, 0.0]
  )
  p.resetSimulation()
  p.setPhysicsEngineParameter(numSolverIterations=150)
  sim_timestep = 1.0 / 240
  p.setTimeStep(sim_timestep)
  p.setGravity(0, 0, -9.8)

  # # config camera params : https://www.intelrealsense.com/depth-camera-d435/
  # # D435 serial_num=141722071222, fx=605, fy=605, cx=323, cy=240
  # intrinsic matrix
  depth_threshold = 2.0
  width, height = 512, 512
  fx = fy = 605
  fov = 2 * np.arctan(height / fy) * 180.0 / np.pi
  far = 1000.
  near = 0.01
  cx = 0.5 * width
  cy = 0.5 * height
  intrinsic = np.array([
    [fx, 0., cx],
    [0., fy, cy],
    [0., 0., 1.]
  ])
  projection_matrix, intrinsic = get_projmat_and_intrinsic(width, height, fx, fy, far, near)

  # extrinsic matrix for hanging pose
  cameraEyePosition = [-0.24, 0.16, 0.1]
  cameraTargetPosition = [0.0, 0.0, 0.0]
  cameraUpVector =  [0.0, 0.0, 1.0]
  pcd_view_matrix, pcd_extrinsic = get_viewmat_and_extrinsic(cameraEyePosition, cameraTargetPosition, cameraUpVector)

  # cameraEyePosition = [-0.12, 0.08, 0.0]
  # cameraTargetPosition = [0.0, 0.0, 0.0]
  # cameraUpVector =  [0.0, 0.0, 1.0]
  cameraEyePosition = [0.0, 0.36, -0.24]
  cameraTargetPosition = [0.0, 0.0, 0.0]
  cameraUpVector =  [1.0, 0.0, 0.0]
  rgb_view_matrix, rgb_extrinsic = get_viewmat_and_extrinsic(cameraEyePosition, cameraTargetPosition, cameraUpVector)

  origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
  # urdf_files = glob.glob(f'{input_dir}/*/{file_token}.urdf')
  urdf_files = glob.glob(f'{input_dir}/Hook_*/base.urdf')
  urdf_files.sort()
  for urdf_file in tqdm(urdf_files):
    print(urdf_file)

    # if len(urdf_file.split('#')) > 1:
    #   continue
    # if 'scissor' not in urdf_file:
    #   continue
      # serial_num = int(urdf_file.split('/')[-2].split('#')[1])
      # if serial_num < 0:
      #   continue
    print(f'processing {urdf_file}')
    obj_id, center, scale = load_obj_urdf(urdf_file, [0, 0, 0])
    # obj_id = p.loadURDF(urdf_file)

    # reset_camera(yaw, pitch, cameraDistance)

    time.sleep(0.01)
    img = p.getCameraImage(width, height, viewMatrix=pcd_view_matrix, projectionMatrix=projection_matrix)
    rgb_buffer = np.reshape(img[2], (height, width, 4))[:,:,:3]
    depth_buffer = np.reshape(img[3], [height, width])

    # get real depth
    depth_buffer = far * near / (far - (far - near) * depth_buffer)

    # adjustment
    pcd = create_rgbd(rgb_buffer, depth_buffer, intrinsic, pcd_extrinsic, dscale=1, depth_threshold=depth_threshold)

    mesh_file = os.path.splitext(urdf_file)[0] + '.obj'
    pcd_ori = o3d.io.read_triangle_mesh(mesh_file)
    pcd_ori.scale(scale, [0., 0., 0.,])
    
    # save ply
    # o3d.visualization.draw_geometries([origin, pcd_ori, pcd], point_show_normal=False)
    output_ply_path = os.path.splitext(urdf_file)[0] + '.ply'
    o3d.io.write_point_cloud(output_ply_path, pcd)

    img = p.getCameraImage(height, height, viewMatrix=rgb_view_matrix, projectionMatrix=projection_matrix)
    rgb_buffer = np.reshape(img[2], (height, height, 4))[:,:,:3]

    # # save image
    # # output_jpg_path = os.path.splitext(urdf_file)[0] + '.jpg'
    # sub_dir = urdf_file.split('/')[-2]
    # # output_jpg_path = f'{input_dir}/All_img/{sub_dir}.jpg'
    # output_jpg_path = f'{input_dir}/report/{sub_dir}.jpg'
    # pil_img = Image.fromarray(rgb_buffer)
    # pil_img.save(output_jpg_path)

    # print(f'{output_ply_path} and {output_jpg_path} saved')
    p.removeBody(obj_id)

start_msg = \
'''
======================================================================================
this script will create point cloud of the given objects using virtual RGBD camera 
and wirte them to the object folder [root]/[obj_name]/base.urdf

dependency :
- [root]/[obj_name]/base.urdf
======================================================================================
'''

print(start_msg)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-dir', '-id', type=str, default='../shapes/hook')
  parser.add_argument('--file-token', '-ft', type=str, default='base')
  args = parser.parse_args()

  main(args)