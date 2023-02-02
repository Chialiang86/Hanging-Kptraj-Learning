import os, json, glob, sys
import numpy as np
import open3d as o3d
import xml.etree.ElementTree as ET

from utils.bullet_utils import get_pose_from_matrix, get_matrix_from_pose, pose_7d_to_6d

def parse_obj_center_and_scale(urdf_path, pos=[0, 0, 0], rot=[0, 0, 0]):

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
    )[0]
    return center, scale

def main(obj_dir : str):

    obj_name = obj_dir.split('/')[-1]
    pcd_path = f'{obj_dir}/base.ply'
    mesh_path = f'{obj_dir}/base.obj'
    urdf_path = f'{obj_dir}/base.urdf'
    # traj_path = f'../raw/keypoint_trajectory_1104/{obj_name}.json'

    # traj_json = None
    # with open(traj_path, 'r') as f:
    #     traj_json = json.load(f)
    # trajs = traj_json['trajectory'] # first trajectory

    assert os.path.exists(pcd_path), f'{pcd_path} not exists'
    assert os.path.exists(urdf_path), f'{urdf_path} not exists'

    center, scale = parse_obj_center_and_scale(urdf_path)

    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    pcd = o3d.io.read_point_cloud(pcd_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.scale(scale, np.zeros(3))

    geometries = [pcd, mesh]
    # for traj in trajs:
    #     for wid, waypoint in enumerate(traj):
    #         coor = o3d.geometry.TriangleMesh.create_coordinate_frame(0.002)
    #         trans = get_matrix_from_pose(waypoint)
    #         coor.transform(trans)
    #         geometries.append(coor)

    o3d.visualization.draw_geometries(geometries)

    return 

if __name__=="__main__":

    obj_dir = '../shapes/hook/Hook1'
    if len(sys.argv) > 1:
        obj_dir = sys.argv[1]

    main(obj_dir)