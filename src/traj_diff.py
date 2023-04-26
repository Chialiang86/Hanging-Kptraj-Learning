import os, glob, json, argparse, imageio
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch, cv2

from sklearn.neighbors import NearestNeighbors
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

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

def main(args):
    input_dataset = args.input_dataset
    traj_num = args.traj_num
    traj_dim = args.traj_dim
    wpt_dim = args.wpt_dim
    
    assert os.path.exists(input_dataset), f'{input_dataset} not exists'

    traj_name = input_dataset.split('/')[-3]
    traj_id = input_dataset.split('/')[-2]
    outdir = f'visualization/trajectory_pca/{traj_name}-{traj_id}-{traj_num}-{traj_dim}-{wpt_dim}'
    os.makedirs(outdir, exist_ok=True)

    hook_shape_dirs = glob.glob(f'{input_dataset}/*')

    hook_difficulties = []
    hook_trajectories = []
    hook_trajectories_raw = []
    hook_pcds = []
    hook_names = []
    for hook_shape_dir in hook_shape_dirs:
        hook_name = hook_shape_dir.split('/')[-1]
        hook_names.append(hook_name)

        difficulty = 'easy' if 'easy' in hook_name else \
                    'normal' if 'normal' in hook_name else \
                    'hard' if 'hard' in hook_name else \
                    'devil'

        hook_pcd_path = glob.glob(f'{hook_shape_dir}/*.npy')[0]

        hook_traj_paths = glob.glob(f'{hook_shape_dir}/*.json')
        hook_traj_paths.sort(key=lambda x : int(x.split('/')[-1].split('-')[-1].split('.')[0])) # sort by trajectory id : [parent_dir]/traj-8.json => 8
        hook_traj_paths = hook_traj_paths[:traj_num]

        for hook_traj_path in hook_traj_paths:
            
            traj = json.load(open(hook_traj_path, 'r'))['trajectory']
            
            if wpt_dim == 3:
                traj_3d = np.asarray(traj)[:traj_dim, :3]
                hook_trajectories_raw.append(np.copy(traj_3d))
                traj_3d[:, :3] -= traj_3d[0, :3]
                hook_trajectories.append(traj_3d)

            if wpt_dim == 6:
                traj6d = np.asarray(traj)[:traj_dim]
                hook_trajectories_raw.append(np.copy(traj6d))
                traj6d[:, :3] -= traj6d[0, :3]
                hook_trajectories.append(traj6d)

            if wpt_dim == 9:
                traj_9d = np.zeros((traj_dim, 9))
                traj = np.asarray(traj)[:traj_dim]
                traj_rot = R.from_rotvec(traj[:, 3:]).as_matrix().reshape(-1, 9)[:, :6]
                traj_9d[:, 3:] = traj_rot
                hook_trajectories_raw.append(np.copy(traj_9d))
                traj_9d[:, :3] -= traj[0, :3]
                hook_trajectories.append(traj_9d)
            
            hook_pcds.append(hook_pcd_path)
            hook_difficulties.append(difficulty)
    
    hook_trajectories = np.asarray(hook_trajectories)
    hook_trajectories_reshape = np.reshape(hook_trajectories, (hook_trajectories.shape[0], -1))

    X_embedded = PCA(n_components=2).fit_transform(hook_trajectories_reshape)
    X_embedded = np.hstack((X_embedded, np.zeros((X_embedded.shape[0], 1)))).astype(np.float32)

    # centroid_points = torch.from_numpy(X_embedded).unsqueeze(0).to('cuda').contiguous()
    # input_pcid = furthest_point_sample(centroid_points, 10).cpu().long().reshape(-1)  # BN
    # selected_points = X_embedded[input_pcid]

    # num_nn = 5
    # nn = NearestNeighbors(n_neighbors=num_nn, algorithm='ball_tree').fit(X_embedded)
    # distances, indices = nn.kneighbors(selected_points)

    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot()
    # ax.scatter(X_embedded[:, 0],   X_embedded[:, 1], c=[[0.8, 0.8, 0.8]], s=10)

    # colors = cv2.applyColorMap((255 * np.linspace(0, 1, num_nn)).astype(np.uint8), colormap=cv2.COLORMAP_JET).squeeze() / 255.0
    # for cls_id, (indice, color) in enumerate(zip(indices, colors)):

    #     for ind in indice:
    #         pcd_path = hook_pcds[ind]
    #         pcd = np.load(pcd_path)
    #         pcd_points = pcd[:, :3]
    #         pcd_afford = pcd[:, 4]
    #         pcd_colors = cv2.applyColorMap((255 * pcd_afford).astype(np.uint8), colormap=cv2.COLORMAP_JET).squeeze()
    #         point_cloud = o3d.geometry.PointCloud()
    #         point_cloud.points = o3d.utility.Vector3dVector(pcd_points)
    #         point_cloud.colors = o3d.utility.Vector3dVector(pcd_colors / 255)
    #         r = point_cloud.get_rotation_matrix_from_xyz((0, np.pi / 2, 0)) # (rx, ry, rz) = (right, up, inner)
    #         point_cloud.rotate(r, center=(0, 0, 0))

    #         wpts = hook_trajectories_raw[ind]
    #         geometries = []
    #         for wpt_raw in wpts[:20]:
    #             wpt = wpt_raw[:3]
    #             coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.003)
    #             coor.translate(wpt.reshape((3, 1)))
    #             coor.rotate(r, center=(0, 0, 0))
    #             geometries.append(coor)
    #         geometries.append(point_cloud)

    #         img = capture_from_viewer(geometries)
    #         save_path = f'{outdir}/cls-{cls_id}-{ind}.png'
    #         imageio.imsave(save_path, img)
    #         print(f'{save_path} saved')

    #     ax.scatter(X_embedded[indice, 0],   X_embedded[indice, 1], c=color.reshape(1, -1), s=10)

    # out_path = f'{outdir}/fps-clustering.png'
    # plt.savefig(out_path)
    # plt.clf()
    
    hook_difficulties = np.asarray(hook_difficulties)
    easy_ind = np.where(hook_difficulties == 'easy')[0]
    normal_ind = np.where(hook_difficulties == 'normal')[0]
    hard_ind = np.where(hook_difficulties == 'hard')[0]
    devil_ind = np.where(hook_difficulties == 'devil')[0]
    max_rgb = 255.0

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    ax.scatter(X_embedded[easy_ind, 0],   X_embedded[easy_ind, 1],   c=[[123/max_rgb, 234/max_rgb ,0/max_rgb]],   label='easy',   s=10)
    ax.scatter(X_embedded[normal_ind, 0], X_embedded[normal_ind, 1], c=[[123/max_rgb, 0/max_rgb   ,234/max_rgb]], label='normal', s=10)
    ax.scatter(X_embedded[hard_ind, 0],   X_embedded[hard_ind, 1],   c=[[234/max_rgb, 123/max_rgb ,0/max_rgb]],   label='hard',   s=10)
    ax.scatter(X_embedded[devil_ind, 0],  X_embedded[devil_ind, 1],  c=[[234/max_rgb, 0/max_rgb   ,123/max_rgb]], label='devil',  s=10)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_embedded[easy_ind, 0],   X_embedded[easy_ind, 1],   X_embedded[easy_ind, 2], c=[[123/max_rgb, 234/max_rgb ,0/max_rgb]],   label='easy',   s=10)
    # ax.scatter(X_embedded[normal_ind, 0], X_embedded[normal_ind, 1], X_embedded[normal_ind, 2], c=[[123/max_rgb, 0/max_rgb   ,234/max_rgb]], label='normal', s=10)
    # ax.scatter(X_embedded[hard_ind, 0],   X_embedded[hard_ind, 1],   X_embedded[hard_ind, 2], c=[[234/max_rgb, 123/max_rgb ,0/max_rgb]],   label='hard',   s=10)
    # ax.scatter(X_embedded[devil_ind, 0],  X_embedded[devil_ind, 1],  X_embedded[devil_ind, 2], c=[[234/max_rgb, 0/max_rgb   ,123/max_rgb]], label='devil',  s=10)
    
    out_path = f'{outdir}/clustering.png'
    plt.legend()
    plt.savefig(out_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', '-id', type=str, default='../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train')
    parser.add_argument('--traj_num', '-tn', type=int, default=50)
    parser.add_argument('--traj_dim', '-td', type=int, default=40)
    parser.add_argument('--wpt_dim', '-wd', type=int, default=3)

    args = parser.parse_args()
    main(args)