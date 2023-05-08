import os, glob, json, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pointnet2_ops.pointnet2_utils import furthest_point_sample

def main(args):

    input_dir = args.input_dir
    
    assert os.path.exists(input_dir), f'{input_dir} not exists'

    pcd_paths = glob.glob(f'{input_dir}/*/*.npy')
    
    interval = 1000
    part_interval = 500
    pn = 5000
    point_nums = {interval * n: 0 for n in range(20)}
    part_point_nums = {part_interval * n: 0 for n in range(20)}
    for pcd_path in tqdm(pcd_paths[3:]):
        pcd = np.load(pcd_path).astype(np.float32)

        # if pcd.shape[0] < pn:
        #     points = pcd[:,:3]
        #     mod_num = pn % pcd.shape[0] 
        #     # centroid_points = torch.from_numpy(points).unsqueeze(0).to('cuda').contiguous()
        #     # input_pcid = furthest_point_sample(centroid_points, 1000).long().reshape(-1)  # BN
        #     # centroid_points = centroid_points[0, input_pcid, :].squeeze()


        #     points_batch = torch.from_numpy(points).unsqueeze(0).to('cuda').contiguous()
        #     input_pcid = furthest_point_sample(centroid_points, 1000).long().reshape(-1)  # BN
        #     centroid_points = centroid_points[0, input_pcid, :].squeeze()
        #     repeat_num = int(pn // points_batch.shape[1])
        #     pcd = torch.cat([points_batch.repeat(1, repeat_num, 1), points_batch[:, input_pcid]], dim=1)
        #     for i in range(1000):
        #         print(pcd[0, -1000+i, :3])

        point_num = pcd.shape[0]
        point_nums[int((point_num // interval) * interval)] += 1

        part = (pcd[:, 3] + pcd[:, 4]) / 2
        part_high_response = np.where(part > 0.25)
        part_high_response_num = len(part_high_response[-1])
        part_point_nums[int((part_high_response_num // part_interval) * part_interval)] += 1

    x_keys = list(point_nums.keys())
    x_labels = [f'{x_keys[i]} ~ {x_keys[i+1]}' for i in range(len(x_keys) - 1)]
    x_labels.append(f'> {x_keys[-1]}')
    y_values = point_nums.values()

    plt.figure(figsize=(8, 12))
    plt.title('pcd point distribution')
    plt.xticks(range(len(x_labels)), x_labels, rotation = 90)
    plt.bar(range(len(y_values)), y_values)
    plt.savefig('pcd_point_distribution.png')

    x_keys = list(part_point_nums.keys())
    x_labels = [f'{x_keys[i]} ~ {x_keys[i+1]}' for i in range(len(x_keys) - 1)]
    x_labels.append(f'> {x_keys[-1]}')
    y_values = part_point_nums.values()

    plt.figure(figsize=(8, 12))
    plt.title('part point distribution')
    plt.xticks(range(len(x_labels)), x_labels, rotation = 90)
    plt.bar(range(len(y_values)), y_values)
    plt.savefig('part_distribution.png')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train')
    args = parser.parse_args()

    main(args)