import matplotlib.pyplot as plt
import os, time
import glob
import numpy as np
# easy: 23 / 30 | 6 / 12 
# normal: 22 / 12 | 19 / 28
# hard: 11 / 12 | 8 / 25
# devil: 7 / 5 | 9 / 14

# inference_hook_dir = "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/alltraj-1000/train", # for hook shapes
inference_hook_dirs = [
        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/alltraj-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/alltraj-1000/inference",
        "../dataset/traj_recon_affordance/kptraj_all_new-residual-10/alltraj-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new-residual-10/alltraj-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/onetraj-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/onetraj-1000/inference",
        "../dataset/traj_recon_affordance/kptraj_all_new-residual-10/onetraj-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new-residual-10/onetraj-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/alltraj-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/alltraj-1000/inference",
        "../dataset/traj_recon_affordance/kptraj_all_new-residual-10/alltraj-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new-residual-10/alltraj-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/onetraj-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/onetraj-1000/inference",
        "../dataset/traj_recon_affordance/kptraj_all_new-residual-10/onetraj-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new-residual-10/onetraj-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000/inference",

        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/train",
        "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference"
    ]

traj_recon_shape_checkpoints=[
        "checkpoints/traj_af_10-03.11.17.12-alltraj_10/kptraj_all_new-absolute-10-alltraj-1000",
        "checkpoints/traj_af_10-03.11.17.12-alltraj_10/kptraj_all_new-absolute-10-alltraj-1000",
        "checkpoints/traj_af_10-03.11.17.12-alltraj_10/kptraj_all_new-residual-10-alltraj-1000",
        "checkpoints/traj_af_10-03.11.17.12-alltraj_10/kptraj_all_new-residual-10-alltraj-1000",
        
        "checkpoints/traj_af_10-03.11.17.12-onetraj_10/kptraj_all_new-absolute-10-onetraj-1000",
        "checkpoints/traj_af_10-03.11.17.12-onetraj_10/kptraj_all_new-absolute-10-onetraj-1000",
        "checkpoints/traj_af_10-03.11.17.12-onetraj_10/kptraj_all_new-residual-10-onetraj-1000",
        "checkpoints/traj_af_10-03.11.17.12-onetraj_10/kptraj_all_new-residual-10-onetraj-1000",

        "checkpoints/traj_af_nn_dist_mr_10-03.11.17.16-alltraj_10/kptraj_all_new-absolute-10-alltraj-1000",
        "checkpoints/traj_af_nn_dist_mr_10-03.11.17.16-alltraj_10/kptraj_all_new-absolute-10-alltraj-1000",
        "checkpoints/traj_af_nn_dist_mr_10-03.11.17.16-alltraj_10/kptraj_all_new-residual-10-alltraj-1000",
        "checkpoints/traj_af_nn_dist_mr_10-03.11.17.16-alltraj_10/kptraj_all_new-residual-10-alltraj-1000",

        "checkpoints/traj_af_nn_dist_mr_10-03.11.17.17-onetraj_10/kptraj_all_new-absolute-10-onetraj-1000",
        "checkpoints/traj_af_nn_dist_mr_10-03.11.17.17-onetraj_10/kptraj_all_new-absolute-10-onetraj-1000",
        "checkpoints/traj_af_nn_dist_mr_10-03.11.17.17-onetraj_10/kptraj_all_new-residual-10-onetraj-1000",
        "checkpoints/traj_af_nn_dist_mr_10-03.11.17.17-onetraj_10/kptraj_all_new-residual-10-onetraj-1000",

        "checkpoints/traj_af-03.06.19.15-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj_af-03.06.19.15-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj_af-03.06.19.15-kl_l/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000",
        "checkpoints/traj_af-03.06.19.15-kl_l/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000",
        
        "checkpoints/traj_af_nn_dist-03.08.16.52-kl_l/kptraj_all_new_0-absolute-40-02.27.10.29-1000",
        "checkpoints/traj_af_nn_dist-03.08.16.52-kl_l/kptraj_all_new_0-absolute-40-02.27.10.29-1000",
        "checkpoints/traj_af_nn_dist-03.08.16.52-kl_l/kptraj_all_new_0-residual-40-02.27.10.32-1000",
        "checkpoints/traj_af_nn_dist-03.08.16.52-kl_l/kptraj_all_new_0-residual-40-02.27.10.32-1000",

        "checkpoints/traj_af_nn_dist_mr-03.08.16.52-kl_l/kptraj_all_new_0-absolute-40-02.27.10.29-1000",
        "checkpoints/traj_af_nn_dist_mr-03.08.16.52-kl_l/kptraj_all_new_0-absolute-40-02.27.10.29-1000",
        "checkpoints/traj_af_nn_dist_mr-03.08.16.52-kl_l/kptraj_all_new_0-residual-40-02.27.10.32-1000",
        "checkpoints/traj_af_nn_dist_mr-03.08.16.52-kl_l/kptraj_all_new_0-residual-40-02.27.10.32-1000",

        "checkpoints/traj_af_align-02.28.19.50-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj_af_align-02.28.19.50-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj_af_align-02.28.19.50-kl_l/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000",
        "checkpoints/traj_af_align-02.28.19.50-kl_l/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000",

        "checkpoints/traj_af_align4d-02.28.19.50-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj_af_align4d-02.28.19.50-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj_af_align4d-02.28.19.50-kl_l/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000",
        "checkpoints/traj_af_align4d-02.28.19.50-kl_l/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000",

        "checkpoints/traj_af_nn-03.04.18.36-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj_af_nn-03.04.18.36-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",

        "checkpoints/traj_af_nn_mr-03.05.13.57-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj_af_nn_mr-03.05.13.57-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",

        "checkpoints/traj3d_af-03.06.19.10-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj3d_af-03.06.19.10-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj3d_af-03.06.19.10-kl_l/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000",
        "checkpoints/traj3d_af-03.06.19.10-kl_l/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000",

        "checkpoints/traj3d_af_mr-03.05.13.58-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000",
        "checkpoints/traj3d_af_mr-03.05.13.58-kl_l/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000"
    ]

inference_root = 'inference/inference_trajs'

oneday_secs = 10000
file_sizes = []

f = open('out.csv', 'w')
csv_strings = []
for (inference_hook_dir, traj_recon_shape_checkpoint) in zip(inference_hook_dirs, traj_recon_shape_checkpoints):
    inference_hook_whole_paths = glob.glob(f'{inference_hook_dir}/*/affordance-0.npy')[:21]

    training_tag = traj_recon_shape_checkpoint.split('/')[-2]
    dataset_info = traj_recon_shape_checkpoint.split('/')[-1]
    dataset_subset = inference_hook_dir.split('/')[-1]
    inference_dir = f'{inference_root}/{training_tag}/{dataset_info}/{dataset_subset}'

    print(f'==========================================================================================================')
    print(f'processing: {traj_recon_shape_checkpoint} [{dataset_subset}]')

    id_difficult = {}
    for sid, inference_hook_whole_path in enumerate(inference_hook_whole_paths):

        id_difficult[str(sid)] = 'easy' if 'easy' in inference_hook_whole_path else \
                                'normal' if 'normal' in inference_hook_whole_path else \
                                'hard' if 'hard' in inference_hook_whole_path else \
                                'devil'

    out_dict_all = {
        'easy': 0,
        'normal': 0,
        'hard': 0,
        'devil': 0
    }
    out_dict_success = {
        'easy': 0,
        'normal': 0,
        'hard': 0,
        'devil': 0
    }

    target_paths_raw = glob.glob(f'{inference_dir}/*.gif')
    for target_path_raw in target_paths_raw:
        if 'success' in target_path_raw or 'failed' in target_path_raw:
            sid = target_path_raw.split('/')[-1].split('-')[-3]
            out_dict_all[id_difficult[sid]] += 1
            if 'success' in target_path_raw:
                out_dict_success[id_difficult[sid]] += 1

    csv_string = ''
    for d in out_dict_all.keys():
        csv_string += '{:00.01f}%,'.format(out_dict_success[d] / out_dict_all[d] * 100)
        # print('{}: {:00.01f}'.format(d, out_dict_success[d] / out_dict_all[d] * 100))
    csv_string = csv_string[:-1]
    csv_strings.append(csv_string)

    # target_paths = []
    # for target_path_raw in target_paths_raw:
    #     if ('failed' in target_path_raw or 'success' in target_path_raw):
    #         if (time.time() - os.path.getatime(target_path_raw) < oneday_secs):
    #             target_paths.append(target_path_raw)
    #             file_sizes.append(os.path.getsize(target_path_raw) / 1024)
    #             if os.path.getsize(target_path_raw) / 1024 < 400:
    #                 print(f'{target_path_raw}: [{os.path.getsize(target_path_raw) / 1024}]')
    
    # target_paths.sort()

    # print(f'{inference_dir}, [ {len(target_paths)} ]')

    # for target_path in target_paths:
    #     print(target_path)

for i in range(0, len(csv_strings), 2):
    train = csv_strings[i].split(',')
    inference = csv_strings[i + 1].split(',')
    assert len(train) == len(inference)

    s = ''
    for (t, i) in zip(train, inference):
        s += f'{t} / {i},'
    s = s[:-1]
    f.write(f'{s}\n')
f.close()

# plt.scatter(file_sizes, file_sizes)
# plt.show()
# print(np.min(np.asarray(file_sizes)))
print('process completed')