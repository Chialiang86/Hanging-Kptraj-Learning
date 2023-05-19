import glob, os, json, shutil, sys
from tqdm import tqdm

val_hook_ids = [
                'Hook_hcu_104_devil', 'Hook_hcu_134_normal', 'Hook_hcu_138_hard', 'Hook_hcu_181_hard', 'Hook_hcu_190_easy', 
                'Hook_hcu_243_normal', 'Hook_hcu_279_devil', 'Hook_hcu_293_easy', 'Hook_hcu_296_normal', 'Hook_hcu_303_normal', 
                'Hook_hcu_306_normal', 'Hook_hcu_335_normal', 'Hook_hcu_359_easy', 'Hook_hcu_362_easy', 'Hook_hcu_364_normal', 
                'Hook_hcu_376_devil', 'Hook_hcu_380_hard', 'Hook_hcu_390_easy', 'Hook_hcu_3_hard', 'Hook_hcu_75_easy', 
                'Hook_hcu_89_devil', 'Hook_hs_105_hard', 'Hook_hs_117_hard', 'Hook_hs_154_hard', 'Hook_hs_156_hard', 
                'Hook_hs_190_easy', 'Hook_hs_216_easy', 'Hook_hs_229_normal', 'Hook_hs_275_devil', 'Hook_hs_293_normal', 
                'Hook_hs_314_easy', 'Hook_hs_317_easy', 'Hook_hs_339_devil', 'Hook_hs_363_devil', 'Hook_hs_370_easy', 
                'Hook_hs_393_devil', 'Hook_hs_42_hard', 'Hook_hs_70_easy', 'Hook_hs_94_hard', 'Hook_hs_95_devil', 
                'Hook_hsr_118_hard', 'Hook_hsr_123_normal', 'Hook_hsr_125_easy', 'Hook_hsr_13_devil', 'Hook_hsr_15_normal', 
                'Hook_hsr_218_devil', 'Hook_hsr_22_normal', 'Hook_hsr_263_hard', 'Hook_hsr_298_normal', 'Hook_hsr_304_hard', 
                'Hook_hsr_312_devil', 'Hook_hsr_321_devil', 'Hook_hsr_335_hard', 'Hook_hsr_371_devil', 'Hook_hsr_381_easy', 
                'Hook_hsr_391_hard', 'Hook_hsr_56_normal', 'Hook_hsr_5_normal', 'Hook_hsr_71_easy', 'Hook_omni_124_devil'
            ]

def main(src_dir, dst_dir):

    src_npy_paths = glob.glob(f'{src_dir}/*/affordance*.npy')
    
    for src_npy_path in tqdm(src_npy_paths):

        if 'fullview' in src_npy_path:
            continue

        hook_id = src_npy_path.split('/')[-2]
        hook_subpath = src_npy_path.split('/')[-1]

        if hook_id in val_hook_ids:
            target_npy_path = f'{dst_dir}/val/{hook_id}/{hook_subpath}'
            if os.path.exists(target_npy_path):
                # print(src_npy_path, '=>', target_npy_path)
                shutil.copy2(src_npy_path, target_npy_path)
            target_npy_path = f'{dst_dir}/val_deform/{hook_id}/{hook_subpath}'
            if os.path.exists(target_npy_path):
                # print(src_npy_path, '=>', target_npy_path)
                shutil.copy2(src_npy_path, target_npy_path)
        else :
            target_npy_path = f'{dst_dir}/train/{hook_id}/{hook_subpath}'
            if os.path.exists(target_npy_path):
                # print(src_npy_path, '=>', target_npy_path)
                shutil.copy2(src_npy_path, target_npy_path)


if __name__=="__main__":

    src_dir = 'shapes/hook_all_new'
    # dst_dir = 'dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview'
    # dst_dir = 'dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview'
    # dst_dir = 'dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview'
    # dst_dir = 'dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview'
    # dst_dir = 'dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview'
    dst_dir = 'dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview'
    
    if sys.argv == 3:
        src_dir = sys.argv[1]
        dst_dir = sys.argv[2]

    assert os.path.exists(src_dir), f'{src_dir} not exists'
    assert os.path.exists(dst_dir), f'{dst_dir} not exists'

    main(src_dir, dst_dir)