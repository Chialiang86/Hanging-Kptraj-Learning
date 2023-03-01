# bin/sh

obj_shape_root="../shapes/inference_objs"
# hook_shape_root="../shapes/hook_all_new_0"
hook_shape_root="../shapes/hook_all_new_devil"

model_configs=(

    # "traj_recon_affordance_cvae_kl_large_30" 
    # "traj_recon_affordance_cvae_kl_large_30" 

    # "traj_recon_affordance_cvae_kl_small_30" 
    # "traj_recon_affordance_cvae_kl_small_30" 

    # "traj_recon_affordance_cvae_kl_large" 
    # "traj_recon_affordance_cvae_kl_large" 

    # "traj_recon_affordance_cvae_kl_small" 
    # "traj_recon_affordance_cvae_kl_small" 

    "traj_recon_affordance_nofp_cvae_kl_large" 
    "traj_recon_affordance_nofp_cvae_kl_large" 

    "traj_recon_affordance_nofp_4d_cvae_kl_large" 
    "traj_recon_affordance_nofp_4d_cvae_kl_large" 
)

dataset_dirs=(

    "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000"
    "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-residual-40/02.27.10.32-1000"

    "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000"
    "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-residual-40/02.27.10.32-1000"
)

inference_dirs=(

    # "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference"
    # "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-residual-40/02.27.10.32-1000/inference"

    # "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference"
    # "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-residual-40/02.27.10.32-1000/inference"

    "../shapes/hook_all_new_devil"
    "../shapes/hook_all_new_devil"

    "../shapes/hook_all_new_devil"
    "../shapes/hook_all_new_devil"
)

traj_recon_shape_checkpoints=(

    # "checkpoints/traj_recon_affordance_cvae_kl_large_02.11.13.44/hook-kptraj_1104_origin_last2hook-absolute-30_02.03.13.28"
    # "checkpoints/traj_recon_affordance_cvae_kl_large_02.11.13.44/hook-kptraj_1104_origin_last2hook-residual-30_02.03.13.29"

    # "checkpoints/traj_recon_affordance_cvae_kl_large_02.28.19.50/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000" # 25000 the bset
    # "checkpoints/traj_recon_affordance_cvae_kl_large_02.28.19.50/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000" # 25000 the best
    
    # "checkpoints/traj_recon_affordance_cvae_kl_small_02.28.19.50/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000" # 30000 the best
    # "checkpoints/traj_recon_affordance_cvae_kl_small_02.28.19.50/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000" # 30000 the best
    
    "checkpoints/traj_recon_affordance_nofp_cvae_kl_large_02.28.19.50/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000" # 30000 the best
    "checkpoints/traj_recon_affordance_nofp_cvae_kl_large_02.28.19.50/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000" # 30000 the best
    
    "checkpoints/traj_recon_affordance_nofp_4d_cvae_kl_large_02.28.19.50/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000" # 25000 the bset
    "checkpoints/traj_recon_affordance_nofp_4d_cvae_kl_large_02.28.19.50/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000" # 25000 the best
    
)

num_of_points=(
    
    '1000' '1000' 
    '1000' '1000'
)

iters=(
    '25000' '30000' 
)

length=${#model_configs[@]}

for (( i=0; i<$length; i++ )) 
do

    for iter in "${iters[@]}"
    do 

        python3 train_kptraj_recon_affordance_cvae.py --training_mode 'test' \
                                                    --dataset_dir ${dataset_dirs[$i]} \
                                                    --inference_dir ${inference_dirs[$i]} \
                                                    --checkpoint_dir ${traj_recon_shape_checkpoints[$i]} \
                                                    --config "../config/${model_configs[$i]}.yaml" \
                                                    --weight_subpath "${num_of_points[$i]}_points-network_epoch-${iter}.pth" \
                                                    --obj_shape_root ${obj_shape_root} \
                                                    --hook_shape_root ${hook_shape_root} \
                                                    --visualize 

    done 

done 
