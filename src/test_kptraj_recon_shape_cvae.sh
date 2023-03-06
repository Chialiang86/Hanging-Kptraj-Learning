# bin/sh


obj_shape_root="../shapes/inference_objs"
hook_shape_root="../shapes/hook_all_new_0"
# hook_shape_root="../shapes/hook_all_new_devil"

model_configs=(
    "traj_recon_shape_cvae_kl_large" 
    "traj_recon_shape_cvae_kl_large" 
)

dataset_dirs=(

    "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000"
    "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-residual-40/02.27.10.32-1000"

)

traj_recon_shape_checkpoints=(

    "checkpoints/traj_recon_shape_cvae_kl_large_03.01.20.56/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000/"
    "checkpoints/traj_recon_shape_cvae_kl_large_03.01.20.56/hook_all_new_0-kptraj_all_new_0-residual-40_02.27.10.32-1000/"
    
)

inference_dirs=(

    "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000/inference"
    "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-residual-40/02.27.10.32-1000/inference"

    # "../shapes/hook_all_new_devil"
    # "../shapes/hook_all_new_devil"
)

num_of_points=(
    '1000' '1000' 
)

iters=(
    '30000' 
)


length=${#model_configs[@]}

for (( i=0; i<$length; i++ )) 
do

    for iter in "${iters[@]}"
    do 

        python3 train_kptraj_recon_shape_cvae.py --training_mode 'test' \
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
