# bin/sh


iters=(
    '20000'
)

model_configs=(
    # "af" 
    "af_msg" 
)

# element number should be the same as model_configs
inference_directories=(

    # "../shapes/realworld_hook"
    # "../shapes/realworld_hook"
    # "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000/test"
    "../dataset/traj_recon_affordance/kptraj_all_new-absolute-40/03.05.12.50-1000/test"
)

# element number should be the same as model_configs
affordance_checkpoints=(

    "checkpoints/af_msg-03.05.13.45/hook_all_new-kptraj_all_new-absolute-40_03.05.12.50-1000"

)

# element number should be the same as model_configs
num_of_points=(
    '1000' 
)

length=${#model_configs[@]}

for (( i=0; i<$length; i++ )) 
do 
    for iter in "${iters[@]}"
    do 

        python3 train_affordance.py -tm 'test' \
                                    --inference_dir ${inference_directories[$i]} \
                                    --checkpoint_dir ${affordance_checkpoints[$i]} \
                                    --config "../config/${model_configs[$i]}.yaml" \
                                    --weight_subpath "${num_of_points[$i]}_points-network_epoch-${iter}.pth" \
                                    -v 
                                    # --evaluate
 
    done 
done 
