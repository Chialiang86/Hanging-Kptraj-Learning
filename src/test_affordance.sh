# bin/sh

model_configs=(
    "affordance" "affordance_msg" 
)

iters=(
    '20' '100'
)

inference_directories=(
    # "../shapes/realworld_hook"
    "../data/traj_recon_affordance/hook_all-kptraj_all_one_0214-absolute-40/02.15.17.24/train"
    "../data/traj_recon_affordance/hook_all-kptraj_all_one_0214-absolute-40/02.15.17.24/test"
    "../data/traj_recon_affordance/hook_all-kptraj_all_one_0214-absolute-40/02.15.17.24/train"
    "../data/traj_recon_affordance/hook_all-kptraj_all_one_0214-absolute-40/02.15.17.24/test"
    # "../data/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-absolute-30/02.03.13.28/train"
    # "../data/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-absolute-30/02.03.13.28/test"
)

# element number should be the same as inference_directories
affordance_checkpoints=(

    # 20230220
    # "checkpoints/affordance_02.20.21.59/hook_all-kptraj_all_one_0214-absolute-40_02.15.17.24"
    # "checkpoints/affordance_02.20.21.59/hook_all-kptraj_all_one_0214-absolute-40_02.15.17.24"
    # "checkpoints/affordance_02.20.21.59/hook-kptraj_1104_origin_last2hook-absolute-30_02.03.13.28"
    # "checkpoints/affordance_02.20.21.59/hook-kptraj_1104_origin_last2hook-absolute-30_02.03.13.28"

    # 20230223
    "checkpoints/affordance_02.23.15.02/hook_all-kptraj_all_one_0214-absolute-40_02.15.17.24"
    "checkpoints/affordance_02.23.15.02/hook_all-kptraj_all_one_0214-absolute-40_02.15.17.24"
    # "checkpoints/affordance_02.23.15.02/hook-kptraj_1104_origin_last2hook-absolute-30_02.03.13.28"

    "checkpoints/affordance_msg_02.23.18.21/hook_all-kptraj_all_one_0214-absolute-40_02.15.17.24"
    "checkpoints/affordance_msg_02.23.18.21/hook_all-kptraj_all_one_0214-absolute-40_02.15.17.24"
    # "checkpoints/affordance_msg_02.23.18.21/hook-kptraj_1104_origin_last2hook-absolute-30_02.03.13.28"

)

# element number should be the same as inference_directories
num_of_points=(
    # 20230220
    # '1000' 
    # '1000' 
    # '3500'
    # '3500'

    # 20230223
    # '3500'
    '1000' 
    '1000' 
    '1000' 
    '1000' 
)

length=${#inference_directories[@]}

for model_config in "${model_configs[@]}"
do

    for iter in "${iters[@]}"
    do 
        if [[ $model_config == *"affordance"* ]]; then 
            
            for (( i=0; i<$length; i++ )) 
            do 
                python3 train_affordance.py -tm 'test' \
                                            --inference_dir ${inference_directories[$i]} \
                                            --weight_subpath "${num_of_points[$i]}_points-network_epoch-${iter}.pth" \
                                            --checkpoint_dir ${affordance_checkpoints[$i]} \
                                            --config "../config/${model_config}.yaml" \

            done 

        else 

            echo "Wrong model type : ${model_config}"

        fi 
    done 

done 
