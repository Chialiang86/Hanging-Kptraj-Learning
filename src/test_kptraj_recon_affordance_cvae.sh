# bin/sh

model_configs=(
    # "traj_recon_affordance_cvae_kl_annealing"
    "traj_recon_affordance_cvae_kl_large" 
    # "traj_recon_affordance_cvae_kl_small" 
)

iters=(
    # '40' '60' '80' 
    '100' 
    # '120' '140' '160' '180' 
    '200' 
)

traj_recon_shape_checkpoints=(
    # "checkpoints/traj_recon_affordance_cvae_kl_annealing_02.11.13.44/hook-kptraj_1104_origin_last2hook_aug-absolute-30_02.11.13.38"
    # "checkpoints/traj_recon_affordance_cvae_kl_annealing_02.11.13.44/hook-kptraj_1104_origin_last2hook_aug-residual-30_02.11.13.39"
    # "checkpoints/traj_recon_affordance_cvae_kl_annealing_02.11.13.44/hook-kptraj_1104_origin_last2hook-absolute-30_02.03.13.28"
    # "checkpoints/traj_recon_affordance_cvae_kl_annealing_02.11.13.44/hook-kptraj_1104_origin_last2hook-residual-30_02.03.13.29"
    
    # "checkpoints/traj_recon_affordance_cvae_kl_large_02.11.13.44/hook-kptraj_1104_origin_last2hook_aug-absolute-30_02.11.13.38"
    # "checkpoints/traj_recon_affordance_cvae_kl_large_02.11.13.44/hook-kptraj_1104_origin_last2hook_aug-residual-30_02.11.13.39"
    # "checkpoints/traj_recon_affordance_cvae_kl_large_02.11.13.44/hook-kptraj_1104_origin_last2hook-absolute-30_02.03.13.28"
    # "checkpoints/traj_recon_affordance_cvae_kl_large_02.11.13.44/hook-kptraj_1104_origin_last2hook-residual-30_02.03.13.29"
    
    # "checkpoints/traj_recon_affordance_cvae_kl_small_02.11.13.44/hook-kptraj_1104_origin_last2hook_aug-absolute-30_02.11.13.38"
    # "checkpoints/traj_recon_affordance_cvae_kl_small_02.11.13.44/hook-kptraj_1104_origin_last2hook_aug-residual-30_02.11.13.39"
    # "checkpoints/traj_recon_affordance_cvae_kl_small_02.11.13.44/hook-kptraj_1104_origin_last2hook-absolute-30_02.03.13.28"
    # "checkpoints/traj_recon_affordance_cvae_kl_small_02.11.13.44/hook-kptraj_1104_origin_last2hook-residual-30_02.03.13.29"

    "checkpoints/traj_recon_affordance_cvae_kl_large_02.20.18.38/hook-kptraj_1104_origin_last2hook_aug-absolute-30_02.11.13.38"
    "checkpoints/traj_recon_affordance_cvae_kl_large_02.20.18.38/hook-kptraj_1104_origin_last2hook_aug-residual-30_02.11.13.39"
    "checkpoints/traj_recon_affordance_cvae_kl_large_02.20.18.38/hook-kptraj_1104_origin_last2hook-absolute-30_02.03.13.28"
    "checkpoints/traj_recon_affordance_cvae_kl_large_02.20.18.38/hook-kptraj_1104_origin_last2hook-residual-30_02.03.13.29"
)

num_of_points=(
    # '2000' '2000' '3500' '4000'
    '2000' '2000' '3500' '4000'
    # '2000' '2000' '3500' '4000'
)

length=${#traj_recon_shape_checkpoints[@]}

for model_config in "${model_configs[@]}"
do

    for iter in "${iters[@]}"
    do 
        if [[ $model_config == *"_cvae"* ]]; then 
            
            for (( i=0; i<$length; i++ )) 
            do 
                python3 train_kptraj_recon_affordance_cvae.py -tm 'test' --weight_subpath "${num_of_points[$i]}_points-network_epoch-${iter}.pth" --checkpoint_dir ${traj_recon_shape_checkpoints[$i]} --config "../config/${model_config}.yaml"
            done 

        else 

            echo "Wrong model type : ${model_config}"

        fi 
    done 


done 
