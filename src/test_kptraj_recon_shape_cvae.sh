# bin/sh

# model_configs=("traj_recon_ae" "traj_recon_vae")
model_configs=(
    "traj_recon_shape_cvae_kl_large" 
)

traj_recon_shape_checkpoints=(
    "checkpoints/traj_recon_shape_cvae_kl_large_01.17.13.56/hook-kptraj_1104-absolute-30_01.16.13.41"
    "checkpoints/traj_recon_shape_cvae_kl_large_01.17.13.56/hook-kptraj_1104-residual-30_01.16.13.42"
    "checkpoints/traj_recon_shape_cvae_kl_large_01.17.13.56/hook-kptraj_1104_aug-absolute-30_01.16.13.56"
    "checkpoints/traj_recon_shape_cvae_kl_large_01.17.13.56/hook-kptraj_1104_aug-residual-30_01.16.13.57"

    # "checkpoints/traj_recon_shape_cvae_kl_small_01.17.13.56/hook-kptraj_1104_aug-absolute-30_01.16.13.56"
    # "checkpoints/traj_recon_shape_cvae_kl_small_01.17.13.56/hook-kptraj_1104_aug-residual-30_01.16.13.57"
    # "checkpoints/traj_recon_shape_cvae_kl_small_01.17.13.56/hook-kptraj_1104-absolute-30_01.16.13.41"
    # "checkpoints/traj_recon_shape_cvae_kl_small_01.17.13.56/hook-kptraj_1104-residual-30_01.16.13.42"

    # "checkpoints/traj_recon_shape_cvae_kl_annealing_01.17.13.56/hook-kptraj_1104_aug-absolute-30_01.16.13.56"
    # "checkpoints/traj_recon_shape_cvae_kl_annealing_01.17.13.56/hook-kptraj_1104_aug-residual-30_01.16.13.57"
    # "checkpoints/traj_recon_shape_cvae_kl_annealing_01.17.13.56/hook-kptraj_1104-absolute-30_01.16.13.41"
    # "checkpoints/traj_recon_shape_cvae_kl_annealing_01.17.13.56/hook-kptraj_1104-residual-30_01.16.13.42"
)

for model_config in "${model_configs[@]}"
do

    if [[ $model_config == *"_cvae"* ]]; then 
        
        for traj_recon_shape_checkpoint in "${traj_recon_shape_checkpoints[@]}"
        do 

            python3 train_kptraj_recon_shape_cvae.py -tm 'test' --weight_subpath "3000_points-network_epoch-2.pth" --checkpoint_dir $traj_recon_shape_checkpoint --config "../config/${model_config}.yaml"

        done 

    else 

        echo "Wrong model type : ${model_config}"

    fi 

done 
