# bin/sh

model_configs=(
    "traj_recon_ae"
)
traj_recon_checkpoints=(
    "checkpoints/traj_recon_ae_01.17.13.34/hook-kptraj_1104_aug-absolute-30_01.16.13.53"
    "checkpoints/traj_recon_ae_01.17.13.34/hook-kptraj_1104_aug-residual-30_01.16.13.55"
    "checkpoints/traj_recon_ae_01.17.13.34/hook-kptraj_1104-absolute-30_01.16.13.41"
    "checkpoints/traj_recon_ae_01.17.13.34/hook-kptraj_1104-residual-30_01.16.13.41"
)

for model_config in "${model_configs[@]}"
do

    if [[ $model_config == *"_ae"* ]]; then 

        for traj_recon_checkpoint in "${traj_recon_checkpoints[@]}"
        do 

            python3 train_kptraj_recon_ae.py -tm 'test' --weight_subpath "network_epoch-2.pth" --checkpoint_dir $traj_recon_checkpoint --config "../config/${model_config}.yaml" 

        done 

    else 

        echo "Wrong model type : ${model_config}"

    fi 

done 
