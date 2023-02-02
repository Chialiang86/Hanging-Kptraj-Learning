# bin/sh

model_configs=(
    "traj_recon_ae"
)
traj_recon_datasets=(
    "../data/traj_recon/hook-kptraj_1104_aug-absolute-30/01.16.13.53"
    "../data/traj_recon/hook-kptraj_1104_aug-residual-30/01.16.13.55"
    "../data/traj_recon/hook-kptraj_1104-absolute-30/01.16.13.41"
    "../data/traj_recon/hook-kptraj_1104-residual-30/01.16.13.41"
)
# traj_recon_datasets=(
#     "../data/traj_recon/hook-keypoint_trajectory_1104_aug-absolute-30-2023.01.05.20.33"
#     "../data/traj_recon/hook-keypoint_trajectory_1104_aug-residual-30-2023.01.11.16.07"
#     "../data/traj_recon/hook-keypoint_trajectory_1104-absolute-30-2023.01.05.21.12"
#     "../data/traj_recon/hook-keypoint_trajectory_1104-residual-30-2023.01.11.16.03"
# )

time_stamp=$(date +%m.%d.%H.%M)
log='save'
training_tag=''
if [ $# -ge 2 ]; then 

    log=$2
    training_tag=$1

elif [ $# -ge 1 ]; then 
    
    training_tag=$1

elif [[ $training_tag = "" ]]; then 
    training_tag=$time_stamp
fi 
echo "training_tag : ${training_tag}"


for model_config in "${model_configs[@]}"
do

    if [[ $model_config == *"_ae"* ]]; then 

        for traj_recon_dataset in "${traj_recon_datasets[@]}"
        do 

            dataset_name=($(echo $traj_recon_dataset | tr "/" "\n"))
            echo "=============================================="
            echo "model_config=${model_config}" 
            echo "dataset=${dataset_name[-1]}"
            echo "=============================================="

            mkdir "logs/${model_config}_${training_tag}"

            if [ $log = 'save' ]; then 

                # output_log="logs/${model_config}/${dataset_name[-2]}/${dataset_name[-1]}_log.txt"
                output_log="logs/${model_config}_${training_tag}/${dataset_name[-2]}_${dataset_name[-1]}.txt"
                python3 train_kptraj_recon_ae.py --dataset_dir $traj_recon_dataset --training_tag $training_tag --config "../config/${model_config}.yaml" > $output_log
                python3 plot_history.py $output_log
            
            else 
            
                python3 train_kptraj_recon_ae.py --dataset_dir $traj_recon_dataset --config "../config/${model_config}.yaml"
            
            fi

        done 

    else 

        echo "Wrong model type : ${model_config}"

    fi 

done 
