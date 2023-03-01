# bin/sh


model_configs=(
    # "traj_recon_affordance_cvae_kl_large_30" 
    # "traj_recon_affordance_cvae_kl_small_30" 
    # "traj_recon_affordance_cvae_kl_annealing_30"
    # "traj_recon_affordance_cvae_kl_large" 
    # "traj_recon_affordance_cvae_kl_small" 
    "traj_recon_affordance_rot_3d_cvae_kl_large" 
    "traj_recon_affordance_rot_3d_cvae_kl_small" 
    # "traj_recon_affordance_nofp_cvae_kl_large" 
    # "traj_recon_affordance_nofp_4d_cvae_kl_large" 
)

traj_recon_affordance_datasets=(
    "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000"
    "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-residual-40/02.27.10.32-1000"

    # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-absolute-30/02.03.13.28"
    # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook-residual-30/02.03.13.29"

    # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-absolute-30/02.11.13.38"
    # "../dataset/traj_recon_affordance/hook-kptraj_1104_origin_last2hook_aug-residual-30/02.11.13.39"
)

training_tag='' # $1
log='save' # $2
time_stamp=$(date +%m.%d.%H.%M)
training_tag=''
if [ $# -ge 2 ]; then 

    training_tag=$1
    log=$2

elif [ $# -ge 1 ]; then 
    
    training_tag=$1
input_pcid1
elif [[ $training_tag = "" ]]; then 
    training_tag=$time_stamp
fi 
echo "training_tag : ${training_tag}"
echo "log : ${log}"

for model_config in "${model_configs[@]}"
do

    if [[ $model_config == *"_cvae"* ]]; then 
        
        for traj_recon_affordance_dataset in "${traj_recon_affordance_datasets[@]}"
        do 

            dataset_name=($(echo $traj_recon_affordance_dataset | tr "/" "\n"))
            echo "=============================================="
            echo "model_config=${model_config}" 
            echo "dataset=${dataset_name[-1]}"
            echo "=============================================="
            
            mkdir "training_logs/${model_config}_${training_tag}"

            if [ $log = 'save' ]; then 

                # output_log="logs/${model_config}/${dataset_name[-2]}/${dataset_name[-1]}_log.txt"
                output_log="training_logs/${model_config}_${training_tag}/${dataset_name[-2]}_${dataset_name[-1]}.txt"
                python3 train_kptraj_recon_affordance_cvae.py --dataset_dir $traj_recon_affordance_dataset --training_tag $training_tag --config "../config/${model_config}.yaml" > $output_log
                python3 plot_history.py $output_log

            else 

                python3 train_kptraj_recon_affordance_cvae.py --dataset_dir $traj_recon_affordance_dataset --training_tag $training_tag --config "../config/${model_config}.yaml"

            fi 

        done 

    else 

        echo "Wrong model type : ${model_config}"

    fi 

done 
