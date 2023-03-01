# bin/sh

model_configs=(
    "affordance" "affordance_msg"
)

affordance_datasets=(
    # "../dataset/traj_recon_affordance/hook_all_new-kptraj_all_new-absolute-40/02.27.22.44-1000"
    # "../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000"
    "../dataset/traj_recon_affordance/hook_all_new-kptraj_all_new-absolute-40/02.27.22.44-1000"
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

elif [[ $training_tag = "" ]]; then 
    training_tag=$time_stamp
fi 
echo "training_tag : ${training_tag}"
echo "log : ${log}"

for model_config in "${model_configs[@]}"
do
        
    for affordance_dataset in "${affordance_datasets[@]}"
    do 

        dataset_name=($(echo $affordance_dataset | tr "/" "\n"))
        echo "=============================================="
        echo "model_config=${model_config}" 
        echo "dataset=${dataset_name[-1]}"
        echo "=============================================="
        
        mkdir "training_logs/${model_config}_${training_tag}"

        if [ $log = 'save' ]; then 

            # output_log="logs/${model_config}/${dataset_name[-2]}/${dataset_name[-1]}_log.txt"
            output_log="training_logs/${model_config}_${training_tag}/${dataset_name[-2]}_${dataset_name[-1]}.txt"
            python3 train_affordance.py --dataset_dir $affordance_dataset --training_tag $training_tag --config "../config/${model_config}.yaml" > $output_log
            python3 plot_history.py $output_log

        else 

            python3 train_affordance.py --dataset_dir $affordance_dataset --training_tag $training_tag --config "../config/${model_config}.yaml"

        fi 

    done 

done 
