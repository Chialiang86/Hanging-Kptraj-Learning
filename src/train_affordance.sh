# bin/sh

if [ $# -lt 1 ]; then 
    echo 'please specify running type: train, val, test'
    exit 1
fi

if [ $1 = 'train' ]; then 

    model_configs=(
        "af" 
        "af_msg"
        "fusion" 
        "fusion_msg"
    )

    affordance_datasets=(
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.20.13.31-1000"
    )

    training_tag='' # $1
    log='save' # $2
    time_stamp=$(date +%m.%d.%H.%M)
    training_tag=''

    if [ $# -ge 2 ]; then 
        
        training_tag="${time_stamp}-${2}"

    elif [ $# -ge 3 ]; then 

        training_tag="${time_stamp}-${2}"
        log=$3

    elif [[ $training_tag = "" ]]; then 

        training_tag="${time_stamp}"

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
                # python3 plot_history.py $output_log

            else 

                python3 train_affordance.py --dataset_dir $affordance_dataset --training_tag $training_tag --config "../config/${model_config}.yaml"

            fi 

        done 

    done 

elif [ $1 = 'val' ]; then 

    iters=(
        '20000'
    )

    model_configs=(
        # "af" 
        "af_msg" 
    )

    dataset_dirs=(
        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-40/03.05.12.50-1000"
    )

    # element number should be the same as model_configs
    affordance_checkpoints=(

        "checkpoints/af_msg-03.05.13.45/hook_all_new-kptraj_all_new-absolute-40_03.05.12.50-1000"

    )
    # element number should be the same as model_configs

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do 
        for iter in "${iters[@]}"
        do 

            python3 train_affordance.py -tm 'val' \
                                        --checkpoint_dir ${affordance_checkpoints[$i]} \
                                        --config "../config/${model_configs[$i]}.yaml" \
                                        --dataset_dir "${dataset_dirs[$i]}" \
                                        --weight_subpath "1000_points-network_epoch-${iter}.pth" \
                                        -v 
                                        # --evaluate
    
        done 
    done 

    
elif [ $1 = 'test' ]; then 

    iters=(
        '1000' '2000'
    )

    model_configs=(
        "af" 
        "af_msg" 
        "fusion" 
        "fusion_msg" 
    )

    # element number should be the same as model_configs
    inference_directories=(

        # "../shapes/realworld_hook"
        # "../shapes/realworld_hook"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.20.13.31-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.20.13.31-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.20.13.31-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.20.13.31-1000/val"
    )

    # element number should be the same as model_configs
    affordance_checkpoints=(

        "checkpoints/af_03.23.16.56/kptraj_all_smooth-absolute-10-k0_03.20.13.31-1000"
        "checkpoints/af_msg_03.23.16.56/kptraj_all_smooth-absolute-10-k0_03.20.13.31-1000"
        "checkpoints/fusion_03.23.16.56/kptraj_all_smooth-absolute-10-k0_03.20.13.31-1000"
        "checkpoints/fusion_msg_03.23.16.56/kptraj_all_smooth-absolute-10-k0_03.20.13.31-1000"

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
                                        --weight_subpath "1000_points-network_epoch-${iter}.pth" \
                                        --evaluate
                                        # -v \
    
        done 
    done 

else 

    echo '[Error] wrong runing type (should be train, val, test)'
    exit 

fi