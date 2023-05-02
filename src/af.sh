# bin/sh

if [ $# -lt 1 ]; then 
    echo 'please specify running type: train, val, test'
    exit 1
fi

if [ $1 = 'train' ]; then 

    model_configs=(
        # "af" 
        # "af_msg"
        "part" 
        "part_msg" 
        "fusion" 
        "fusion_msg"
    )

    affordance_datasets=(
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.15.51-1000-fullview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.18.59-1000-fullview"
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
                python3 train_affordance.py --dataset_dir $affordance_dataset --training_tag $training_tag --config "../config/af/${model_config}.yaml" > $output_log
                # python3 plot_history.py $output_log

            else 

                python3 train_affordance.py --dataset_dir $affordance_dataset --training_tag $training_tag --config "../config/af/${model_config}.yaml"

            fi 

        done 

    done 

# elif [ $1 = 'val' ]; then 

    # iters=(
    #     '600'
    # )

    # model_configs=(
    #     # "af" 
    #     "af_msg" 
    # )

    # dataset_dirs=(
    #     "../dataset/traj_recon_affordance/kptraj_all_new-absolute-40/03.05.12.50-1000"
    # )

    # # element number should be the same as model_configs
    # affordance_checkpoints=(

    #     # "checkpoints/af_msg-03.05.13.45/hook_all_new-kptraj_all_new-absolute-40_03.05.12.50-1000"
    #     "checkpoints/af_msg_04.26.10.59/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"

    # )
    # # element number should be the same as model_configs

    # length=${#model_configs[@]}

    # for (( i=0; i<$length; i++ )) 
    # do 
    #     for iter in "${iters[@]}"
    #     do 

    #         python3 train_affordance.py -tm 'val' \
    #                                     --checkpoint_dir ${affordance_checkpoints[$i]} \
    #                                     --config "../config/af/${model_configs[$i]}.yaml" \
    #                                     --dataset_dir "${dataset_dirs[$i]}" \
    #                                     --weight_subpath "1000_points-network_epoch-${iter}.pth" \
    #                                     -v 
    #                                     # --evaluate
    
    #     done 
    # done 

    
elif [ $1 = 'test' ]; then 

    iters=(
        '600' '800' '1000'
    )

    points='3000'

    model_configs=(
        # "fusion_msg" 
        # "fusion_msg" 
        # "fusion_msg" 
        # "fusion_msg" 

        # "af_msg"
        # "af_msg"

        # "af"
        # "af"

        # "fusion_msg" 
        # "fusion_msg" 

        # "part" 
        "part_msg" 
        "fusion" 
        "fusion_msg" 
    )

    # element number should be the same as model_configs
    inference_directories=(

        # "../shapes/realworld_hook"
        # "../shapes/realworld_hook"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/val"
    )

    # element number should be the same as model_configs
    affordance_checkpoints=(

        # "checkpoints/fusion_msg_04.19.13.45/kptraj_all_smooth-absolute-10-k0_03.24.19.24-1000"
        # "checkpoints/fusion_msg_04.19.13.45/kptraj_all_smooth-absolute-10-k0_03.24.19.24-1000"
        # "checkpoints/fusion_msg_04.25.20.30/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"
        # "checkpoints/fusion_msg_04.25.20.30/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"

        # "checkpoints/af_msg_04.26.10.59/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"
        # "checkpoints/af_msg_04.26.10.59/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"

        # "checkpoints/af_04.26.10.59/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"
        # "checkpoints/af_04.26.10.59/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"

        # "checkpoints/fusion_msg_05.01.14.27-3000/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"
        # "checkpoints/fusion_msg_05.01.14.27-3000/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"
        
        # "checkpoints/part_05.01.23.19-3000_noise/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"
        # "checkpoints/part_msg_05.01.23.19-3000_noise/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"
        # "checkpoints/fusion_05.01.23.19-3000_noise/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"
        # "checkpoints/fusion_msg_05.01.23.19-3000_noise/kptraj_all_smooth-absolute-40-k0_04.25.19.37-1000"

        # "checkpoints/part_05.02.16.22-3000_fullview/kptraj_all_smooth-absolute-40-k0_05.02.15.51-1000-fullview"
        "checkpoints/part_msg_05.02.16.22-3000_fullview/kptraj_all_smooth-absolute-40-k0_05.02.15.51-1000-fullview"
        "checkpoints/fusion_05.02.16.22-3000_fullview/kptraj_all_smooth-absolute-40-k0_05.02.15.51-1000-fullview"
        "checkpoints/fusion_msg_05.02.16.22-3000_fullview/kptraj_all_smooth-absolute-40-k0_05.02.15.51-1000-fullview"

    )

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do 
        for iter in "${iters[@]}"
        do 

            python3 train_affordance.py -tm 'test' \
                                        --inference_dir ${inference_directories[$i]} \
                                        --checkpoint_dir ${affordance_checkpoints[$i]} \
                                        --config "../config/af/${model_configs[$i]}.yaml" \
                                        --weight_subpath "${points}_points-network_epoch-${iter}.pth" \
                                        -v \
                                        --evaluate
    
        done 
    done 
    
elif [ $1 = 'analysis' ]; then 

    iters=(
        # '200' '400' '600' '800' '1000'
        # '2200' '2400' '2600' '2800' '3000'
        # '4200' '4400' '4600' '4800' '5000'
        # '6200' '6400' '6600' '6800' '7000'
        # '8200' '8400' '8600' '8800' '9000'
        # '600' '800' 
        '1000'
    )

    model_configs=(
        "fusion_msg" 
        "fusion_msg" 
    )

    # element number should be the same as model_configs
    inference_directories=(

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/train"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/04.25.19.37-1000/val"
    )

    # element number should be the same as model_configs
    affordance_checkpoints=(

        "checkpoints/fusion_msg_04.19.13.45/kptraj_all_smooth-absolute-10-k0_03.24.19.24-1000"
        "checkpoints/fusion_msg_04.19.13.45/kptraj_all_smooth-absolute-10-k0_03.24.19.24-1000"

    )

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do 
        for iter in "${iters[@]}"
        do 

            python3 train_affordance.py -tm 'analysis' \
                                        --inference_dir ${inference_directories[$i]} \
                                        --checkpoint_dir ${affordance_checkpoints[$i]} \
                                        --config "../config/af/${model_configs[$i]}.yaml" \
                                        --weight_subpath "1000_points-network_epoch-${iter}.pth" \
                                        -v \
                                        --evaluate
    
        done 
    done 

else 

    echo '[Error] wrong runing type (should be train, val, test)'
    exit 

fi