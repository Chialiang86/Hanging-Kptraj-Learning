# bin/sh

if [ $# -lt 1 ]; then 
    echo 'please specify running type: train, val, test'
    exit 1
fi

if [ $1 = 'train' ]; then 

    model_configs=(
        # "traj_af_mutual_10"
 
        # "traj_af_mutual_20"
 
        "traj_af_mutual"
    )

    traj_recon_affordance_datasets=(

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
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

        for traj_recon_affordance_dataset in "${traj_recon_affordance_datasets[@]}"
        do 

            dataset_name=($(echo $traj_recon_affordance_dataset | tr "/" "\n"))
            echo "=============================================="
            echo "model_config=${model_config}" 
            echo "dataset=${dataset_name[-1]}"
            echo "=============================================="
            
            mkdir "training_logs/${model_config}-${training_tag}"

            if [ $log = 'save' ]; then 

                # output_log="logs/${model_config}/${dataset_name[-2]}/${dataset_name[-1]}_log.txt"
                output_log="training_logs/${model_config}-${training_tag}/${dataset_name[-2]}-${dataset_name[-1]}.txt"
                python3 train_kptraj_recon_affordance_cvae_mutual.py --dataset_dir $traj_recon_affordance_dataset --training_tag $training_tag --config "../config/traj_af_mutual/${model_config}.yaml" > $output_log
                # python3 plot_history.py $output_log

            else 

                python3 train_kptraj_recon_affordance_cvae_mutual.py --dataset_dir $traj_recon_affordance_dataset --training_tag $training_tag --config "../config/traj_af_mutual/${model_config}.yaml"

            fi 

        done 
            
    done

elif [ $1 = 'val' ]; then 

    model_configs=(
        # "traj_af_mutual"
        # "traj_af_mutual_10"
    )

    dataset_dirs=(
        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/onetraj-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_new_0-absolute-40/02.27.10.29-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_new_0-residual-40/02.27.10.32-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_new-absolute-40/onetraj-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_new-residual-40/onetraj-1000"

        # "../dataset/traj_recon_affordance/kptraj_1104_origin_last2hook-absolute-30/02.03.13.28"
        # "../dataset/traj_recon_affordance/kptraj_1104_origin_last2hook-residual-30/02.03.13.29"

        # "../dataset/traj_recon_affordance/kptraj_1104_origin_last2hook_aug-absolute-30/02.11.13.38"
        # "../dataset/traj_recon_affordance/kptraj_1104_origin_last2hook_aug-residual-30/02.11.13.39"
    )

    traj_recon_shape_checkpoints=(
        "checkpoints/traj_af_mutual_10-03.11.17.27-onetraj_10/kptraj_all_new-absolute-10-onetraj-1000"
    )

    num_of_points=(
        '1000' #'1000' 
    )

    iters=(
        #'10000' '20000' 
        '30000'  
    )

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do

        for iter in "${iters[@]}"
        do 

            python3 train_kptraj_recon_affordance_cvae_mutual.py --training_mode 'val' \
                                                            --dataset_dir ${dataset_dirs[$i]} \
                                                            --checkpoint_dir ${traj_recon_shape_checkpoints[$i]} \
                                                            --config "../config/${model_configs[$i]}.yaml" \
                                                            --weight_subpath "${num_of_points[$i]}_points-network_epoch-${iter}.pth" 
        done
    done

elif [ $1 = 'test' ]; then

    obj_shape_root="../shapes/inference_objs_5"
    # obj_shape_root="../shapes/inference_objs"
    hook_shape_root="../shapes/hook_all_new"
    # hook_shape_root="../shapes/hook_all_new_0"
    # hook_shape_root="../shapes/hook_all_new_devil"

    model_configs=(

        # "traj_af_mutual_10"
        # "traj_af_mutual_10"
        # "traj_af_mutual_10"
        # "traj_af_mutual_10"

        # "traj_af_mutual_20"
        # "traj_af_mutual_20"
        # "traj_af_mutual_20"
        # "traj_af_mutual_20"

        "traj_af_mutual"
        # "traj_af_mutual"
        "traj_af_mutual"
        # "traj_af_mutual"

    )

    dataset_dirs=(

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"

        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
    )

    inference_dirs=(

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/val"

        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"
    )

    traj_recon_shape_checkpoints=(

        # "checkpoints/traj_af_mutual_10-03.25.15.31/kptraj_all_smooth-absolute-10-k0-03.20.13.31-1000"
        # "checkpoints/traj_af_mutual_10-03.25.15.31/kptraj_all_smooth-absolute-10-k0-03.20.13.31-1000"
        # "checkpoints/traj_af_mutual_10-03.25.15.31/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000"
        # "checkpoints/traj_af_mutual_10-03.25.15.31/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000"

        # "checkpoints/traj_af_mutual_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000"
        # "checkpoints/traj_af_mutual_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000"
        # "checkpoints/traj_af_mutual_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000"
        # "checkpoints/traj_af_mutual_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000"

        "checkpoints/traj_af_mutual-03.23.23.16/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_af_mutual-03.23.23.16/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        "checkpoints/traj_af_mutual-03.23.23.16/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_af_mutual-03.23.23.16/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"
    )

    iters=(
        #'20000' '22000' '24000' '26000' '28000' 
        # '20000' 
        '30000'  
    )



    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do

        for iter in "${iters[@]}"
        do 

            python3 train_kptraj_recon_affordance_cvae_mutual.py --training_mode 'test' \
                                                        --dataset_dir ${dataset_dirs[$i]} \
                                                        --inference_dir ${inference_dirs[$i]} \
                                                        --checkpoint_dir ${traj_recon_shape_checkpoints[$i]} \
                                                        --config "../config/traj_af_mutual/${model_configs[$i]}.yaml" \
                                                        --weight_subpath "1000_points-network_epoch-${iter}.pth" \
                                                        --obj_shape_root ${obj_shape_root} \
                                                        --hook_shape_root ${hook_shape_root} \
                                                        --evaluate 
                                                        # --visualize 
        done
    done

elif [ $1 = 'analysis' ]; then

    obj_shape_root="../shapes/inference_objs_5"
    # obj_shape_root="../shapes/inference_objs"
    hook_shape_root="../shapes/hook_all_new"
    # hook_shape_root="../shapes/hook_all_new_0"
    # hook_shape_root="../shapes/hook_all_new_devil"

    model_configs=(

        # "traj_af_mutual_10"
        # "traj_af_mutual_10"
        # "traj_af_mutual_10"
        # "traj_af_mutual_10"

        # "traj_af_mutual_20"
        # "traj_af_mutual_20"
        # "traj_af_mutual_20"
        # "traj_af_mutual_20"

        "traj_af_mutual"
        # "traj_af_mutual"
        "traj_af_mutual"
        # "traj_af_mutual"

    )

    dataset_dirs=(

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"

        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
    )

    inference_dirs=(

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/val"

        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"
    )

    traj_recon_shape_checkpoints=(

        # "checkpoints/traj_af_mutual_10-03.25.15.31/kptraj_all_smooth-absolute-10-k0-03.20.13.31-1000"
        # "checkpoints/traj_af_mutual_10-03.25.15.31/kptraj_all_smooth-absolute-10-k0-03.20.13.31-1000"
        # "checkpoints/traj_af_mutual_10-03.25.15.31/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000"
        # "checkpoints/traj_af_mutual_10-03.25.15.31/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000"

        # "checkpoints/traj_af_mutual_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000"
        # "checkpoints/traj_af_mutual_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000"
        # "checkpoints/traj_af_mutual_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000"
        # "checkpoints/traj_af_mutual_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000"

        "checkpoints/traj_af_mutual-03.23.23.16/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_af_mutual-03.23.23.16/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        "checkpoints/traj_af_mutual-03.23.23.16/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_af_mutual-03.23.23.16/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"
    )

    iters=(
        #'20000' '22000' '24000' '26000' '28000' 
        # '20000' 
        '30000'  
    )

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do

        for iter in "${iters[@]}"
        do 

            python3 train_kptraj_recon_affordance_cvae_mutual.py --training_mode 'analysis' \
                                                        --dataset_dir ${dataset_dirs[$i]} \
                                                        --inference_dir ${inference_dirs[$i]} \
                                                        --checkpoint_dir ${traj_recon_shape_checkpoints[$i]} \
                                                        --config "../config/traj_af_mutual/${model_configs[$i]}.yaml" \
                                                        --weight_subpath "1000_points-network_epoch-${iter}.pth" \
                                                        --obj_shape_root ${obj_shape_root} \
                                                        --hook_shape_root ${hook_shape_root}
        done
    done

else 

    echo '[Error] wrong runing type (should be train, val, test)'
    exit 

fi

