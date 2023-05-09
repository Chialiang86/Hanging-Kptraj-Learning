# bin/sh

if [ $# -lt 1 ]; then 
    echo 'please specify running type: train, val, test'
    exit 1
fi

if [ $1 = 'train' ]; then 

    model_configs=(

        "traj_3d_fusion_mutual_noise"
        "traj_3d_fusion_mutual_noise"
        "traj_3d_fusion_mutual_noise_10"
        "traj_3d_fusion_mutual_noise_10"
        "traj_3d_fusion_mutual_noise_20"
        "traj_3d_fusion_mutual_noise_20"

        "traj_fusion_mutual_noise"
        "traj_fusion_mutual_noise"
        "traj_fusion_mutual_noise_10"
        "traj_fusion_mutual_noise_10"
        "traj_fusion_mutual_noise_20"
        "traj_fusion_mutual_noise_20"
    )

    traj_recon_affordance_datasets=(

        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview"

        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview"
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

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do

        model_config=${model_configs[$i]}
        traj_recon_affordance_dataset=${traj_recon_affordance_datasets[$i]}
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

elif [ $1 = 'val' ]; then 

    model_configs=(
        # "traj_af_mutual_10"
        # "traj_af_mutual"
        # "traj_part_mutual"
    )

    dataset_dirs=(
        "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/onetraj-1000"
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

    model_configs=(

        # "traj_fusion_mutual"
        # "traj_fusion_mutual"
        # "traj_fusion_mutual"
        # "traj_fusion_mutual"

        # "traj_fusion_mutual_nn_dist_mr"
        # "traj_fusion_mutual_nn_dist_mr"
        # "traj_fusion_mutual_nn_dist_mr"
        # "traj_fusion_mutual_nn_dist_mr"

        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"

        # "traj_fusion_mutual_nn_dist_mr_10"
        # "traj_fusion_mutual_nn_dist_mr_10"
        # "traj_fusion_mutual_nn_dist_mr_10"
        # "traj_fusion_mutual_nn_dist_mr_10"

        # "traj_fusion_mutual_20"
        # "traj_fusion_mutual_20"
        # "traj_fusion_mutual_20"
        # "traj_fusion_mutual_20"

        # "traj_fusion_mutual_nn_dist_mr_20"
        # "traj_fusion_mutual_nn_dist_mr_20"
        # "traj_fusion_mutual_nn_dist_mr_20"
        # "traj_fusion_mutual_nn_dist_mr_20"

        # "traj_fusion_mutual_noise"
        # "traj_fusion_mutual_noise"
        # "traj_fusion_mutual_noise"
        # "traj_fusion_mutual_noise"

        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"

        # "traj_fusion_mutual"
        # "traj_fusion_mutual"
        # "traj_fusion_mutual"
        # "traj_fusion_mutual"

        # "traj_fusion_mutual"
        # "traj_fusion_mutual"
        # "traj_fusion_mutual"
        # "traj_fusion_mutual"

        "traj_fusion_mutual_noise"
        "traj_fusion_mutual_noise"
 
        "traj_fusion_mutual"
        "traj_fusion_mutual"

    )

    dataset_dirs=(

        # # 40
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"

        # # 10
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"

        # # 20
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"

        # # 40
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"

        # # 10 devil
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"

        # # 40 devil
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"

        # 40 
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"

        # 40 
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview"
        
        # 40 
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview"
    )

    inference_dirs=(
        # # 40
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"

        # # 10
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/val"

        # # 20
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/val"

        # # 40
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"

        # # 10 devil
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/val"
        
        # # 40 devil
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"
        
        # 40 
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"

        # 40 
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview/val"
        
        # 40 
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview/val"
    )

    traj_recon_shape_checkpoints=(

        # # 40
        # "checkpoints/traj_fusion_mutual-03.27.17.37/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual-03.27.17.37/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_fusion_mutual-03.27.17.37/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual-03.27.17.37/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"

        # "checkpoints/traj_fusion_mutual_nn_dist_mr-03.27.17.38/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_nn_dist_mr-03.27.17.38/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_fusion_mutual_nn_dist_mr-03.27.17.38/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_nn_dist_mr-03.27.17.38/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"

        # # 10
        # "checkpoints/traj_fusion_mutual_10-03.27.17.10/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_10-03.27.17.10/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000"
        # "checkpoints/traj_fusion_mutual_10-03.27.17.10/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_10-03.27.17.10/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000"

        # "checkpoints/traj_fusion_mutual_nn_dist_mr_10-03.27.17.10/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_10-03.27.17.10/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000"
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_10-03.27.17.10/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_10-03.27.17.10/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000"

        # # 20
        # "checkpoints/traj_fusion_mutual_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000"
        # "checkpoints/traj_fusion_mutual_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000"

        # "checkpoints/traj_fusion_mutual_nn_dist_mr_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000"
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000"

        # # 40
        # "checkpoints/traj_fusion_mutual_noise-04.06.16.33/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_noise-04.06.16.33/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_fusion_mutual_noise-04.06.16.33/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_noise-04.06.16.33/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"
        
        # # 10 devil
        # "checkpoints/traj_fusion_mutual_10-04.12.23.14-devil/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_10-04.12.23.14-devil/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000"
        # "checkpoints/traj_fusion_mutual_10-04.12.23.14-devil/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_10-04.12.23.14-devil/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000"
        
        # # 40 devil
        # "checkpoints/traj_fusion_mutual-04.12.23.15-devil/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_fusion_mutual-04.12.23.15-devil/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_fusion_mutual-04.12.23.15-devil/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_fusion_mutual-04.12.23.15-devil/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"
        
        # 40
        # "checkpoints/traj_fusion_mutual-05.01.15.45_3000/kptraj_all_smooth-absolute-40-k0-04.25.19.37-1000"
        # "checkpoints/traj_fusion_mutual-05.01.15.45_3000/kptraj_all_smooth-absolute-40-k0-04.25.19.37-1000"
        # "checkpoints/traj_fusion_mutual-05.01.15.45_3000/kptraj_all_smooth-residual-40-k0-04.25.19.37-1000"
        # "checkpoints/traj_fusion_mutual-05.01.15.45_3000/kptraj_all_smooth-residual-40-k0-04.25.19.37-1000"

        # 40
        "checkpoints/traj_fusion_mutual_noise-05.02.22.52-3000_fullview/kptraj_all_smooth-absolute-40-k0-05.02.18.59-1000-fullview"
        "checkpoints/traj_fusion_mutual_noise-05.02.22.52-3000_fullview/kptraj_all_smooth-residual-40-k0-05.02.18.59-1000-fullview"
        
        # 40
        "checkpoints/traj_fusion_mutual-05.02.22.55-3000_singleview/kptraj_all_smooth-absolute-40-k0-05.02.20.23-1000-singleview"
        "checkpoints/traj_fusion_mutual-05.02.22.55-3000_singleview/kptraj_all_smooth-residual-40-k0-05.02.20.23-1000-singleview"
    )

    points='3000'

    iters=(
        '10000' '20000'
        # '20000' 
        # '30000'  
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
                                                        --weight_subpath "${points}_points-network_epoch-${iter}.pth" \
                                                        --obj_shape_root ${obj_shape_root} \
                                                        --hook_shape_root ${hook_shape_root} \
                                                        --evaluate \
                                                        --visualize 
        done
    done

elif [ $1 = 'analysis' ]; then

    obj_shape_root="../shapes/inference_objs_5"
    # obj_shape_root="../shapes/inference_objs"
    hook_shape_root="../shapes/hook_all_new"
    # hook_shape_root="../shapes/hook_all_new_0"

    model_configs=(

        "traj_fusion_mutual"
        "traj_fusion_mutual"
        "traj_fusion_mutual"
        "traj_fusion_mutual"

        "traj_fusion_mutual_nn_dist_mr"
        "traj_fusion_mutual_nn_dist_mr"
        "traj_fusion_mutual_nn_dist_mr"
        "traj_fusion_mutual_nn_dist_mr"

        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"

        # "traj_fusion_mutual_nn_dist_mr_10"
        # "traj_fusion_mutual_nn_dist_mr_10"
        # "traj_fusion_mutual_nn_dist_mr_10"
        # "traj_fusion_mutual_nn_dist_mr_10"

        # "traj_fusion_mutual_20"
        # "traj_fusion_mutual_20"
        # "traj_fusion_mutual_20"
        # "traj_fusion_mutual_20"

        # "traj_fusion_mutual_nn_dist_mr_20"
        # "traj_fusion_mutual_nn_dist_mr_20"
        # "traj_fusion_mutual_nn_dist_mr_20"
        # "traj_fusion_mutual_nn_dist_mr_20"

        # "traj_fusion_mutual_noise"
        # "traj_fusion_mutual_noise"
        # "traj_fusion_mutual_noise"
        # "traj_fusion_mutual_noise"

        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"
        # "traj_fusion_mutual_10"

        # "traj_fusion_mutual"
        # "traj_fusion_mutual"
        # "traj_fusion_mutual"
        # "traj_fusion_mutual"

    )

    dataset_dirs=(

        # 40
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"

        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"

        # # 10
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"

        # # 20
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000"

        # # 40
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"

        # # 10 devil
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000"

        # # 40 devil
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000"
    )

    inference_dirs=(
        # 40
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"

        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"

        # # 10
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/val"

        # # 20
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/val"

        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/03.24.19.28-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/03.24.19.28-1000/val"

        # # 40
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"

        # # 10 devil
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/03.24.19.24-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/03.24.19.24-1000/val"
        
        # # 40 devil
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/03.20.13.31-1000/val"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/train"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/03.20.13.31-1000/val"

    )

    traj_recon_shape_checkpoints=(

        # 40
        "checkpoints/traj_fusion_mutual-03.27.17.37/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000" # sigmoid
        "checkpoints/traj_fusion_mutual-03.27.17.37/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        "checkpoints/traj_fusion_mutual-03.27.17.37/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000" # sigmoid
        "checkpoints/traj_fusion_mutual-03.27.17.37/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"

        "checkpoints/traj_fusion_mutual_nn_dist_mr-03.27.17.38/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000" # sigmoid
        "checkpoints/traj_fusion_mutual_nn_dist_mr-03.27.17.38/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        "checkpoints/traj_fusion_mutual_nn_dist_mr-03.27.17.38/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000" # sigmoid
        "checkpoints/traj_fusion_mutual_nn_dist_mr-03.27.17.38/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"

        # # 10
        # "checkpoints/traj_fusion_mutual_10-03.27.17.10/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_10-03.27.17.10/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000"
        # "checkpoints/traj_fusion_mutual_10-03.27.17.10/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_10-03.27.17.10/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000"

        # "checkpoints/traj_fusion_mutual_nn_dist_mr_10-03.27.17.10/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_10-03.27.17.10/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000"
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_10-03.27.17.10/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_10-03.27.17.10/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000"

        # # 20
        # "checkpoints/traj_fusion_mutual_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000"
        # "checkpoints/traj_fusion_mutual_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000"

        # "checkpoints/traj_fusion_mutual_nn_dist_mr_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_20-03.25.15.52/kptraj_all_smooth-absolute-20-k0-03.24.19.28-1000"
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_nn_dist_mr_20-03.25.15.52/kptraj_all_smooth-residual-20-k0-03.24.19.28-1000"

        # # 40
        # "checkpoints/traj_fusion_mutual_noise-04.06.16.33/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_noise-04.06.16.33/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_fusion_mutual_noise-04.06.16.33/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_noise-04.06.16.33/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"
        
        # # 10 devil
        # "checkpoints/traj_fusion_mutual_10-04.12.23.14-devil/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_10-04.12.23.14-devil/kptraj_all_smooth-absolute-10-k0-03.24.19.24-1000"
        # "checkpoints/traj_fusion_mutual_10-04.12.23.14-devil/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000" # sigmoid
        # "checkpoints/traj_fusion_mutual_10-04.12.23.14-devil/kptraj_all_smooth-residual-10-k0-03.24.19.24-1000"
        
        # # 40 devil
        # "checkpoints/traj_fusion_mutual-04.12.23.15-devil/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_fusion_mutual-04.12.23.15-devil/kptraj_all_smooth-absolute-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_fusion_mutual-04.12.23.15-devil/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"
        # "checkpoints/traj_fusion_mutual-04.12.23.15-devil/kptraj_all_smooth-residual-40-k0-03.20.13.31-1000"

    )

    iters=(
        '20000' '30000'
        # '22000' '24000' '26000' '28000' 
        # '20000' 
        # '30000'  
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

