# bin/sh

if [ $# -lt 1 ]; then 
    echo 'please specify running type: train, val, test'
    exit 1
fi

if [ $1 = 'train' ]; then 

    model_configs=(

        "traj3d_deform_fusion_mutual_lstm_seg_v2_noise"
        "traj3d_deform_fusion_mutual_lstm_seg_v2_noise_10"
        "traj3d_deform_fusion_mutual_lstm_seg_v2_noise_20"

        "traj_deform_fusion_mutual_lstm_seg_v2_noise"
        "traj_deform_fusion_mutual_lstm_seg_v2_noise_10"
        "traj_deform_fusion_mutual_lstm_seg_v2_noise_20"

    )

    traj_recon_affordance_datasets=(

        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview"

        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview"
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
        
        # mkdir "training_logs/${model_config}-${training_tag}"

        if [ $log = 'save' ]; then 

            # output_log="logs/${model_config}/${dataset_name[-2]}/${dataset_name[-1]}_log.txt"
            output_log="training_logs/${model_config}-${training_tag}/${dataset_name[-2]}-${dataset_name[-1]}.txt"
            python3 train_kptraj_deform_mutual_seg.py --dataset_dir $traj_recon_affordance_dataset --training_tag $training_tag --config "../config/traj_deform_mutual/${model_config}.yaml" #> $output_log
            # python3 plot_history.py $output_log

        else 

            python3 train_kptraj_deform_mutual_seg.py --dataset_dir $traj_recon_affordance_dataset --training_tag $training_tag --config "../config/traj_deform_mutual/${model_config}.yaml"

        fi 

            
    done

# elif [ $1 = 'val' ]; then 

#     model_configs=(
#         # "traj_af_mutual_10"
#         # "traj_af_mutual"
#         # "traj_part_mutual"
#     )

#     dataset_dirs=(
#         "../dataset/traj_recon_affordance/kptraj_all_new-absolute-10/onetraj-1000"
#     )

#     traj_recon_shape_checkpoints=(
#         "checkpoints/traj_af_mutual_10-03.11.17.27-onetraj_10/kptraj_all_new-absolute-10-onetraj-1000"
#     )

#     iters=(
#         #'10000' '20000' 
#         '30000'  
#     )

#     length=${#model_configs[@]}

#     for (( i=0; i<$length; i++ )) 
#     do

#         for iter in "${iters[@]}"
#         do 

#             python3 train_kptraj_deform_mutual_seg.py --training_mode 'val' \
#                                                             --dataset_dir ${dataset_dirs[$i]} \
#                                                             --checkpoint_dir ${traj_recon_shape_checkpoints[$i]} \
#                                                             --config "../config/traj_deform_mutual/${model_configs[$i]}.yaml" \
#                                                             --weight_subpath "1000_points-network_epoch-${iter}.pth" 
#         done
#     done

elif [ $1 = 'test' ]; then

    obj_shape_root="../shapes/inference_daily"
    # obj_shape_root="../shapes/inference_mug"
    # obj_shape_root="../shapes/inference_scissor"
    # obj_shape_root="../shapes/inference_cooking"
    # obj_shape_root="../shapes/inference_tool"
    # obj_shape_root="../shapes/inference_objs_50"
    # obj_shape_root="../shapes/inference_objs"
    hook_shape_root="../shapes/hook_all_new"
    # hook_shape_root="../shapes/hook_all_new_0"


    model_configs=(
        # # 10
        # "traj3d_deform_fusion_mutual_lstm_seg_v2_noise_10"
        # "traj3d_deform_fusion_mutual_lstm_seg_v2_noise_10"

        # # 20
        # "traj3d_deform_fusion_mutual_lstm_seg_v2_noise_20"
        # "traj3d_deform_fusion_mutual_lstm_seg_v2_noise_20"

        # # 40
        "traj3d_deform_fusion_mutual_lstm_seg_v2_noise"
        # "traj3d_deform_fusion_mutual_lstm_seg_v2_noise"

        # 10
        # "traj_deform_fusion_mutual_lstm_seg_v2_noise_10"
        # "traj_deform_fusion_mutual_lstm_seg_v2_noise_10"

        # 20
        # "traj_deform_fusion_mutual_lstm_seg_v2_noise_20"
        # "traj_deform_fusion_mutual_lstm_seg_v2_noise_20"

        # 40
        # "traj_deform_fusion_mutual_lstm_seg_v2_noise"
        # "traj_deform_fusion_mutual_lstm_seg_v2_noise"
    )

    dataset_dirs=(

        # # 10
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview"

        # # 20
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview"

        # # 40
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview"

        # 10
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview"

        # 20
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview"

        # 40
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview"

    )

    inference_dirs=(

        # # 10
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview/val_deform"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview/val_deform"

        # # 20
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview/val_deform"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview/val_deform"

        # # 40
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/val_deform"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview/val_deform"

        # 10
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview/val_deform"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview/val_deform"

        # 20
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview/val_deform"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview/val_deform"

        # 40
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/val_deform"
        # "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview/val_deform"

    )

    traj_recon_shape_checkpoints=(
        
        # # 10
        # "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise_10-05.08.16.47-formal_0/kptraj_all_smooth-absolute-10-k0-05.02.20.53-1000-singleview"
        # "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise_10-05.08.16.47-formal_0/kptraj_all_smooth-residual-10-k0-05.02.20.53-1000-singleview"
        
        # # 20
        # "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise_20-05.08.16.47-formal_0/kptraj_all_smooth-absolute-20-k0-05.02.20.39-1000-singleview"
        # "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise_20-05.08.16.47-formal_0/kptraj_all_smooth-residual-20-k0-05.02.20.39-1000-singleview"
        
        # # 40
        "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise-05.08.16.47-formal_0/kptraj_all_smooth-absolute-40-k0-05.02.20.23-1000-singleview"
        # "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise-05.08.16.47-formal_0/kptraj_all_smooth-residual-40-k0-05.02.20.23-1000-singleview"
        
        # # 10
        # "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise_10-05.08.21.05-formal_0/kptraj_all_smooth-absolute-10-k0-05.02.20.53-1000-singleview"
        # "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise_10-05.08.21.05-formal_0/kptraj_all_smooth-residual-10-k0-05.02.20.53-1000-singleview"
        
        # # 20
        # "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise_20-05.08.21.05-formal_0/kptraj_all_smooth-absolute-20-k0-05.02.20.39-1000-singleview"
        # "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise_20-05.08.21.05-formal_0/kptraj_all_smooth-residual-20-k0-05.02.20.39-1000-singleview"
        
        # # 40
        # "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise-05.08.21.05-formal_0/kptraj_all_smooth-absolute-40-k0-05.02.20.23-1000-singleview"
        # "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise-05.08.21.05-formal_0/kptraj_all_smooth-absolute-40-k0-05.02.20.23-1000-singleview"
    )

    iters=(
        # "1000" "3000" "3500" "4000" "4500"
        # "2000" 
        "5000"
    )

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do

        for iter in "${iters[@]}"
        do 

            python3 train_kptraj_deform_mutual_seg.py --training_mode 'test' \
                                                        --dataset_dir ${dataset_dirs[$i]} \
                                                        --inference_dir ${inference_dirs[$i]} \
                                                        --checkpoint_dir ${traj_recon_shape_checkpoints[$i]} \
                                                        --config "../config/traj_deform_mutual/${model_configs[$i]}.yaml" \
                                                        --weight_subpath "1000_points-network_epoch-${iter}.pth" \
                                                        --obj_shape_root ${obj_shape_root} \
                                                        --hook_shape_root ${hook_shape_root} \
                                                        --evaluate \
                                                        --visualize 
                                                        # --use_temp
                                                        # --use_gt_cls \
                                                        # --use_gt_cp 
        done
    done

elif [ $1 = 'analysis' ]; then

    obj_shape_root="../shapes/inference_objs_5"
    # obj_shape_root="../shapes/inference_objs"
    hook_shape_root="../shapes/hook_all_new"
    # hook_shape_root="../shapes/hook_all_new_0"

    model_configs=(
        # 10
        "traj3d_deform_fusion_mutual_lstm_seg_v2_noise_10"
        "traj3d_deform_fusion_mutual_lstm_seg_v2_noise_10"

        # 20
        "traj3d_deform_fusion_mutual_lstm_seg_v2_noise_20"
        "traj3d_deform_fusion_mutual_lstm_seg_v2_noise_20"

        # 40
        "traj3d_deform_fusion_mutual_lstm_seg_v2_noise"
        "traj3d_deform_fusion_mutual_lstm_seg_v2_noise"

        # 10
        "traj_deform_fusion_mutual_lstm_seg_v2_noise_10"
        "traj_deform_fusion_mutual_lstm_seg_v2_noise_10"

        # 20
        "traj_deform_fusion_mutual_lstm_seg_v2_noise_20"
        "traj_deform_fusion_mutual_lstm_seg_v2_noise_20"

        # 40
        "traj_deform_fusion_mutual_lstm_seg_v2_noise"
        "traj_deform_fusion_mutual_lstm_seg_v2_noise"
    )

    dataset_dirs=(

        # 10
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview"

        # 20
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview"

        # 40
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview"

        # 10
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview"

        # 20
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview"

        # 40
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview"

    )

    inference_dirs=(

        # 10
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview/val_deform"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview/val_deform"

        # 20
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview/val_deform"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview/val_deform"

        # 40
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/val_deform"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview/val_deform"

        # 10
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview/val_deform"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0/05.02.20.53-1000-singleview/val_deform"

        # 20
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview/val_deform"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0/05.02.20.39-1000-singleview/val_deform"

        # 40
        "../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview/val_deform"
        "../dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0/05.02.20.23-1000-singleview/val_deform"

    )

    traj_recon_shape_checkpoints=(
        
        # 10
        "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise_10-05.08.16.47-formal_0/kptraj_all_smooth-absolute-10-k0-05.02.20.53-1000-singleview"
        "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise_10-05.08.16.47-formal_0/kptraj_all_smooth-residual-10-k0-05.02.20.53-1000-singleview"
        
        # 20
        "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise_20-05.08.16.47-formal_0/kptraj_all_smooth-absolute-20-k0-05.02.20.39-1000-singleview"
        "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise_20-05.08.16.47-formal_0/kptraj_all_smooth-residual-20-k0-05.02.20.39-1000-singleview"
        
        # 40
        "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise-05.08.16.47-formal_0/kptraj_all_smooth-absolute-40-k0-05.02.20.23-1000-singleview"
        "checkpoints/traj3d_deform_fusion_mutual_lstm_seg_v2_noise-05.08.16.47-formal_0/kptraj_all_smooth-residual-40-k0-05.02.20.23-1000-singleview"
        
        # 10
        "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise_10-05.08.21.05-formal_0/kptraj_all_smooth-absolute-10-k0-05.02.20.53-1000-singleview"
        "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise_10-05.08.21.05-formal_0/kptraj_all_smooth-residual-10-k0-05.02.20.53-1000-singleview"
        
        # 20
        "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise_20-05.08.21.05-formal_0/kptraj_all_smooth-absolute-20-k0-05.02.20.39-1000-singleview"
        "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise_20-05.08.21.05-formal_0/kptraj_all_smooth-residual-20-k0-05.02.20.39-1000-singleview"
        
        # 40
        "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise-05.08.21.05-formal_0/kptraj_all_smooth-absolute-40-k0-05.02.20.23-1000-singleview"
        "checkpoints/traj_deform_fusion_mutual_lstm_seg_v2_noise-05.08.21.05-formal_0/kptraj_all_smooth-absolute-40-k0-05.02.20.23-1000-singleview"
    )

    iters=(
        "1000" "3000" "3500" "4000" "4500"
    )

    length=${#model_configs[@]}

    for (( i=0; i<$length; i++ )) 
    do

        for iter in "${iters[@]}"
        do 

            python3 train_kptraj_deform_mutual_seg.py --training_mode 'analysis' \
                                                        --dataset_dir ${dataset_dirs[$i]} \
                                                        --inference_dir ${inference_dirs[$i]} \
                                                        --checkpoint_dir ${traj_recon_shape_checkpoints[$i]} \
                                                        --config "../config/traj_deform_mutual/${model_configs[$i]}.yaml" \
                                                        --weight_subpath "1000_points-network_epoch-${iter}.pth" \
                                                        --obj_shape_root ${obj_shape_root} \
                                                        --hook_shape_root ${hook_shape_root} \
                                                        --evaluate \
                                                        --visualize 
        done
    done
else 

    echo '[Error] wrong runing type (should be train, val, test)'
    exit 

fi

