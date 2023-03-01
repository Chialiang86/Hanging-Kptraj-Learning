# bin/sh
obj_shape_root="../shapes/inference_objs"
hook_shape_root="../shapes/hook_all_new_0"
inference_dir="../dataset/traj_recon_affordance/hook_all_new_0-kptraj_all_new_0-absolute-40/02.27.10.29-1000/test"
affordance_checkpoint_dir="checkpoints/affordance_02.27.17.30/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000"
affordance_weight_subpath="1000_points-network_epoch-10000.pth"
trajectory_checkpoint_dir="checkpoints/traj_recon_affordance_cvae_kl_large_02.28.19.50/hook_all_new_0-kptraj_all_new_0-absolute-40_02.27.10.29-1000"
trajectory_weight_subpath="1000_points-network_epoch-25000.pth"

affordance_config="../config/affordance.yaml"
trajectory_config="../config/traj_recon_affordance_cvae_kl_large.yaml"

python3 inference.py --inference_dir $inference_dir \
                    --obj_shape_root $obj_shape_root \
                    --hook_shape_root $hook_shape_root \
                    --affordance_checkpoint_dir $affordance_checkpoint_dir \
                    --affordance_weight_subpath $affordance_weight_subpath \
                    --trajectory_checkpoint_dir $trajectory_checkpoint_dir \
                    --trajectory_weight_subpath $trajectory_weight_subpath \
                    --affordance_config $affordance_config \
                    --trajectory_config $trajectory_config


