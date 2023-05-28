# bin/sh

server=$1
# cat ~/.ssh/id_rsa.pub | ssh $server "cat >> ~/.ssh/authorized_keys"

scp -r src/checkpoints/traj3d_deform_fusion_mutual_lstm_v2_noise_10-05.24.14.16-formal_4 "$server":~/chialiang/Hanging-Kptraj-Learning/src/checkpoints
scp -r src/checkpoints/traj3d_deform_fusion_mutual_lstm_v2_noise_20-05.24.14.16-formal_4 "$server":~/chialiang/Hanging-Kptraj-Learning/src/checkpoints
scp -r src/checkpoints/traj3d_deform_fusion_mutual_lstm_v2_noise-05.24.14.16-formal_4 "$server":~/chialiang/Hanging-Kptraj-Learning/src/checkpoints
scp -r src/checkpoints/traj_deform_fusion_mutual_lstm_v2_noise_10-05.24.14.17-formal_4 "$server":~/chialiang/Hanging-Kptraj-Learning/src/checkpoints
scp -r src/checkpoints/traj_deform_fusion_mutual_lstm_v2_noise_20-05.24.14.17-formal_4 "$server":~/chialiang/Hanging-Kptraj-Learning/src/checkpoints
scp -r src/checkpoints/traj_deform_fusion_mutual_lstm_v2_noise-05.24.14.17-formal_4 "$server":~/chialiang/Hanging-Kptraj-Learning/src/checkpoints
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview-5c "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview-5c "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-5c "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview-10c "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview-10c "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-10c "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0