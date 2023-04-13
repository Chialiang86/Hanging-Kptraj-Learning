# bin/sh

server=$1
# cat ~/.ssh/id_rsa.pub | ssh $server "cat >> ~/.ssh/authorized_keys"

scp -r dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0 "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0 "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0 "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0 "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0 "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/
scp -r dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0 "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/
# scp -r dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0 "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/
# scp -r dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0 "$server":~/chialiang/Hanging-Kptraj-Learning/dataset/traj_recon_affordance/