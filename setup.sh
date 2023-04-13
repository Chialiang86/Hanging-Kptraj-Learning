# bin/sh

/usr/bin/python -m pip install --upgrade pip
mkdir dataset
mkdir dataset/traj_recon_affordance
mkdir dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0
mkdir dataset/traj_recon_affordance/kptraj_all_smooth-residual-10-k0
mkdir dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0
mkdir dataset/traj_recon_affordance/kptraj_all_smooth-residual-20-k0
mkdir dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0
mkdir dataset/traj_recon_affordance/kptraj_all_smooth-residual-40-k0
mkdir src/checkpoints
pip install -r requirements.txt
cd ..
git clone https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .