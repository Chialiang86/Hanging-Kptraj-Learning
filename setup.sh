# bin/sh

/usr/bin/python -m pip install --upgrade pip
mkdir dataset
mkdir dataset/traj_recon_affordance
mkdir src/checkpoints
pip install -r requirements.txt
cd ..
git clone https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .