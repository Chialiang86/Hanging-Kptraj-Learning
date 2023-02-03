# bin/sh

python3 data_preprocessing.py -kt 0 -dc 2 -kd kptraj_1104_origin_last2hook
python3 data_preprocessing.py -kt 1 -dc 2 -kd kptraj_1104_origin_last2hook
python3 data_preprocessing.py -kt 0 -dc 2 -kd kptraj_1104_origin_last2hook_aug
python3 data_preprocessing.py -kt 1 -dc 2 -kd kptraj_1104_origin_last2hook_aug