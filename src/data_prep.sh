# bin/sh

# 0.02, 0.008, 0.026
python3 data_preprocessing.py -dc '2' -sd 'hook_all_new' -kd 'kptraj_all_smooth' --kptraj_length '40' --kptraj_sample_distance '0.002' --kptraj_num '1000'
python3 data_preprocessing.py -dc '2' -sd 'hook_all_new' -kd 'kptraj_all_smooth' --kptraj_length '20' --kptraj_sample_distance '0.008' --kptraj_num '1000'
python3 data_preprocessing.py -dc '2' -sd 'hook_all_new' -kd 'kptraj_all_smooth' --kptraj_length '10' --kptraj_sample_distance '0.028' --kptraj_num '1000'