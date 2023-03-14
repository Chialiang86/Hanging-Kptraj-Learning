# bin/sh

# python3 data_preprocessing.py -kt '0' -dc '2' -sd 'hook_all_new' -kd 'kptraj_all_new' --kptraj_length '10' --data_tag 'onetraj' -kn '1' -ksd '0.01131370849898476'
python3 data_preprocessing.py -kt '1' -dc '2' -sd 'hook_all_new' -kd 'kptraj_all_new' --kptraj_length '10' --data_tag 'alltraj' -kn '1000' -ksd '0.01131370849898476'