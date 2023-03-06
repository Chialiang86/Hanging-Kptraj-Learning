# bin/sh

python3 data_preprocessing.py -kt '0' -dc '2' -sd 'hook_all_new' -kd 'kptraj_all_new' # --data_tag 'alltraj' -kn '1000'
python3 data_preprocessing.py -kt '1' -dc '2' -sd 'hook_all_new' -kd 'kptraj_all_new' # --data_tag 'alltraj' -kn '1000'