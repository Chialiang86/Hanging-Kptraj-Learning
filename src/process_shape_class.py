import os, sys, glob, shutil

def main(input_dir, input_cls_files):

    assert os.path.exists(input_dir), f'{input_dir} not exists'

    for input_cls_file in input_cls_files:
        
        target_paths = []
        target_paths = glob.glob(f'{input_dir}/train/*')
        target_paths.extend(glob.glob(f'{input_dir}/val/*'))
        target_paths.extend(glob.glob(f'{input_dir}/val_deform/*'))
        
        assert os.path.exists(input_cls_file), f'{input_cls_file} not exists'
        f_cls = open(input_cls_file, 'r')
        f_cls_content = f_cls.readlines()
        
        cls_dict = {}
        dict_door_open = False
        center_door_open = False
        centers = []
        for line in f_cls_content:
            
            if '==============================================================================================\n' == line:
                dict_door_open = not dict_door_open
                continue
            if '===============================================\n' == line:
                center_door_open = not center_door_open
                continue

            if dict_door_open:
                line_strip = line.strip()
                hook_name = line_strip.split('=>')[0].strip()
                val = line_strip.split('=>')[1].strip()
                cls_dict[hook_name] = val

            if center_door_open:
                line_strip = line.strip()
                hook_name = line_strip.split(':')[-1].strip()
                centers.append(hook_name)

            # key_split = key[:-len(key.split('_')[-1])]
            # cls_dict[key_split] = key.split('_')[-1]

        print('=============================================')
        
        for target_path in target_paths:
            hook_name = target_path.split('/')[-1]

            if hook_name not in cls_dict.keys():
                continue
            
            if hook_name in centers:
                cls_dict[hook_name] = f'{cls_dict[hook_name]}-center'
            # hook_name_split = hook_name[:-len(hook_name.split('_')[-1])]
            # if hook_name_split not in cls_dict.keys():
            #     continue
            # modified_target_path = '{}/{}{}'.format(os.path.split(target_path)[0], hook_name[:-len(hook_name.split('_')[-1])], cls_dict[hook_name_split])
            
            modified_target_path = '{}/{}{}'.format(os.path.split(target_path)[0], hook_name[:-len(hook_name.split('_')[-1])], cls_dict[hook_name])
            # print(f'{target_path} =>\n {modified_target_path}')
            os.rename(target_path, modified_target_path)


if __name__=="__main__":

    cat_num = 5
    input_dir = f'../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-40-k0/05.02.20.23-1000-singleview-{cat_num}c'
    # input_dir = f'../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-20-k0/05.02.20.39-1000-singleview-{cat_num}c'
    # input_dir = f'../dataset/traj_recon_affordance/kptraj_all_smooth-absolute-10-k0/05.02.20.53-1000-singleview-{cat_num}c'
    input_cls_files = [f'labels_{cat_num}.txt']

    main(input_dir, input_cls_files)