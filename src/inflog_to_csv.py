import sys, os, glob
import numpy as np

def main(input_dir):
    assert os.path.exists(input_dir), f'{input_dir} not exists'

    input_txts = glob.glob(f'{input_dir}/*.txt')

    for input_txt in input_txts:

        f_input_txt = open(input_txt, 'r')
        input_logs = f_input_txt.readlines()

        match_token = '===============================================================================================\n'

        line_num = len(input_logs)
        i = 0

        log_dict = {
        }
        while i < line_num:

            if input_logs[i] == match_token:

                assert i + 8 < line_num and input_logs[i + 8] == match_token

                ckpt_line = input_logs[i + 1].strip()
                infd_line = input_logs[i + 2].strip()
                easy_line = input_logs[i + 3].strip()
                normal_line = input_logs[i + 4].strip()
                hard_line = input_logs[i + 5].strip()
                devil_line = input_logs[i + 6].strip()
                all_line = input_logs[i + 7].strip()

                assert 'checkpoint' in ckpt_line
                assert 'inference_dir' in infd_line
                assert 'easy' in easy_line
                assert 'normal' in normal_line
                assert 'hard' in hard_line
                assert 'devil' in devil_line
                assert 'all' in all_line

                config = ckpt_line.split('/')[1].split('-')[0]
                ckpt = ckpt_line.split(' ')[1][:-len(ckpt_line.split('/')[-1])]
                
                dataset = infd_line.split('/')[3]

                subset = infd_line.split('/')[-1]
                assert subset == 'val' or subset == 'train'

                easy_suc_rate = float(easy_line.split(' ')[-1][:-1]) # ignore %
                normal_suc_rate = float(normal_line.split(' ')[-1][:-1]) # ignore %
                hard_suc_rate = float(hard_line.split(' ')[-1][:-1]) # ignore %
                devil_suc_rate = float(devil_line.split(' ')[-1][:-1]) # ignore %
                all_suc_rate = float(all_line.split(' ')[-1][:-1]) # ignore %

                if ckpt not in log_dict.keys():
                    log_dict[ckpt] = {
                        'config': config,
                        'dataset': dataset,
                    }

                if subset in log_dict[ckpt].keys():
                    
                    log_dict[ckpt][subset]['all'].append(all_suc_rate)
                    log_dict[ckpt][subset]['easy'].append(easy_suc_rate)
                    log_dict[ckpt][subset]['normal'].append(normal_suc_rate)
                    log_dict[ckpt][subset]['hard'].append(hard_suc_rate)
                    log_dict[ckpt][subset]['devil'].append(devil_suc_rate)
                
                else :

                    log_dict[ckpt][subset] = {
                        'all': [all_suc_rate],
                        'easy': [easy_suc_rate],
                        'normal': [normal_suc_rate],
                        'hard': [hard_suc_rate],
                        'devil': [devil_suc_rate]
                    }

                i += 9

            i += 1

        f_out = open('{}.csv'.format(input_txt.split('.')[0]), 'w')
        for ckpt in log_dict.keys():
            if '{}.csv'.format(input_txt.split('.')[0]) == 'inference/inference_logs/traj_fusion_mutual_10_samp.csv':
                print(ckpt)
            config = log_dict[ckpt]['config']
            dataset = log_dict[ckpt]['dataset']
            all_item = '{:00.03f}% / {:00.03f}%'.format(
                            np.max(log_dict[ckpt]['train']['all']) if 'train' in log_dict[ckpt].keys() else -1, 
                            np.max(log_dict[ckpt]['val']['all']) if 'val' in log_dict[ckpt].keys() else -1
                        )
            easy_item = '{:00.03f}% / {:00.03f}%'.format(
                            np.max(log_dict[ckpt]['train']['easy']) if 'train' in log_dict[ckpt].keys() else -1, 
                            np.max(log_dict[ckpt]['val']['easy']) if 'val' in log_dict[ckpt].keys() else -1
                        )
            normal_item = '{:00.03f}% / {:00.03f}%'.format(
                            np.max(log_dict[ckpt]['train']['normal']) if 'train' in log_dict[ckpt].keys() else -1, 
                            np.max(log_dict[ckpt]['val']['normal']) if 'val' in log_dict[ckpt].keys() else -1
                        )
            hard_item = '{:00.03f}% / {:00.03f}%'.format(
                            np.max(log_dict[ckpt]['train']['hard']) if 'train' in log_dict[ckpt].keys() else -1, 
                            np.max(log_dict[ckpt]['val']['hard']) if 'val' in log_dict[ckpt].keys() else -1
                        )
            devil_item = '{:00.03f}% / {:00.03f}%'.format(
                            np.max(log_dict[ckpt]['train']['devil']) if 'train' in log_dict[ckpt].keys() else -1, 
                            np.max(log_dict[ckpt]['val']['devil']) if 'val' in log_dict[ckpt].keys() else -1
                        )

            csv_line = f'{config}, {dataset}, {all_item}, {easy_item}, {normal_item}, {hard_item}, {devil_item}\n'
            f_out.write(csv_line)
        f_out.close()

if __name__=="__main__":

    if len(sys.argv) < 2:
        print('please specify input log file')
        exit(-1)

    input_dir = sys.argv[1]
    main(input_dir)