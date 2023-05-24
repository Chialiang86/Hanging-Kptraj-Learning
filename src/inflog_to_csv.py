import sys, os, glob
import numpy as np

def main(input_dir):
    assert os.path.exists(input_dir), f'{input_dir} not exists'

    input_txts = glob.glob(f'{input_dir}/traj*.txt')

    for input_txt in input_txts:

        f_input_txt = open(input_txt, 'r')
        input_logs = f_input_txt.readlines()

        match_token = '===============================================================================================\n'

        line_num = len(input_logs)
        i = 0

        log_dict = {
        }
        while i < line_num:

            if input_logs[i] == match_token and i + 8 < line_num and input_logs[i + 8] == match_token:

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
                assert subset == 'test' or subset == 'test_deform' or subset == 'val' or subset == 'val_deform' or subset == 'train', f'subset should be train or val or val_deform, but got {subset}'

                easy_suc_rate = float(easy_line.split(' ')[-1][:-1]) # ignore %
                normal_suc_rate = float(normal_line.split(' ')[-1][:-1]) # ignore %
                hard_suc_rate = float(hard_line.split(' ')[-1][:-1]) # ignore %
                devil_suc_rate = float(devil_line.split(' ')[-1][:-1]) # ignore %
                all_suc_rate = float(all_line.split(' ')[-1][:-1]) # ignore %
                ckpt_num = int(ckpt_line.split('/')[-1].split('-')[-1].split('.')[0])

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
                    log_dict[ckpt][subset]['devil'].append(devil_suc_rate)
                    log_dict[ckpt][subset]['ckpt_num'].append(ckpt_num)
                
                else :

                    log_dict[ckpt][subset] = {
                        'all': [all_suc_rate],
                        'easy': [easy_suc_rate],
                        'normal': [normal_suc_rate],
                        'hard': [hard_suc_rate],
                        'devil': [devil_suc_rate],
                        'ckpt_num': [ckpt_num]
                    }

                i += 9

            i += 1

        f_out = open('{}.csv'.format(input_txt.split('.')[0]), 'w')
        for ckpt in log_dict.keys():
            config = log_dict[ckpt]['config']
            dataset = log_dict[ckpt]['dataset']

            max_train_index = np.argmax(log_dict[ckpt]['train']['all']) if 'train' in log_dict[ckpt].keys() else -1,
            max_val_index = np.argmax(log_dict[ckpt]['val']['all']) if 'val' in log_dict[ckpt].keys() else \
                            np.argmax(log_dict[ckpt]['val_deform']['all']) if 'val_deform' in log_dict[ckpt].keys() else \
                            np.argmax(log_dict[ckpt]['test']['all']) if 'test' in log_dict[ckpt].keys() else \
                            np.argmax(log_dict[ckpt]['test_deform']['all']) if 'test_deform' in log_dict[ckpt].keys() else -1
            ckpt_num = log_dict[ckpt]['val']['ckpt_num'][max_val_index] if 'val' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['val_deform']['ckpt_num'][max_val_index] if 'val_deform' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test']['ckpt_num'][max_val_index] if 'test' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test_deform']['ckpt_num'][max_val_index] if 'test_deform' in log_dict[ckpt].keys() else -1
            
            all_item = '{:00.03f}%'.format(
            # all_item = '{:00.03f}% / {:00.03f}%'.format(
                            # log_dict[ckpt]['train']['all'][max_train_index] if 'train' in log_dict[ckpt].keys() else -1,
                            log_dict[ckpt]['val']['all'][max_val_index] if 'val' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['val_deform']['all'][max_val_index] if 'val_deform' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test']['all'][max_val_index] if 'test' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test_deform']['all'][max_val_index] if 'test_deform' in log_dict[ckpt].keys() else -1
                        )
            easy_item = '{:00.03f}%'.format(
            # easy_item = '{:00.03f}% / {:00.03f}%'.format(
                            # log_dict[ckpt]['train']['easy'][max_train_index] if 'train' in log_dict[ckpt].keys() else -1,
                            log_dict[ckpt]['val']['easy'][max_val_index] if 'val' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['val_deform']['easy'][max_val_index] if 'val_deform' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test']['easy'][max_val_index] if 'test' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test_deform']['easy'][max_val_index] if 'test_deform' in log_dict[ckpt].keys() else -1
                        )
            normal_item = '{:00.03f}%'.format(
            # normal_item = '{:00.03f}% / {:00.03f}%'.format(
                            # log_dict[ckpt]['train']['normal'][max_train_index] if 'train' in log_dict[ckpt].keys() else -1,
                            log_dict[ckpt]['val']['normal'][max_val_index] if 'val' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['val_deform']['normal'][max_val_index] if 'val_deform' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test']['normal'][max_val_index] if 'test' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test_deform']['normal'][max_val_index] if 'test_deform' in log_dict[ckpt].keys() else -1
                        )
            hard_item = '{:00.03f}%'.format(
            # hard_item = '{:00.03f}% / {:00.03f}%'.format(
                            # log_dict[ckpt]['train']['hard'][max_train_index] if 'train' in log_dict[ckpt].keys() else -1,
                            log_dict[ckpt]['val']['hard'][max_val_index] if 'val' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['val_deform']['hard'][max_val_index] if 'val_deform' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test']['hard'][max_val_index] if 'test' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test_deform']['hard'][max_val_index] if 'test_deform' in log_dict[ckpt].keys() else -1
                        )
            devil_item = '{:00.03f}%'.format(
            # devil_item = '{:00.03f}% / {:00.03f}%'.format(
                            # log_dict[ckpt]['train']['devil'][max_train_index] if 'train' in log_dict[ckpt].keys() else -1,
                            log_dict[ckpt]['val']['devil'][max_val_index] if 'val' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['val_deform']['devil'][max_val_index] if 'val_deform' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test']['devil'][max_val_index] if 'test' in log_dict[ckpt].keys() else \
                            log_dict[ckpt]['test_deform']['devil'][max_val_index] if 'test_deform' in log_dict[ckpt].keys() else -1
                        )
            # all_item = '{:00.03f}%'.format(
            #                 np.max(log_dict[ckpt]['val']['all']) if 'val' in log_dict[ckpt].keys() else -1
            #             )
            # easy_item = '{:00.03f}%'.format(
            #                 np.max(log_dict[ckpt]['val']['easy']) if 'val' in log_dict[ckpt].keys() else -1
            #             )
            # normal_item = '{:00.03f}%'.format(
            #                 np.max(log_dict[ckpt]['val']['normal']) if 'val' in log_dict[ckpt].keys() else -1
            #             )
            # hard_item = '{:00.03f}%'.format(
            #                 np.max(log_dict[ckpt]['val']['hard']) if 'val' in log_dict[ckpt].keys() else -1
            #             )
            # devil_item = '{:00.03f}%'.format(
            #                 np.max(log_dict[ckpt]['val']['devil']) if 'val' in log_dict[ckpt].keys() else -1
            #             )

            csv_line = f'{config}, {dataset}, {all_item}, {easy_item}, {normal_item}, {hard_item}, {devil_item}, {ckpt_num}\n'
            f_out.write(csv_line)
        f_out.close()

if __name__=="__main__":

    if len(sys.argv) < 2:
        print('please specify input log file')
        exit(-1)

    input_dir = sys.argv[1]
    main(input_dir)