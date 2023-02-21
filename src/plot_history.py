import sys, os , glob
import numpy as np
import matplotlib.pyplot as plt

def main(root_name : str):

    fnames = [root_name]
    if os.path.isdir(root_name):
        fnames = glob.glob(f'{root_name}/*/*_log.txt')

    for fname in fnames:
        f = open(fname, 'r')
        lines = f.readlines()

        training_model_type = None
        if '_ae' in fname:
            training_model_type = 'ae'
        elif '_vae' in fname:
            training_model_type = 'vae'
        elif '_cae' in fname:
            training_model_type = 'cae'
        elif '_cvae' in fname:
            training_model_type = 'cvae'
        elif 'affordance' in fname:
            training_model_type = 'affordance'
        else :
            print('the training type of the log file is not specified')
            exit(0)

        training_epoch = []
        training_res = None
        validation_epoch = []
        validation_res = None

        if training_model_type == 'ae' or training_model_type == 'cae':
            training_res = {
                'total_loss': []
            }
            validation_res = {
                'total_loss': []
            }

        if training_model_type == 'vae' or training_model_type == 'cvae':
            training_res = {
                'kl_loss': [],
                'recon_loss': [],
                'total_loss': []
            }
            validation_res = {
                'kl_loss': [],
                'recon_loss': [],
                'total_loss': []
            }

        if training_model_type == 'affordance':
            training_res = {
                'total_loss': []
            }
            validation_res = {
                'total_loss': []
            }

        i = 0
        for i, line in enumerate(lines):
            if 'training stage' in line:

                if training_model_type == 'ae' or training_model_type == 'cae':
                    time_line = lines[i + 1]
                    epoch_line = lines[i + 2]
                    lr_line = lines[i + 3]
                    total_liss_line = lines[i + 4]

                    training_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                    training_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))
                
                if training_model_type == 'vae' or training_model_type == 'cvae':
                    time_line = lines[i + 1]
                    epoch_line = lines[i + 2]
                    lr_line = lines[i + 3]
                    kl_loss_line = lines[i + 4]
                    recon_loss_line = lines[i + 5]
                    total_liss_line = lines[i + 6]

                    training_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                    training_res['kl_loss'].append(float(kl_loss_line.split(':')[-1].strip()))
                    training_res['recon_loss'].append(float(recon_loss_line.split(':')[-1].strip()))
                    training_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))
                
                if training_model_type == 'affordance':
                    time_line = lines[i + 1]
                    epoch_line = lines[i + 2]
                    lr_line = lines[i + 3]
                    total_liss_line = lines[i + 4]

                    training_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                    training_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))

            if 'validation stage' in line:
                if training_model_type == 'ae' or training_model_type == 'cae':
                    time_line = lines[i + 1]
                    epoch_line = lines[i + 2]
                    lr_line = lines[i + 3]
                    total_liss_line = lines[i + 4]

                    validation_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                    validation_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))
                
                if training_model_type == 'vae' or training_model_type == 'cvae':
                    time_line = lines[i + 1]
                    epoch_line = lines[i + 2]
                    lr_line = lines[i + 3]
                    kl_loss_line = lines[i + 4]
                    recon_loss_line = lines[i + 5]
                    total_liss_line = lines[i + 6]

                    validation_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                    validation_res['kl_loss'].append(float(kl_loss_line.split(':')[-1].strip()))
                    validation_res['recon_loss'].append(float(recon_loss_line.split(':')[-1].strip()))
                    validation_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))

                if training_model_type == 'affordance':
                    time_line = lines[i + 1]
                    epoch_line = lines[i + 2]
                    lr_line = lines[i + 3]
                    total_liss_line = lines[i + 4]

                    validation_epoch.append(int(epoch_line.split('/')[0].split(' ')[-1]))
                    validation_res['total_loss'].append(float(total_liss_line.split(':')[-1].strip()))
        
        fname_1 = fname.split('/')[1]
        fname_2 = fname.split('/')[2][:-4]

        output_dir = f'figures/{fname_1}/{fname_2}'
        os.makedirs(output_dir, exist_ok=True)
        
        # total only
        plt.figure(figsize=(10, 5))
        plt.plot(training_res['total_loss'], label=f'train_total_loss', zorder=2)
        plt.plot(validation_res['total_loss'], label=f'val_total_loss', zorder=2)
        plt.title('Training History (Total Loss)')
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('loss', fontsize=16)
        plt.yscale("log")
        plt.legend()
        plt.grid()
        plt.savefig(f'{output_dir}/total.png')
        plt.close()
        
        if training_model_type == 'vae' or training_model_type == 'cvae':
            # all
            plt.figure(figsize=(10, 5))
            for key in training_res.keys():
                plt.plot(training_res[key], label=f'train_{key}', zorder=2)
            for key in training_res.keys():
                plt.plot(validation_res[key], label=f'val_{key}', zorder=2)
            plt.title('Training History (All Loss)')
            plt.xlabel('epoch', fontsize=16)
            plt.ylabel('loss', fontsize=16)
            plt.yscale("log")
            plt.legend()
            plt.grid()
            plt.savefig(f'{output_dir}/all.png')
            plt.close()

if __name__=="__main__":
    if len(sys.argv) < 2:
        print('no input file specified')
        exit(-1)
    root_name = sys.argv[1]
    main(root_name)