import numpy as np

log_file = "inference/inference_logs/traj_af_all_0317.txt"

f = open(log_file, 'r')
lines = f.readlines()

opened = False
out_list = []

i = 0
while i < len(lines):
    if '===============' in lines[i]:
        opened = not opened

        if opened:
            ckpt = lines[i+1].split('/')[1]
            subset = lines[i+2].split('/')[-1]
            sucrate = float(lines[i+3].split(' ')[-1])
            one_dict = {
                'ckpt': ckpt.strip(),
                'subset': subset.strip(),
                'sucrate': '{:00.01f}'.format(100 * sucrate)
            }
            out_list.append(one_dict)
    
    i += 1

fout = open('sucrate.csv', 'w')
for i in range(0, len(out_list), 2):
    first = out_list[i]
    second = out_list[i+1]

    out_line = '{} / {},{}% / {}%, {}\n'.format(first['subset'], second['subset'], first['sucrate'], second['sucrate'], first['ckpt'])
    fout.write(out_line)
fout.close()

            

