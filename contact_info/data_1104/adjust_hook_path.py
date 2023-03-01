import os, glob, json
from tqdm import tqdm

def main():
    json_files = glob.glob('*/*.json')
    origin_term = 'hook'
    fix_term = 'hook_1104'
    for json_file in tqdm(json_files):
        f = open(json_file, 'r')
        json_dict = json.load(f)
        f.close()

        if 'hook_path' in json_dict.keys():
            path_split_by_hook = json_dict['hook_path'].split(origin_term)
            if len(path_split_by_hook) == 2:

                true_path = path_split_by_hook[0] + fix_term + path_split_by_hook[1]
                json_dict['hook_path'] = true_path

                f = open(json_file, 'w')
                json_dict = json.dump(json_dict, f, indent=4)
                f.close()

if __name__=="__main__":
    main()