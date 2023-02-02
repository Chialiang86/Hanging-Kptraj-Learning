import glob, os, shutil
from tqdm import tqdm

def main():
    traj_paths = glob.glob(f'data/*/*/*/*/*/base.json')
    for traj_path in tqdm(traj_paths):
        traj_new_path = f'{traj_path[:-9]}traj.json'
        os.rename(traj_path, traj_new_path)

    ply_paths = glob.glob(f'data/*/*/*/*/*/base.ply')
    for ply_path in tqdm(ply_paths):
        ply_new_path = f'{ply_path[:-8]}shape.ply'
        os.rename(ply_path, ply_new_path)

if __name__=="__main__":
    main()