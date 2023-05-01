import argparse
import open3d as o3d
import imageio
import os
import glob
import cv2
import copy
import numpy as np

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def capture_from_viewer(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)

    # Updates
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    o3d_screenshot_mat = vis.capture_screen_float_buffer(do_render=True) # need to be true to capture the image
    o3d_screenshot_mat = (255.0 * np.asarray(o3d_screenshot_mat)).astype(np.uint8)
    o3d_screenshot_mat = cv2.cvtColor(o3d_screenshot_mat,cv2.COLOR_BGR2RGB)
    o3d_screenshot_mat = cv2.resize(o3d_screenshot_mat, (o3d_screenshot_mat.shape[1] // 6, o3d_screenshot_mat.shape[0] // 6))
    vis.destroy_window()

    return o3d_screenshot_mat

train_list = [
                'Hook_hs_252_hard', 'Hook_hcu_108_easy', 'Hook_hs_236_easy', 'Hook_hsr_27_devil', 'Hook_hsr_350_easy', 
                'Hook_hcu_227_easy', 'Hook_omni_2_easy', 'Hook_hcu_176_normal', 'Hook_hs_350_normal', 'Hook_hcu_340_normal', 
                'Hook_hcu_180_devil', 'Hook_my_45_hard', 'Hook_hsr_44_normal', 'Hook_hs_50_normal', 'Hook_hcu_137_normal', 
                'Hook_hcu_205_easy', 'Hook_hsr_151_easy', 'Hook_hs_137_hard', 'Hook_hsr_323_easy', 'Hook_hs_315_easy', 
                'Hook_hcu_356_hard', 'Hook_hs_291_hard', 'Hook_hsr_205_hard', 'Hook_hsr_268_normal', 'Hook_omni_47_normal', 
                'Hook_hsr_286_hard', 'Hook_omni_44_devil', 'Hook_hs_304_hard', 'Hook_hsr_238_hard', 'Hook_hcu_347_easy', 
                'Hook_hs_46_hard', 'Hook_hs_230_hard', 'Hook_hsr_114_devil', 'Hook_hsr_143_hard', 'Hook_hs_338_easy', 
                'Hook_omni_1_devil', 'Hook_hsr_215_easy', 'Hook_hs_235_normal', 'Hook_hcu_207_easy', 'Hook_hsr_260_normal', 
                'Hook_hsr_165_hard', 'Hook_hcu_198_easy', 'Hook_hs_48_easy', 'Hook_hcu_38_normal', 'Hook_hsr_259_hard', 
                'Hook_hcu_249_hard', 'Hook_hsr_207_normal', 'Hook_hs_294_devil', 'Hook_my_90_devil', 'Hook_hs_331_normal', 
                'Hook_hs_251_normal', 'Hook_hsr_34_devil', 'Hook_hcu_158_normal', 'Hook_hs_92_devil', 'Hook_hsr_231_devil', 
                'Hook_hs_232_devil', 'Hook_hs_174_devil', 'Hook_hs_321_devil', 'Hook_hsr_258_devil', 'Hook_hcu_189_devil'
            ]

val_list = [
                'Hook_hcu_104_devil', 'Hook_hcu_134_normal', 'Hook_hcu_138_hard', 'Hook_hcu_181_hard', 'Hook_hcu_190_easy', 
                'Hook_hcu_243_normal', 'Hook_hcu_279_devil', 'Hook_hcu_293_easy', 'Hook_hcu_296_normal', 'Hook_hcu_303_normal', 
                'Hook_hcu_306_normal', 'Hook_hcu_335_normal', 'Hook_hcu_359_easy', 'Hook_hcu_362_easy', 'Hook_hcu_364_normal', 
                'Hook_hcu_376_devil', 'Hook_hcu_380_hard', 'Hook_hcu_390_easy', 'Hook_hcu_3_hard', 'Hook_hcu_75_easy', 
                'Hook_hcu_89_devil', 'Hook_hs_105_hard', 'Hook_hs_117_hard', 'Hook_hs_154_hard', 'Hook_hs_156_hard', 
                'Hook_hs_190_easy', 'Hook_hs_216_easy', 'Hook_hs_229_normal', 'Hook_hs_275_devil', 'Hook_hs_293_normal', 
                'Hook_hs_314_easy', 'Hook_hs_317_easy', 'Hook_hs_339_devil', 'Hook_hs_363_devil', 'Hook_hs_370_easy', 
                'Hook_hs_393_devil', 'Hook_hs_42_hard', 'Hook_hs_70_easy', 'Hook_hs_94_hard', 'Hook_hs_95_devil', 
                'Hook_hsr_118_hard', 'Hook_hsr_123_normal', 'Hook_hsr_125_easy', 'Hook_hsr_13_devil', 'Hook_hsr_15_normal', 
                'Hook_hsr_218_devil', 'Hook_hsr_22_normal', 'Hook_hsr_263_hard', 'Hook_hsr_298_normal', 'Hook_hsr_304_hard', 
                'Hook_hsr_312_devil', 'Hook_hsr_321_devil', 'Hook_hsr_335_hard', 'Hook_hsr_371_devil', 'Hook_hsr_381_easy', 
                'Hook_hsr_391_hard', 'Hook_hsr_56_normal', 'Hook_hsr_5_normal', 'Hook_hsr_71_easy', 'Hook_omni_124_devil'
            ]

def main(args):

    input_dir = args.input_dir
    assert os.path.exists(input_dir), f'{input_dir} not exists'

    sub_dir = input_dir.split('/')[-1]
    out_dir = f'visualization/hanging_objects_train/{sub_dir}'
    os.makedirs(out_dir, exist_ok=True)

    # frames for visualization
    obj_files = glob.glob(f'{input_dir}/*/base.obj')
    obj_files.sort()
    frames = 18
    rotate_per_frame = (2 * np.pi) / frames

    for obj_file in tqdm(obj_files):

        hook_name = obj_file.split('/')[-2]
        if hook_name not in train_list:
            continue

        mesh = o3d.io.read_triangle_mesh(obj_file)
        mesh_90 = copy.deepcopy(mesh)
        r_90 = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi / 2)) # (rx, ry, rz) = (right, up, inner)
        mesh_90.rotate(r_90, center=(0, 0, 0))

        obj_dir = os.path.split(obj_file)[0].split('/')[-1]

        img_list = []
        # img_list_90 = []
        for _ in range(frames):
            r = mesh.get_rotation_matrix_from_xyz((0, rotate_per_frame, 0)) # (rx, ry, rz) = (right, up, inner)
            mesh.rotate(r, center=(0, 0, 0))
            # mesh_90.rotate(r, center=(0, 0, 0))

            img = capture_from_viewer(mesh)
            # img_90 = capture_from_viewer(mesh_90)
            img_list.append(img)
            # img_list_90.append(img_90)
        
        save_path = f'{out_dir}/{sub_dir}_{obj_dir}-0.gif'
        # save_path_90 = f'{out_dir}/{sub_dir}_{obj_dir}-90.gif'
        imageio.mimsave(save_path, img_list, fps=10)
        # imageio.mimsave(save_path_90, img_list_90, fps=20)
        # print(f'{save_path} and {save_path_90} saved')

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-id', type=str, default='shapes/hook_all_new')
    args = parser.parse_args()

    main(args)