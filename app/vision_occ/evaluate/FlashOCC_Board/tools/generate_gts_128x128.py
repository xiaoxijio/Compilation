import numpy as np
import os
import random
import argparse
label_file_names = ['semantics', 'mask_lidar', 'mask_camera']

def main(args):
    scene_dirs = os.listdir(args.data_path)
    for scene in scene_dirs:
        scene_path = os.path.join(args.data_path, scene)
        token_dir = os.listdir(scene_path)
        for token in token_dir:
            label_path = os.path.join(scene_path, token, 'labels.npz')
            occ_label = np.load(label_path)
            save_path = os.path.join(args.save_path, scene, token)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez(os.path.join(save_path,'labels.npz'), \
                semantics=occ_label['semantics'][36:164, 36:164, :], \
                mask_lidar=occ_label['mask_lidar'][36:164, 36:164, :], \
                mask_camera=occ_label['mask_camera'][36:164, 36:164, :])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data/nuscenes/gts', help='t')
    parser.add_argument('--save-path', type=str, default='./data/nuscenes/gts_opt', help='t')
    args = parser.parse_args()
    main(args)
