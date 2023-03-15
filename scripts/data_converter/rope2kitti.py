import os
import sys
import argparse
import json
import csv
import cv2

import numpy as np
from tqdm import tqdm
from shutil import copyfile

category_map = {'car': 'Car', 'van': 'Car', 'truck': 'Bus', 'bus': 'Bus', 'pedestrian': 'Pedestrian', 'cyclist': 'Cyclist', 'motorcyclist': 'Cyclist', 'tricyclist': 'Cyclist'}

def parse_option():
    parser = argparse.ArgumentParser('Convert rope3D dataset to standard kitti format', add_help=False)
    parser.add_argument('--source-root', type=str, default="data/rope3d", help='root path to rope3d dataset')
    parser.add_argument('--target-root', type=str, default="data/rope3d-kitti", help='root path to rope3d dataset in kitti format')
    args = parser.parse_args()
    return args
    
def copy_file(file_src, file_dest):
    if not os.path.exists(file_dest):
        try:
            copyfile(file_src, file_dest)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)

def convert_calib(src_calib_file, dest_calib_file):
    with open(src_calib_file) as f:
        lines = f.readlines()
    obj = lines[0].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    kitti_calib = dict()
    kitti_calib["P0"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P1"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["P2"] = P2  # Left camera transform.
    kitti_calib["P3"] = np.zeros((3, 4))  # Dummy values.
    # Cameras are already rectified.
    kitti_calib["R0_rect"] = np.identity(3)
    kitti_calib["Tr_velo_to_cam"] = np.zeros((3, 4))  # Dummy values.
    kitti_calib["Tr_imu_to_velo"] = np.zeros((3, 4))  # Dummy values.
    
    with open(dest_calib_file, "w") as calib_file:
        for (key, val) in kitti_calib.items():
            val = val.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            calib_file.write("%s: %s\n" % (key, val_str))

def ry2alpha(ry, pos):
    alpha = ry - np.arctan2(pos[0], pos[2])
    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha

def alpha2roty(alpha, pos):
    ry = alpha + np.arctan2(pos[0], pos[2])
    if ry > np.pi:
        ry -= 2 * np.pi
    if ry < -np.pi:
        ry += 2 * np.pi
    return ry
 
def convert_label(src_label_file, dest_label_file):
    with open(src_label_file, 'r') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        label = line.strip().split(' ')
        cls_type = label[0]
        if cls_type not in category_map.keys(): continue
        label[0] = category_map[cls_type]
        
        truncated = int(label[1])
        if truncated > 0:
            truncated = 0.0
        label[1] = str(truncated)
        alpha = float(label[3])
        pos = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        if np.sum(pos) < 10e-9: continue
        ry = float(label[14])
        if alpha > np.pi:
            alpha -= 2 * np.pi
            ry = alpha2roty(alpha, pos)
        label[3] = str(alpha) 
        label[14] = str(ry)
        new_lines.append(' '.join(label))
        
    with open(dest_label_file,'w') as f:
        for line in new_lines:
            f.write(line)
            f.write("\n")
            
def load_boxes(label_file):
    boxes = []
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                  'dl', 'lx', 'ly', 'lz', 'ry']
    with open(label_file, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            box = [float(row["xmin"]), float(row["ymin"]), float(row["xmax"]), float(row["ymax"])]
            boxes.append(box)
    return boxes
            
def main(src_root, dest_root, split='train', img_id=0):
    if split == 'train':
        src_dir = os.path.join(src_root, "training")
        img_path_list = ["training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d"]
    else:
        src_dir = os.path.join(src_root, "validation")
        img_path_list = ["validation-image_2"]
        
    os.makedirs(dest_root, exist_ok=True)
    os.makedirs(os.path.join(dest_root, "training/image_2"), exist_ok=True)
    os.makedirs(os.path.join(dest_root, "training/label_2"), exist_ok=True)
    os.makedirs(os.path.join(dest_root, "training/calib"), exist_ok=True)
    os.makedirs(os.path.join(dest_root, "training/denorm"), exist_ok=True)

    src_label_path = os.path.join(src_dir, "label_2")
    src_calib_path = os.path.join(src_dir, "calib")
    src_denorm_path = os.path.join(src_dir, "denorm")
    
    split_txt = os.path.join(src_dir, "train.txt" if split=='train' else 'val.txt')
    idx_list = [x.strip() for x in open(split_txt).readlines()]
    idx_list_valid = []
    for index in idx_list:
        for img_path in img_path_list:
            src_img_path = os.path.join(src_dir, "../", img_path)
            img_file = os.path.join(src_img_path, index + ".jpg")
            if os.path.exists(img_file):
                idx_list_valid.append(index)
                break
    img_id_list = []
    map_token2id = dict()
    for index in tqdm(idx_list_valid):
        for img_path in img_path_list:
            src_img_path = os.path.join(src_dir, "../", img_path)
            img_file = os.path.join(src_img_path, index + ".jpg")
            if os.path.exists(img_file):
                src_img_file = img_file
                break
        src_label_file = os.path.join(src_label_path, index + ".txt")
        src_calib_file = os.path.join(src_calib_path, index + ".txt")
        src_denorm_file = os.path.join(src_denorm_path, index + ".txt")
    
        dest_img_file = os.path.join(dest_root, "training/image_2", '{:06d}.png'.format(img_id))
        dest_label_file = os.path.join(dest_root, "training/label_2", '{:06d}.txt'.format(img_id))
        dest_calib_file = os.path.join(dest_root, "training/calib", '{:06d}.txt'.format(img_id))
        dest_denorm_file = os.path.join(dest_root, "training/denorm", '{:06d}.txt'.format(img_id))
        
        map_token2id[index] = "{:06d}".format(img_id)
        img_id_list.append(img_id)
        
        # image_2
        img = cv2.imread(src_img_file)
        cv2.imwrite(dest_img_file, img)
        # calib
        convert_calib(src_calib_file, dest_calib_file)
        # label
        convert_label(src_label_file, dest_label_file)
        # denorm
        copy_file(src_denorm_file, dest_denorm_file)
        img_id = img_id + 1
    return map_token2id, img_id

if __name__ == "__main__":
    args = parse_option()
    source_root, target_root = args.source_root, args.target_root
    map_token2id, img_id = main(source_root, target_root, 'train')
    map, img_id = main(source_root, target_root, 'val', img_id)
    map_token2id.update(map)
    with open(os.path.join(target_root, 'map_token2id.json'), 'w') as file:
        json.dump(map_token2id, file)
