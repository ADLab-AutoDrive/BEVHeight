import os
import csv
import math
import random
import cv2

import mmcv
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

name2nuscenceclass = {
    "car": "vehicle.car",
    "van": "vehicle.car",
    "truck": "vehicle.truck",
    "bus": "vehicle.bus.rigid",
    "cyclist": "vehicle.bicycle",
    "tricyclist": "vehicle.trailer",
    "motorcyclist": "vehicle.motorcycle",
    "pedestrian": "human.pedestrian.adult",
    "trafficcone": "movable_object.trafficcone",
}

def alpha2roty(alpha, pos):
    ry = alpha + np.arctan2(pos[0], pos[2])
    if ry > np.pi:
        ry -= 2 * np.pi
    if ry < -np.pi:
        ry += 2 * np.pi
    return ry

def clip2pi(ry):
    if ry > 2 * np.pi:
        ry -= 2 * np.pi
    if ry < - 2* np.pi:
        ry += 2 * np.pi
    return ry

def load_calib(calib_file):
    with open(calib_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P2 = row[1:]
                P2 = [float(i) for i in P2]
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                continue
    return P2[:3,:3]

def load_denorm(denorm_file):
    with open(denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

def get_cam2lidar(denorm_file):
    denorm = load_denorm(denorm_file)
    Rx = np.array([[1.0, 0.0, 0.0], 
                   [0.0, 0.0, 1.0], 
                   [0.0, -1.0, 0.0]])
    
    Rz = np.array([[0.0, 1.0, 0.0], 
                   [-1.0, 0.0, 0.0],  
                   [0.0, 0.0, 1.0]])
    
    origin_vector = np.array([0, 1, 0])
    target_vector = -1 * np.array([denorm[0], denorm[1], denorm[2]])
    target_vector_norm = target_vector / np.sqrt(target_vector[0]**2 + target_vector[1]**2 + target_vector[2]**2)       
    sita = math.acos(np.inner(target_vector_norm, origin_vector))
    n_vector = np.cross(target_vector_norm, origin_vector) 
    n_vector = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    n_vector = n_vector.astype(np.float32)
    cam2lidar, _ = cv2.Rodrigues(n_vector * sita)
    cam2lidar = cam2lidar.astype(np.float32)
    cam2lidar = np.matmul(Rx, cam2lidar)
    cam2lidar = np.matmul(Rz, cam2lidar)
    
    Ax, By, Cz, D = denorm[0], denorm[1], denorm[2], denorm[3]
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(D) / mod_area
    Tr_cam2lidar = np.eye(4)
    Tr_cam2lidar[:3, :3] = cam2lidar
    Tr_cam2lidar[:3, 3] = [0, 0, d]
    
    translation = [0, 0, d]
    return cam2lidar, translation, Tr_cam2lidar, denorm

def get_annos(label_path, Tr_cam2lidar):
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']
    annos = []
    with open(label_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            if row["type"] in name2nuscenceclass.keys():
                alpha = float(row["alpha"])
                pos = np.array((float(row['lx']), float(row['ly']), float(row['lz'])), dtype=np.float32)
                ry = float(row["ry"])
                if alpha > np.pi:
                    alpha -= 2 * np.pi
                    ry = alpha2roty(alpha, pos)
                alpha = clip2pi(alpha)
                ry = clip2pi(ry)
                rotation =  0.5 * np.pi - ry
                
                name = name2nuscenceclass[row["type"]]
                dim = [float(row['dl']), float(row['dw']), float(row['dh'])]
                box2d = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                truncated_state = int(row["truncated"])
                occluded_state = int(row["occluded"])
                if sum(dim) == 0:
                    continue
                loc_cam = np.array([float(row['lx']), float(row['ly']), float(row['lz']), 1.0]).reshape(4, 1)
                loc_lidar = np.matmul(Tr_cam2lidar, loc_cam).squeeze(-1)[:3]
                loc_lidar[2] += 0.5 * float(row['dh'])
                anno = {"dim": dim, "loc": loc_lidar, "rotation": rotation, "name": name, "box2d": box2d, "truncated_state": truncated_state, "occluded_state": occluded_state}
                annos.append(anno)
    return annos

def generate_info_rope3d(rope3d_root, split='train', img_id=0):
    if split == 'train':
        src_dir = os.path.join(rope3d_root, "training")
        img_path = ["training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d"]
    else:
        src_dir = os.path.join(rope3d_root, "validation")
        img_path = ["validation-image_2"]
    label_path = os.path.join(src_dir, "label_2")
    calib_path = os.path.join(src_dir, "calib")
    denorm_path = os.path.join(src_dir, "denorm")
    split_txt = os.path.join(src_dir, "train.txt" if split=='train' else 'val.txt')
    idx_list = [x.strip() for x in open(split_txt).readlines()]
    idx_list_valid = []
    
    infos = list()
    for index in idx_list:
        for sub_img_path in img_path:
            img_file = os.path.join(rope3d_root, sub_img_path, index + ".jpg")
            if os.path.exists(img_file):
                idx_list_valid.append((sub_img_path, index))
                break
            
    for idx in tqdm(range(len(idx_list_valid))):
        sub_img_path, index = idx_list_valid[idx]
        img_file = os.path.join(sub_img_path, index + ".jpg")
        label_file = os.path.join(label_path, index + ".txt")
        calib_file = os.path.join(calib_path, index + ".txt")
        denorm_file = os.path.join(denorm_path, index + ".txt")
        
        info = dict()
        cam_info = dict()
        info['sample_token'] = index
        info['timestamp'] = 1000000
        info['scene_token'] = index
        
        cam_names = ['CAM_FRONT']
        lidar_names = ['LIDAR_TOP']
        cam_infos, lidar_infos = dict(), dict()
        for cam_name in cam_names:
            cam_info = dict()
            cam_info['sample_token'] = index
            cam_info['timestamp'] = 1000000
            cam_info['is_key_frame'] = True
            cam_info['height'] = 1080
            cam_info['width'] = 1920
            cam_info['filename'] = img_file 
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": index, "timestamp": 1000000}
            cam_info['ego_pose'] = ego_pose
            
            camera_intrinsic = load_calib(calib_file)
            cam2lidar, translation, Tr_cam2lidar, denorm = get_cam2lidar(denorm_file)
            
            calibrated_sensor = {"token": index, "sensor_token": index, "translation": translation, "rotation_matrix": cam2lidar, "camera_intrinsic": camera_intrinsic}
            cam_info['calibrated_sensor'] = calibrated_sensor
            cam_info['denorm'] = denorm
            cam_infos[cam_name] = cam_info
            
        for lidar_name in lidar_names:
            lidar_info = dict()
            lidar_infos[lidar_name] = lidar_info

        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        info['sweeps'] = list()
        ann_infos = list()
        annos = get_annos(label_file, Tr_cam2lidar)
        for anno in annos:
            category_name = anno["name"]
            translation = anno["loc"]
            size = anno["dim"]
            yaw_lidar = anno["rotation"]
            rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                                [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                                [0, 0, 1]])    
            rotation = Quaternion(matrix=rot_mat)
            ann_info = dict()
            ann_info["category_name"] = category_name
            ann_info["translation"] = translation
            ann_info["rotation"] = rotation
            ann_info["size"] = size
            ann_info["prev"] = ""
            ann_info["next"] = ""
            ann_info["sample_token"] = index
            ann_info["instance_token"] = index
            ann_info["token"] = index
            ann_info["visibility_token"] = str(anno["occluded_state"])
            ann_info["num_lidar_pts"] = 3
            ann_info["num_radar_pts"] = 0            
            ann_info['velocity'] = np.zeros(3)
            ann_infos.append(ann_info)
        info['ann_infos'] = ann_infos
        infos.append(info)
    return infos

def main():
    rope3d_root = "data/rope3d"
    train_infos = generate_info_rope3d(rope3d_root, split='train')
    val_infos = generate_info_rope3d(rope3d_root, split='val')

    total_infos = train_infos + val_infos
    random.shuffle(total_infos)
    train_infos = total_infos[:int(0.7 * len(total_infos))]
    val_infos = total_infos[int(0.7 * len(total_infos)):]
    mmcv.dump(train_infos, './data/rope3d/rope3d_12hz_infos_hom_train.pkl')
    mmcv.dump(val_infos, './data/rope3d/rope3d_12hz_infos_hom_val.pkl')

if __name__ == '__main__':
    main()
