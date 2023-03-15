import os
import math
import json
import mmcv

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from scripts.vis_utils import *

name2nuscenceclass = {
    "car": "vehicle.car",
    "van": "vehicle.car",
    "truck": "vehicle.truck",
    "bus": "vehicle.bus.rigid",
    "cyclist": "vehicle.bicycle",
    "bicycle": "vehicle.bicycle",
    "tricyclist": "vehicle.bicycle",
    "motorcycle": "vehicle.bicycle",
    "motorcyclist": "vehicle.bicycle",
    "barrowlist": "vehicle.bicycle",
    "barrow": "vehicle.bicycle",
    "pedestrian": "human.pedestrian.adult",
    "traffic_cone": "movable_object.trafficcone",
}

def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json

def get_velo2cam(path):
    my_json = read_json(path)
    t_velo2cam = np.array(my_json["translation"])
    r_velo2cam = np.array(my_json["rotation"])
    return r_velo2cam, t_velo2cam

def get_P(path):
    my_json = read_json(path)
    P = np.array(my_json["cam_K"]).reshape(3,3)
    return P

def get_annos(path):
    my_json = read_json(path)
    gt_names = []
    gt_boxes = []
    for item in my_json:
        gt_names.append(item["type"].lower())
        x, y, z = float(item["3d_location"]["x"]), float(item["3d_location"]["y"]), float(item["3d_location"]["z"])
        h, w, l = float(item["3d_dimensions"]["h"]), float(item["3d_dimensions"]["w"]), float(item["3d_dimensions"]["l"])                                                            
        lidar_yaw = float(item["rotation"])
        gt_boxes.append([x, y, z, l, w, h, lidar_yaw])
    gt_boxes = np.array(gt_boxes)
    return gt_names, gt_boxes

def load_data(dair_root, token):
    sample_id = token.split('/')[1].split('.')[0]
    img_pth = os.path.join(dair_root, token)
    camera_intrinsic_path = os.path.join(dair_root, "calib", "camera_intrinsic", sample_id + ".json")
    virtuallidar_to_camera_path = os.path.join(dair_root, "calib", "virtuallidar_to_camera", sample_id + ".json")
    label_path = os.path.join(dair_root, "label", "camera", sample_id + ".json")
    r_velo2cam, t_velo2cam = get_velo2cam(virtuallidar_to_camera_path)
    P = get_P(camera_intrinsic_path)
    gt_names, gt_boxes = get_annos(label_path)
    return r_velo2cam, t_velo2cam, P, gt_names, gt_boxes, img_pth

def cam2velo(r_velo2cam, t_velo2cam):
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :3] = r_velo2cam
    Tr_velo2cam[:3 ,3] = t_velo2cam.flatten()
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)
    r_cam2velo = Tr_cam2velo[:3, :3]
    t_cam2velo = Tr_cam2velo[:3, 3]
    return r_cam2velo, t_cam2velo
    
def equation_plane(points): 
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])

def get_denorm(rotation_matrix, translation):
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = rotation_matrix
    lidar2cam[:3, 3] = translation.flatten()
    ground_points_lidar = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    ground_points_lidar = np.concatenate((ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), axis=1)
    ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
    denorm = -1 * equation_plane(ground_points_cam)
    
    return denorm

def generate_info_dair(dair_root, split):    
    infos = mmcv.load("data/single-infrastructure-split-data.json")
    split_list = infos[split]
    infos = list()
    for sample_id in tqdm(split_list):
        token = "image/" + sample_id + ".jpg"
        r_velo2cam, t_velo2cam, camera_intrinsic, gt_names, gt_boxes, img_pth = load_data(dair_root, token)

        info = dict()
        cam_info = dict()
        info['sample_token'] = token
        info['timestamp'] = 1000000
        info['scene_token'] = token
        cam_names = ['CAM_FRONT']
        lidar_names = ['LIDAR_TOP']
        cam_infos, lidar_infos = dict(), dict()
        for cam_name in cam_names:
            cam_info = dict()
            cam_info['sample_token'] = token
            cam_info['timestamp'] = 1000000
            cam_info['is_key_frame'] = True
            cam_info['height'] = 1080
            cam_info['width'] = 1920
            cam_info['filename'] = token
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": token, "timestamp": 1000000}
            cam_info['ego_pose'] = ego_pose
            
            denorm = get_denorm(r_velo2cam, t_velo2cam)
            r_cam2velo, t_cam2velo = cam2velo(r_velo2cam, t_velo2cam)
            calibrated_sensor = {"token": token, "sensor_token": token, "translation": t_cam2velo.flatten(), "rotation_matrix": r_cam2velo, "camera_intrinsic": camera_intrinsic}
            cam_info['calibrated_sensor'] = calibrated_sensor
            cam_info['denorm'] = denorm
            cam_infos[cam_name] = cam_info                  
        for lidar_name in lidar_names:
            lidar_info = dict()
            lidar_info['sample_token'] = token
            ego_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0], "token": token, "timestamp": 1000000}
            lidar_info['ego_pose'] = ego_pose
            lidar_info['timestamp'] = 1000000
            lidar_info['filename'] = "velodyne/" + sample_id + ".pcd"
            lidar_info['calibrated_sensor'] = calibrated_sensor
            lidar_infos[lidar_name] = lidar_info            
        info['cam_infos'] = cam_infos
        info['lidar_infos'] = lidar_infos
        info['sweeps'] = list()
        
        # demo(img_pth, gt_boxes, r_velo2cam, t_velo2cam, camera_intrinsic)   
        ann_infos = list()
        for idx in range(gt_boxes.shape[0]):
            category_name = gt_names[idx]
            if category_name not in name2nuscenceclass.keys():
                continue
            gt_box = gt_boxes[idx]
            lwh = gt_box[3:6]
            loc = gt_box[:3]    # need to certify
            yaw_lidar = gt_box[6]
            rot_mat = np.array([[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], 
                                [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], 
                                [0, 0, 1]])    
            rotation = Quaternion(matrix=rot_mat)
            ann_info = dict()
            ann_info["category_name"] = name2nuscenceclass[category_name]
            ann_info["translation"] = loc
            ann_info["rotation"] = rotation
            ann_info["yaw_lidar"] = yaw_lidar
            ann_info["size"] = lwh
            ann_info["prev"] = ""
            ann_info["next"] = ""
            ann_info["sample_token"] = token
            ann_info["instance_token"] = token
            ann_info["token"] = token
            ann_info["visibility_token"] = "0"
            ann_info["num_lidar_pts"] = 3
            ann_info["num_radar_pts"] = 0            
            ann_info['velocity'] = np.zeros(3)
            ann_infos.append(ann_info)
        info['ann_infos'] = ann_infos
        infos.append(info)
        
    return infos

def main():
    dair_root = "data/dair-v2x-i"
    train_infos = generate_info_dair(dair_root, split='train')
    val_infos = generate_info_dair(dair_root, split='val')
    
    mmcv.dump(train_infos, './data/dair-v2x-i/dair_12hz_infos_train.pkl')
    mmcv.dump(val_infos, './data/dair-v2x-i/dair_12hz_infos_val.pkl')

if __name__ == '__main__':
    main()
