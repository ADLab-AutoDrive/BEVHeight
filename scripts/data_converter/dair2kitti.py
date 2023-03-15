'''
Modified from https://github.com/AIR-THU/DAIR-V2X/blob/main/tools/dataset_converter/dair2kitti.py
'''
import argparse
import os
from scripts.data_converter.gen_kitti.label_lidarcoord_to_cameracoord import gen_lidar2cam
from scripts.data_converter.gen_kitti.label_json2kitti import json2kitti, rewrite_label, label_filter
from scripts.data_converter.gen_kitti.gen_calib2kitti import gen_calib2kitti
from scripts.data_converter.gen_kitti.gen_ImageSets_from_split_data import gen_ImageSet_from_split_data
from scripts.data_converter.gen_kitti.utils import pcd2bin

parser = argparse.ArgumentParser("Generate the Kitti Format Data")
parser.add_argument("--source-root", type=str, default="data/dair-v2x-i", help="Raw data root about DAIR-V2X.")
parser.add_argument(
    "--target-root",
    type=str,
    default="data/dair-v2x-i-kitti",
    help="The data root where the data with kitti format is generated",
)
parser.add_argument(
    "--split-path",
    type=str,
    default="data/single-infrastructure-split-data.json",
    help="Json file to split the data into training/validation/testing.",
)
parser.add_argument("--temp-root", type=str, default="./tmp_file", help="Temporary intermediate file root.")


def mdkir_kitti(target_root):
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    os.system("mkdir -p %s/training" % target_root)
    os.system("mkdir -p %s/training/calib" % target_root)
    os.system("mkdir -p %s/training/label_2" % target_root)
    os.system("mkdir -p %s/testing" % target_root)
    os.system("mkdir -p %s/ImageSets" % target_root)


def rawdata_copy(source_root, target_root):
    os.system("cp -r %s/image %s/training/image_2" % (source_root, target_root))
    os.system("cp -r %s/velodyne %s/training/velodyne" % (source_root, target_root))


def kitti_pcd2bin(target_root):
    pcd_dir = os.path.join(target_root, "training/velodyne")
    if not os.path.exists(pcd_dir): return
    fileList = os.listdir(pcd_dir)
    for fileName in fileList:
        if ".pcd" in fileName:
            pcd_file_path = pcd_dir + "/" + fileName
            bin_file_path = pcd_dir + "/" + fileName.replace(".pcd", ".bin")
            pcd2bin(pcd_file_path, bin_file_path)


if __name__ == "__main__":
    print("================ Start to Convert ================")
    args = parser.parse_args()
    source_root = args.source_root
    target_root = args.target_root

    print("================ Start to Copy Raw Data ================")
    mdkir_kitti(target_root)
    rawdata_copy(source_root, target_root)
    kitti_pcd2bin(target_root)

    print("================ Start to Generate Label ================")
    temp_root = args.temp_root
    os.system("mkdir -p %s" % temp_root)
    os.system("rm -rf %s/*" % temp_root)
    gen_lidar2cam(source_root, temp_root, label_type="camera")

    json_root = os.path.join(temp_root, "label", "camera")
    kitti_label_root = os.path.join(target_root, "training/label_2")
    json2kitti(json_root, kitti_label_root)
    rewrite_label(kitti_label_root)
    label_filter(kitti_label_root)

    os.system("rm -rf %s" % temp_root)

    print("================ Start to Generate Calibration Files ================")
    path_camera_intrinsic = os.path.join(source_root, "calib/camera_intrinsic")
    path_lidar_to_camera = os.path.join(source_root, "calib/virtuallidar_to_camera")
    path_calib = os.path.join(target_root, "training/calib")
    gen_calib2kitti(path_camera_intrinsic, path_lidar_to_camera, path_calib)

    print("================ Start to Generate ImageSet Files ================")
    split_json_path = args.split_path
    ImageSets_path = os.path.join(target_root, "ImageSets")
    gen_ImageSet_from_split_data(ImageSets_path, split_json_path, "infrastructure")
