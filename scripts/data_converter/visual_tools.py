import argparse
import os
import cv2

from scripts.data_converter.visual_utils import *

import warnings
warnings.filterwarnings("ignore")

def kitti_visual_tool(data_root, demo_dir):
    if not os.path.exists(data_root):
        raise ValueError("data_root Not Found")
    image_path = os.path.join(data_root, "training/image_2")
    calib_path = os.path.join(data_root, "training/calib")
    label_path = os.path.join(data_root, "training/label_2")
    image_ids = []
    for image_file in os.listdir(image_path):
        image_ids.append(image_file.split(".")[0])
    for i in range(len(image_ids)):
        image_2_file = os.path.join(image_path, str(image_ids[i]) + ".png")
        calib_file = os.path.join(calib_path, str(image_ids[i]) + ".txt")
        label_2_file = os.path.join(label_path, str(image_ids[i]) + ".txt")
        image = cv2.imread(image_2_file)
        _, P2, denorm = load_calib(calib_file)
        image = draw_3d_box_on_image(image, label_2_file, P2, denorm)
        cv2.imwrite(os.path.join(demo_dir, str(image_ids[i]) + ".jpg"), image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset in KITTI format Checking ...")
    parser.add_argument("--data_root", type=str,
                        default="",
                        help="Path to Dataset root in KITTI format")
    parser.add_argument("--demo_dir", type=str,
                        default="",
                        help="Path to demo directions")
    args = parser.parse_args()
    os.makedirs(args.demo_dir, exist_ok=True)
    kitti_visual_tool(args.data_root, args.demo_dir)