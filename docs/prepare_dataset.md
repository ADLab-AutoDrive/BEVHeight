# DAIR-V2X-I  Rope3D
Download DAIR-V2X-I or Rope3D dataset from official [website](https://thudair.baai.ac.cn/index).

## Symlink the dataset root to `./data/`.
```
ln -s [single-infrastructure-side root] ./data/dair-v2x
ln -s [rope3d root] ./data/rope3d
```

## Convert DAIR-V2X-I or Rope3D to KITTI format.
```
python scripts/data_converter/dair2kitti.py --source-root data/dair-v2x-i --target-root data/dair-v2x-i-kitti
python scripts/data_converter/rope2kitti.py --source-root data/rope3d --target-root data/rope3d-kitti
```

## Visualize the dataset in KITTI format
```
python scripts/data_converter/visual_tools.py --data_root data/rope3d-kitti --demo_dir ./demo
```


The directory will be as follows.
```
BEVHeight
├── data
│   ├── dair-v2x-i
│   │   ├── velodyne
│   │   ├── image
│   │   ├── calib
│   │   ├── label
|   |   └── data_info.json
|   └── dair-v2x-i-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── ImageSets
|   |        ├── train.txt
|   |        └── val.txt
|   ├── rope3d
|   |   ├── training
|   |   ├── validation
|   |   ├── training-image_2a
|   |   ├── training-image_2b
|   |   ├── training-image_2c
|   |   ├── training-image_2d
|   |   └── validation-image_2
|   ├── rope3d-kitti
|   |   ├── training
|   |   |   ├── calib
|   |   |   ├── denorm
|   |   |   ├── label_2
|   |   |   └── images_2
|   |   └── map_token2id.json
|   |       
└── ...
```

## Prepare DAIR-V2X-I or Rope3D infos.
```
python scripts/gen_info_dair.py
python scripts/gen_info_rope3d.py
```
