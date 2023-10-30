export TORCH_HOME=./$TORCH_HOME
python exps/dair-v2x/bev_height_lss_r101_864_1536_256x256_140.py --amp_backend native -b 2 --gpus 8
python exps/dair-v2x/bev_height_lss_r101_864_1536_256x256_140.py --ckpt outputs/bev_height_lss_r50_864_1536_128x128/checkpoints/ -e -b 2 --gpus 8
