#!/bin/bash

# 定义源目录和输出目录
SOURCE_DIR="/home/saturn/eyes/75000_sample_test/images_nocolor"
OUTPUT_DIR="/home/saturn/eyes/75000_sample_test/images_nocolor"
NPZ_PATH="/home/saturn/eyes/75000_sample_test/samples_2048x256x256x3.npz"

THRESHOLD_PERCENTILE=50  #laplacian threshold

SIMILARITY_THRESHOLD=0.9 #clip threshold

DIRECTORY="/home/saturn/eyes/75000_sample_test/images_nocolor"


# 确保输出目录存在
mkdir -p $OUTPUT_DIR

# 调用第一个 Python 脚本
# python3 showImage.py $NPZ_PATH $OUTPUT_DIRz
# python3 laplacian.py $SOURCE_DIR $THRESHOLD_PERCENTILE
# FILE_COUNT=$(ls -l "$DIRECTORY" | grep "^-" | wc -l)
# echo "Number of directories in $DIRECTORY is $FILE_COUNT"
# # python3 ../Clip_image_encoder.py $SOURCE_DIR $SIMILARITY_THRESHOLD
# echo "Number of directories in $DIRECTORY is $FILE_COUNT"

cd ../pytorch-fid

# 使用 Python 模块运行 FID 计算
python -m pytorch_fid /home/saturn/eyes/realeyes/all $SOURCE_DIR

echo "All scripts executed successfully."
