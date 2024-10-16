import cv2
import numpy as np
import os

def apply_black_rectangle_mask(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # 获取图像尺寸
    h, w = image.shape[:2]

    # 计算矩形框以及虹膜的圆形半径
    center = (w // 2, h // 2)
    radius = min(center[0], center[1])  # 以较小的一边为基准计算半径

    # 创建一个黑色背景
    mask = np.zeros_like(image)

    # 在黑色背景上添加一个白色圆形
    cv2.circle(mask, center, radius, (255, 255, 255), thickness=-1)

    # 应用掩膜，只保留虹膜部分
    masked_image = cv2.bitwise_and(image, mask)

    # 转换所有非虹膜部分为纯黑色
    image[np.where((mask == [0,0,0]).all(axis=2))] = [0, 0, 0]

    # 保存处理后的图像
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to {output_path}")

def process_images(source_dir, dest_dir):
    # 确保目标目录存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 遍历源目录中的所有图片文件
    for file_name in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file_name)
        output_path = os.path.join(dest_dir, file_name)
        apply_black_rectangle_mask(file_path, output_path)

# 设置路径
source_dir = '/home/saturn/eyes/100000_sample_test/image_delete'
dest_dir = '/home/saturn/eyes/100000_sample_test/image_delete_clip'

# 处理所有图片
process_images(source_dir, dest_dir)


# # 设置路径
# source_dir = '/home/saturn/eyes/100000_sample_test/image_delete'
# dest_dir = '/home/saturn/eyes/100000_sample_test/image_delete_clip'

# # 处理所有图片
# process_images(source_dir, dest_dir)
