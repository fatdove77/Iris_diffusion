import os
import cv2
import numpy as np

# 数据集文件夹路径
image_dataset_path = '/home/saturn/eyes/100000_sample_test/images_nocolor'  # 修改为你的数据集路径

# 存储每张图片的清晰度值
clarity_values = []
image_files = os.listdir(image_dataset_path)

def calculate_clarity(image):
    """计算图像的清晰度，基于拉普拉斯算子的方差"""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance

# 遍历数据集中的每一张图片
for image_file in image_files:
    # 读取图像
    image_path = os.path.join(image_dataset_path, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 转换为灰度图像

    if image is None:
        print(f"无法读取图像: {image_file}")
        continue

    # 计算图像的清晰度（拉普拉斯方差）
    clarity = calculate_clarity(image)
    clarity_values.append((image_file, clarity))

# 提取清晰度值并转为 numpy 数组
clarity_mean_values = np.array([cv[1] for cv in clarity_values])

# 计算清晰度的均值和标准差，并确定一个阈值 (例如，低于 25% 百分位数的图像被认为是模糊)
mean_clarity = np.mean(clarity_mean_values)
std_clarity = np.std(clarity_mean_values)
clarity_threshold = np.percentile(clarity_mean_values, 50)  # 使用 25% 百分位数作为阈值

print(f"清晰度均值: {mean_clarity}")
print(f"清晰度标准差: {std_clarity}")
print(f"确定的清晰度阈值（25% 百分位数）: {clarity_threshold}")

# 根据阈值筛选并删除模糊图片
for image_file, clarity in clarity_values:
    if clarity < clarity_threshold:  # 如果清晰度低于阈值，则认为是模糊图片
        print(f"删除模糊图像: {image_file} (Clarity: {clarity})")
        os.remove(os.path.join(image_dataset_path, image_file)) 
    else:
        print(f"保留正常图像: {image_file} (Clarity: {clarity})")
