import os
import cv2
import numpy as np
import argparse

def main(image_dataset_path, threshold_percentile):
    # 存储每张图片的拉普拉斯均值
    laplacian_means = []
    image_files = os.listdir(image_dataset_path)

    # 遍历数据集中的每一张图片
    for image_file in image_files:
        # 读取图像
        image_path = os.path.join(image_dataset_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"无法读取图像: {image_file}")
            continue

        # 计算拉普拉斯算子
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        # 计算拉普拉斯矩阵的均值
        laplacian_mean = np.mean(np.absolute(laplacian))
        laplacian_means.append((image_file, laplacian_mean))

    # 提取均值并转为 numpy 数组
    laplacian_mean_values = np.array([lm[1] for lm in laplacian_means])
    # 计算均值和标准差，确定阈值
    mean_value = np.mean(laplacian_mean_values)
    std_value = np.std(laplacian_mean_values)
    threshold = np.percentile(laplacian_mean_values, threshold_percentile)

    print(f"拉普拉斯均值: {mean_value}")
    print(f"拉普拉斯标准差: {std_value}")
    print(f"确定的阈值（{threshold_percentile}% 百分位数）: {threshold}")

    # 根据阈值筛选并删除噪声图片
    for image_file, laplacian_mean in laplacian_means:
        if laplacian_mean > threshold:
            print(f"删除噪声图像: {image_file} (Laplacian mean: {laplacian_mean})")
            os.remove(os.path.join(image_dataset_path, image_file))
        else:
            print(f"保留正常图像: {image_file} (Laplacian mean: {laplacian_mean})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images and filter based on clarity.")
    parser.add_argument('image_dataset_path', type=str, help="The path to the image dataset directory.")
    parser.add_argument('threshold_percentile', type=float, help="The percentile to use as a threshold for filtering images.")
    args = parser.parse_args()

    main(args.image_dataset_path, args.threshold_percentile)
