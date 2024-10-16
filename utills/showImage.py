# modified_process_images.py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def main(npz_path, output_dir):
    # 加载 .npz 文件
    data = np.load(npz_path)
    print(data.files)  # 输出文件列表

    # 获取图像数组
    images_arr_0 = data['arr_0']

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 输出所有图像并保存到固定路径
    for i in range(images_arr_0.shape[0]):
        plt.imshow(images_arr_0[i])  # 假设图像是 RGB 的
        plt.axis('off')  # 不显示坐标轴
        plt.gca().patch.set_visible(False)  # 隐藏背景
        plt.savefig(os.path.join(output_dir, f'image_{i}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load NPZ file and save images.")
    parser.add_argument('npz_path', help="Path to the .npz data file")
    parser.add_argument('output_dir', help="Directory to save processed images")
    args = parser.parse_args()

    main(args.npz_path, args.output_dir)
