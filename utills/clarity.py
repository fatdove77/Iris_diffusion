

import cv2
import os

def calculate_image_clarity(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None  # 图片无法读取时返回 None
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var

source_dir = '/home/saturn/eyes/100000_sample_test/images_nocolor'  # 将此路径替换为你的图片文件夹路径

# 遍历文件夹中的所有图片文件
for file_name in os.listdir(source_dir):
    file_path = os.path.join(source_dir, file_name)
    clarity = calculate_image_clarity(file_path)
    if clarity is not None:
        print(f'{file_name}: Clarity = {clarity}')
        if clarity < 85:
            os.remove(file_path)
            print(f'Deleted {file_name} due to high clarity value.')
    else:
        print(f'{file_name}: Failed to read image.')
