from PIL import Image
import os
import re

def rotate_and_rename_images(image_dir, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有PNG文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    # 分离两种类型的图片
    bluegreygreen_images = []
    lightbrowndark_images = []
    
    for filename in image_files:
        if filename.startswith('bluegreygreen'):
            bluegreygreen_images.append(filename)
        elif filename.startswith('lightbrowndark'):
            lightbrowndark_images.append(filename)
    
    # 排序文件名
    bluegreygreen_images.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    lightbrowndark_images.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    
    # 计数器
    new_index = 1
    
    # 处理所有图片
    for image_list in [bluegreygreen_images, lightbrowndark_images]:
        for filename in image_list:
            # 获取分类名(bluegreygreen或lightbrowndark)
            category = filename.split('_')[0]
            
            # 加载图片
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            original_size = img.size
            
            # 保存原始图片
            new_filename = f"{category}_{new_index}.png"
            img.save(os.path.join(output_dir, new_filename))
            new_index += 1
            
            # 旋转并保存
            for angle in range(30, 360, 30):
                rotated_img = img.rotate(angle, expand=True)
                
                # 计算中心裁剪框
                rotated_size = rotated_img.size
                left = (rotated_size[0] - original_size[0]) / 2
                top = (rotated_size[1] - original_size[1]) / 2
                right = (rotated_size[0] + original_size[0]) / 2
                bottom = (rotated_size[1] + original_size[1]) / 2
                cropped_img = rotated_img.crop((left, top, right, bottom))
                
                # 保存旋转后的图片
                new_filename = f"{category}_{new_index}.png"
                cropped_img.save(os.path.join(output_dir, new_filename))
                new_index += 1

    print(f"Successfully processed {len(image_files)} images with rotations")
    print(f"Total images generated: {new_index-1}")

# 使用示例
input_dir = '/home/saturn/eyes/realeyes/classified_images_1/images'
output_dir = '/home/saturn/eyes/realeyes/rotated_classified_newname'
rotate_and_rename_images(input_dir, output_dir)