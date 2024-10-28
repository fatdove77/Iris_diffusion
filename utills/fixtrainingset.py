import os
import shutil

def rename_files(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for i, filename in enumerate(sorted(os.listdir(src_dir))):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # 获取原文件名前缀（假设为xxx，去掉最后一个下划线及其后的内容）
            prefix = filename.split('_')[0]
            new_filename = f"{prefix}_{i}.png"  # 根据需要调整命名格式
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, new_filename)
            shutil.copy(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")

# 使用示例
src_dir = '/home/saturn/eyes/realeyes/rotated_classified'
dst_dir = '/home/saturn/eyes/realeyes/rotated_classified_newname'
rename_files(src_dir, dst_dir)
