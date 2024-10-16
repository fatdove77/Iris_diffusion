import os
import shutil

def rename_files(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for i, filename in enumerate(sorted(os.listdir(src_dir))):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            new_filename = f"img_num{i}.png"  # 根据需要调整命名格式
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, new_filename)
            shutil.copy(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")

# 使用示例
src_dir = '/home/saturn/eyes/realeyes/rotate_right'
dst_dir = '/home/saturn/eyes/realeyes/rotate_right_newname'
rename_files(src_dir, dst_dir)

# # 使用示例
# src_dir = '/home/saturn/eyes/realeyes/right'
# dst_dir = '/home/saturn/eyes/realeyes/newnameright'
# create_symlinks(src_dir, dst_dir)