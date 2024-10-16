 python -m pytorch-fid /home/saturn/eyes/realeyes/all /home/saturn/eyes/100000_sample_test/images


# import os
# import shutil

# # 源目录
# source_dir = '/home/saturn/eyes/realeyes/right'
# # 目标目录
# target_dir = '/home/saturn/eyes/realeyes/all'

# # 确保目标目录存在，如果不存在就创建
# if not os.path.exists(target_dir):
#     os.makedirs(target_dir)

# # 遍历源目录中的文件
# for filename in os.listdir(source_dir):
#     # 构造完整的文件路径
#     file_path = os.path.join(source_dir, filename)
#     # 检查文件是否是一个文件而非目录
#     if os.path.isfile(file_path):
#         # 构造目标路径
#         dest_path = os.path.join(target_dir, filename)
#         # 移动文件
#         shutil.copy(file_path, dest_path)
#         print(f'Moved: {filename}')

# print('All files have been moved to the target directory.')
