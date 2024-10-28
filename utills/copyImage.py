#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path

def copy_files(source_dir, target_dir):
    """
    复制源文件夹中的所有文件到目标文件夹
    
    Args:
        source_dir: 源文件夹路径
        target_dir: 目标文件夹路径
    """
    # 转换为Path对象
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 检查源文件夹是否存在
    if not source_path.exists():
        print(f"错误: 源文件夹 '{source_dir}' 不存在")
        sys.exit(1)
    
    # 确保目标文件夹存在
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 遍历源文件夹中的所有文件
    try:
        file_count = 0
        for item in source_path.glob('*'):
            if item.is_file():
                # 构建目标文件路径
                target_file = target_path / item.name
                # 复制文件
                shutil.copy2(item, target_file)
                print(f"已复制: {item.name}")
                file_count += 1
        
        print(f"\n复制完成! 共复制了 {file_count} 个文件")
        
    except Exception as e:
        print(f"复制过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("使用方法: python copy_files.py 源文件夹路径 目标文件夹路径")
        sys.exit(1)
    
    # 获取源文件夹和目标文件夹路径
    source_directory = sys.argv[1]
    target_directory = sys.argv[2]
    
    # 执行复制操作
    copy_files(source_directory, target_directory)