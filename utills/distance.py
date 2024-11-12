
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import cdist

def rgb_to_hex(rgb):
    """将RGB值转换为十六进制颜色代码"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def hex_to_rgb(hex_color):
    """将十六进制颜色代码转换为RGB值"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# 定义颜色类别
COLOR_CLASSES = {
    'blue': '#0173b2',
    'orange': '#de8f05',
    'green': '#029e73',
    'red': '#d55e00',
    'purple': '#cc78bc',
    'brown': '#ca9161',
    'pink': '#fbafe4',
    'gray': '#949494',
    'yellow': '#ece133',
    'light_blue': '#56b4e9'
}

def extract_color_features(img):
    """
    提取图像的颜色特征
    返回每个颜色类别的比例
    """
    # 将图像转换为RGB格式（如果不是的话）
    if len(img.shape) == 2:  # 灰度图
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA图
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # 将图像reshape为二维数组 (pixel_count, 3)
    pixels = img.reshape(-1, 3)
    total_pixels = len(pixels)
    
    # 初始化特征向量
    features = np.zeros(len(COLOR_CLASSES))
    
    # 转换参考颜色为RGB
    reference_colors = np.array([hex_to_rgb(color) for color in COLOR_CLASSES.values()])
    
    # 计算每个像素到各个参考颜色的距离
    distances = cdist(pixels, reference_colors)
    
    # 对每个像素找到最近的参考颜色
    nearest_color_indices = np.argmin(distances, axis=1)
    
    # 计算每个颜色类别的比例
    for i in range(len(COLOR_CLASSES)):
        features[i] = np.sum(nearest_color_indices == i) / total_pixels
    
    return features

def load_images(directory_path):
    """加载图像并提取特征"""
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    image_files = [f for f in os.listdir(directory_path) 
                  if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        raise ValueError(f"No valid image files found in {directory_path}")
    
    features_list = []
    print(f"Loading images from {directory_path}")
    
    for filename in tqdm(image_files):
        file_path = os.path.join(directory_path, filename)
        try:
            img = cv2.imread(file_path)
            if img is None:
                continue
            
            # 转换为RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 提取特征
            features = extract_color_features(img)
            features_list.append(features)
            
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue
    
    return np.array(features_list)

def calculate_distances(features1, features2=None):
    """计算欧氏距离"""
    if features2 is None:
        distances = cdist(features1, features1)
        # 移除自身比较
        distances = distances[~np.eye(distances.shape[0], dtype=bool)]
    else:
        distances = cdist(features1, features2)
    return distances

def plot_intra_distribution(intra_distances, save_path='intra_distribution.png'):
    """绘制训练集内部比较的分布图"""
    plt.figure(figsize=(8, 6))
    
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 上图：训练集内部比较
    sns.histplot(data=intra_distances.flatten(), 
                bins=30, 
                color='skyblue', 
                stat='density')
    plt.title('Intra-Training Set Comparisons', fontsize=14)
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_inter_distribution(inter_distances, save_path='inter_distribution.png'):
    """绘制数据集间最小距离比较的分布图"""
    plt.figure(figsize=(8, 6))
    
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 计算每个生成图像与训练集的最小距离
    min_inter_distances = np.min(inter_distances, axis=1)
    
    # 下图：数据集间最小距离比较
    sns.histplot(data=min_inter_distances, 
                bins=30, 
                color='skyblue', 
                stat='density')
    plt.title('Lowest Inter-Dataset Comparisons', fontsize=14)
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置路径
    train_path = '/home/saturn/eyes/realeyes/all'
    gen_path = '/home/saturn/eyes/1024_sampling_classified/generated'
    
    try:
        # 加载和处理图像
        print("\nLoading training images...")
        training_features = load_images(train_path)
        
        print("\nLoading generated images...")
        generated_features = load_images(gen_path)
        
        # 计算距离
        print("\nCalculating distances...")
        intra_distances = calculate_distances(training_features)
        inter_distances = calculate_distances(training_features, generated_features)
        
        # 保存距离数据
        np.save('intra_distances.npy', intra_distances)
        np.save('inter_distances.npy', inter_distances)
        
        # 计算95百分位数
        percentile_95 = np.percentile(intra_distances, 95)
        
        # 分别绘制两个分布图
        print("\nPlotting distributions...")
        plot_intra_distribution(intra_distances)
        plot_inter_distribution(inter_distances)
        
        # 输出统计信息
        min_inter_distances = np.min(inter_distances, axis=1)
        valid_count = np.sum(min_inter_distances <= percentile_95)
        print("\nStatistics:")
        print(f"95th percentile threshold: {percentile_95:.4f}")
        print(f"Valid samples: {valid_count}/{len(min_inter_distances)} ({valid_count/len(min_inter_distances)*100:.2f}%)")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()