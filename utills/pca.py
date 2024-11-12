import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def process_image(image_path):
    """优化后的图像处理函数,使用直接的颜色统计"""
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image at {image_path}")
    
    # 转换为RGB颜色空间
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 创建掩码来排除黑色背景
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 10
    
    # 提取有效像素
    valid_pixels = img[mask]
    
    if len(valid_pixels) == 0:
        raise ValueError(f"No valid pixels found in {image_path}")
    
    # 计算颜色特征
    features = {
        'mean_r': np.mean(valid_pixels[:, 0]),
        'mean_g': np.mean(valid_pixels[:, 1]),
        'mean_b': np.mean(valid_pixels[:, 2]),
        'std_r': np.std(valid_pixels[:, 0]),
        'std_g': np.std(valid_pixels[:, 1]),
        'std_b': np.std(valid_pixels[:, 2]),
        'median_r': np.median(valid_pixels[:, 0]),
        'median_g': np.median(valid_pixels[:, 1]),
        'median_b': np.median(valid_pixels[:, 2])
    }
    
    # 添加颜色分布特征
    percentiles = [25, 75]
    for p in percentiles:
        features.update({
            f'p{p}_r': np.percentile(valid_pixels[:, 0], p),
            f'p{p}_g': np.percentile(valid_pixels[:, 1], p),
            f'p{p}_b': np.percentile(valid_pixels[:, 2], p)
        })
    
    return features

def classify_iris_color(features):
    """基于颜色特征对虹膜进行分类"""
    # 定义颜色分类阈值
    thresholds = {
        'blue-grey': {
            'mean_b': 100,
            'mean_g': 80,
            'mean_r': 70
        },
        'green': {
            'mean_g': 90,
            'mean_b': 70,
            'mean_r': 70
        },
        'light-brown': {
            'mean_r': 120,
            'mean_g': 90,
            'mean_b': 70
        },
        'dark-brown': {
            'mean_r': 140,
            'mean_g': 80,
            'mean_b': 60
        }
    }
    
    # 基于RGB均值进行分类
    if (features['mean_b'] > thresholds['blue-grey']['mean_b'] and 
        features['mean_g'] > thresholds['blue-grey']['mean_g']):
        return 'blue-grey'
    elif (features['mean_g'] > thresholds['green']['mean_g'] and 
          features['mean_b'] > thresholds['green']['mean_b']):
        return 'green'
    elif (features['mean_r'] > thresholds['dark-brown']['mean_r'] and 
          features['mean_g'] < thresholds['dark-brown']['mean_g']):
        return 'dark-brown'
    else:
        return 'light-brown'

def process_dataset(image_dir, output_file):
    """处理整个数据集并保存为TSV"""
    image_files = glob.glob(str(Path(image_dir) / "*.[jp][pn][g]"))
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    results = []
    total = len(image_files)
    
    for i, image_path in enumerate(image_files):
        try:
            print(f"Processing {i+1}/{total}: {Path(image_path).name}", end='\r')
            image_name = Path(image_path).stem
            color_features = process_image(image_path)
            # 添加颜色分类
            color_features['color_category'] = classify_iris_color(color_features)
            result = {'name': image_name}
            result.update(color_features)
            results.append(result)
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
    
    print("\nProcessing complete")
    df = pd.DataFrame(results)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Results saved to {output_file}")
    return df

def create_color_pca_visualization(training_df, samples_df):
    """创建基于颜色分类的PCA可视化"""
    # 准备数据
    feature_cols = [col for col in training_df.columns if col not in ['name', 'color_category']]
    
    # 合并数据并添加来源标记
    training_data = training_df[feature_cols].copy()
    samples_data = samples_df[feature_cols].copy()
    
    # 合并数据进行PCA
    combined_data = pd.concat([training_data, samples_data])
    
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_data)
    
    # 执行PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # 分离训练集和样本集结果
    training_pca = pca_result[:len(training_data)]
    samples_pca = pca_result[len(training_data):]
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 定义颜色映射
    color_map = {
        'blue-grey': '#4682B4',
        'green': '#2E8B57',
        'light-brown': '#CD853F',
        'dark-brown': '#8B4513'
    }
    
    # 绘制训练集数据点
    for color in color_map.keys():
        mask = training_df['color_category'] == color
        plt.scatter(training_pca[mask, 0], training_pca[mask, 1],
                   c=color_map[color], alpha=0.6, s=100,
                   label=f'Training ({color})')
    
    # 绘制生成样本数据点
    for color in color_map.keys():
        mask = samples_df['color_category'] == color
        plt.scatter(samples_pca[mask, 0], samples_pca[mask, 1],
                   c=color_map[color], alpha=0.3, s=100, marker='x',
                   label=f'Samples ({color})')
    
    # 设置标题和标签
    plt.title('Iris Pigmentation Distribution PCA by Color Category', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    
    # 添加图例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 计算和显示各种颜色的分布比例
    print("\nColor Distribution Analysis:")
    print("\nTraining Set Distribution:")
    train_dist = training_df['color_category'].value_counts(normalize=True)
    for color in color_map.keys():
        print(f"{color}: {train_dist.get(color, 0)*100:.2f}%")
    
    print("\nGenerated Samples Distribution:")
    sample_dist = samples_df['color_category'].value_counts(normalize=True)
    for color in color_map.keys():
        print(f"{color}: {sample_dist.get(color, 0)*100:.2f}%")
    
    # 输出PCA解释方差比
    print("\nPCA Analysis:")
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", sum(pca.explained_variance_ratio_))
    
    plt.tight_layout()
    return plt.gcf()

def analyze_color_distribution(training_dir, samples_dir):
    """主函数：执行完整的颜色分析流程"""
    # 处理训练集
    print("Processing training set...")
    training_df = process_dataset(training_dir, "training_colors.tsv")
    
    # 处理生成样本
    print("\nProcessing generated samples...")
    samples_df = process_dataset(samples_dir, "samples_colors.tsv")
    
    # 创建可视化
    print("\nCreating visualization...")
    fig = create_color_pca_visualization(training_df, samples_df)
    
    # 保存结果
    plt.savefig('iris_color_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return training_df, samples_df

if __name__ == "__main__":
    # 设置输入路径
    training_dir = "/home/saturn/eyes/realeyes/all"
    samples_dir = "/home/saturn/eyes/1024_sampling_classified/generated"
    
    # 执行分析
    training_df, samples_df = analyze_color_distribution(training_dir, samples_dir)