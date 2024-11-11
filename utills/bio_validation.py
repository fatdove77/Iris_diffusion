
import numpy as np
import cv2
import os
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

class IrisAnalyzer:
    def __init__(self):
        self.colors = sns.color_palette("colorblind")
        
    def load_images_from_folder(self, folder_path):
        """加载并预处理图片"""
        images = []
        total_images = 0
        failed_validation = 0
        failed_loading = 0
        
        print(f"Loading images from {folder_path}...")
        for filename in tqdm(os.listdir(folder_path)):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                total_images += 1
                img_path = os.path.join(folder_path, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (256, 256))
                        images.append(img)
                    else:
                        failed_loading += 1
                        print(f"Failed to load: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    failed_loading += 1
        
        print(f"\nProcessing Summary for {folder_path}:")
        print(f"Total images found: {total_images}")
        print(f"Successfully loaded: {len(images)}")
        print(f"Failed to load: {failed_loading}")
        
        return np.array(images) if images else np.array([])

    def process_and_save_data(self, training_folder, generated_folder):
        """处理并保存数据"""
        # 加载图片
        training_images = self.load_images_from_folder(training_folder)
        generated_images = self.load_images_from_folder(generated_folder)
        
        if len(training_images) == 0 or len(generated_images) == 0:
            raise ValueError("No images were successfully loaded!")
        
        # 保存为npy文件
        np.save('training_images.npy', training_images)
        np.save('generated_images.npy', generated_images)
        
        print(f"\nSaved training images: {training_images.shape}")
        print(f"Saved generated images: {generated_images.shape}")
        
        return training_images, generated_images

    def plot_distributions(self, training_distances, sample_distances):
        """绘制分布图"""
        if len(training_distances) == 0 or len(sample_distances) == 0:
            raise ValueError("No distance data available for plotting!")
            
        plt.figure(figsize=(7.16, 5.37))
        
        sns.kdeplot(data=training_distances, color=self.colors[0], 
                   label="Real Irises", linewidth=2.5)
        sns.kdeplot(data=sample_distances, color=self.colors[3], 
                   label="Generated Irises", linewidth=2.5)

        plt.legend(loc='best', fontsize=10)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.title('Real Irises vs. Generated Irises', fontsize=16)
        plt.xlabel('Hamming Distance', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xlim(0.3, 0.5)
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        plt.savefig('hamming_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    try:
        # 初始化分析器
        analyzer = IrisAnalyzer()
        
        # 设置输入文件夹路径
        training_folder = "/home/saturn/eyes/realeyes/training_test"  # 替换为你的路径
        generated_folder = "/home/saturn/eyes/realeyes/sample_test"  # 替换为你的路径
        
        # 验证路径是否存在
        if not os.path.exists(training_folder):
            raise ValueError(f"Training folder not found: {training_folder}")
        if not os.path.exists(generated_folder):
            raise ValueError(f"Generated folder not found: {generated_folder}")
        
        # 处理图片数据
        training_images, generated_images = analyzer.process_and_save_data(
            training_folder, generated_folder
        )
        
        # 如果已经有计算好的Hamming距离数据，直接加载
        if os.path.exists('HD_training.npy') and os.path.exists('HD_samples.npy'):
            training_distances = np.load('HD_training.npy', allow_pickle=True)
            sample_distances = np.load('HD_samples.npy', allow_pickle=True)
        else:
            print("Loading precomputed distances not found, computing new ones...")
            # 这里可以添加你的Hamming距离计算代码
        
        # 绘制分布图
        analyzer.plot_distributions(training_distances, sample_distances)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
