import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

class IrisAnalysis:
    def __init__(self, debug_path):
        # 根据实际图像调整参数
        self.pupil_radius = 63  # 根据输出约62.7
        self.iris_radius = 185  # 根据输出约185.1
        self.pupil_tolerance = 5  # 容差值
        self.iris_tolerance = 5   # 容差值
        self.debug_path = debug_path  # 添加debug路径
        
    def save_debug_image(self, image, name, subfolder=None):
        """保存调试图像到指定文件夹"""
        if subfolder:
            save_path = os.path.join(self.debug_path, subfolder)
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, name)
        else:
            full_path = os.path.join(self.debug_path, name)
        cv2.imwrite(full_path, image)
        
    def segment_iris(self, image, image_name):
        """
        Segment iris using Otsu thresholding and contour detection
        """
        try:
            # 创建该图像的debug子文件夹
            image_debug_folder = f"debug_{os.path.splitext(image_name)[0]}"
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            self.save_debug_image(gray, 'gray.png', image_debug_folder)
            
            # 调整对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            self.save_debug_image(enhanced, 'enhanced.png', image_debug_folder)
            
            # 进行自适应阈值处理，可能比Otsu效果更好
            binary = cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            self.save_debug_image(binary, 'binary.png', image_debug_folder)
            
            # 使用形态学操作改善分割
            kernel = np.ones((3,3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            self.save_debug_image(binary, 'morphology.png', image_debug_folder)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # 绘制所有轮廓
            contour_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
            self.save_debug_image(contour_img, 'contours.png', image_debug_folder)
            
            # 筛选轮廓
            valid_circles = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # 忽略太小的轮廓
                    continue
                    
                # 使用最小包围圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = float(radius)
                
                # 检查圆形度
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                if circularity > 0.5:  # 只保留接近圆形的轮廓
                    valid_circles.append((center, radius))
                    
            # 按半径排序
            valid_circles.sort(key=lambda x: x[1])
            
            # 寻找瞳孔和虹膜
            pupil_center = None
            iris_center = None
            pupil_radius = None
            iris_radius = None
            
            # 绘制检测到的圆
            circles_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for center, radius in valid_circles:
                # 绘制所有有效的圆
                cv2.circle(circles_img, center, int(radius), (0,255,255), 2)
                
                # 检查是否符合瞳孔或虹膜的标准
                if abs(radius - self.pupil_radius) <= self.pupil_tolerance:
                    pupil_center = center
                    pupil_radius = radius
                    cv2.circle(circles_img, center, int(radius), (0,0,255), 2)
                elif abs(radius - self.iris_radius) <= self.iris_tolerance:
                    iris_center = center
                    iris_radius = radius
                    cv2.circle(circles_img, center, int(radius), (0,255,0), 2)
                    
            self.save_debug_image(circles_img, 'detected_circles.png', image_debug_folder)
            
            # 如果没有找到精确匹配，使用最接近的圈
            if pupil_center is None and valid_circles:
                closest_pupil = min(valid_circles, key=lambda x: abs(x[1] - self.pupil_radius))
                pupil_center = closest_pupil[0]
                pupil_radius = closest_pupil[1]
                
            if iris_center is None and valid_circles:
                closest_iris = min(valid_circles, key=lambda x: abs(x[1] - self.iris_radius))
                iris_center = closest_iris[0]
                iris_radius = closest_iris[1]
            
            if pupil_center is None or iris_center is None:
                print(f"Failed to detect pupil or iris for {image_name}")
                return None, None, None
                
            # 绘制最终结果
            result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.circle(result_img, pupil_center, int(pupil_radius), (0,0,255), 2)
            cv2.circle(result_img, iris_center, int(iris_radius), (0,255,0), 2)
            self.save_debug_image(result_img, 'final_detection.png', image_debug_folder)
            
            return enhanced, pupil_center, iris_center
            
        except Exception as e:
            print(f"Segmentation failed for {image_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def normalize_iris(self, image, pupil_center, iris_center, image_name):
        """
        Normalize iris to 45x360 polar form
        """
        try:
            normalized = np.zeros((45, 360))
            image_debug_folder = f"debug_{os.path.splitext(image_name)[0]}"
            
            # 计算从瞳孔到虹膜的每个点
            for theta in range(360):
                for r in range(45):
                    radius_ratio = r / 45.0
                    current_radius = int(self.pupil_radius + 
                                      (self.iris_radius - self.pupil_radius) * radius_ratio)
                    
                    angle = theta * np.pi / 180
                    x = int(pupil_center[0] + current_radius * np.cos(angle))
                    y = int(pupil_center[1] + current_radius * np.sin(angle))
                    
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        normalized[r, theta] = image[y, x]
            
            # 保存未增强的归一化图像
            self.save_debug_image(normalized.astype(np.uint8), 
                                'normalized_before_enhance.png', 
                                image_debug_folder)
            
            # 应用CLAHE增强对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(normalized.astype(np.uint8))
            
            # 保存增强后的归一化图像
            self.save_debug_image(enhanced, 
                                'normalized_final.png', 
                                image_debug_folder)
            
            return enhanced
            
        except Exception as e:
            print(f"Normalization failed for {image_name}: {str(e)}")
            return None

def analyze_iris_dataset(folder_path, output_path="results"):
    """
    Analyze a dataset of iris images
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    debug_path = os.path.join(output_path, "debug")
    os.makedirs(debug_path, exist_ok=True)
    
    analyzer = IrisAnalysis(debug_path)
    
    iris_codes = []
    failed_images = []
    
    print("\nProcessing images...")
    for image_file in os.listdir(folder_path):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nProcessing image: {image_file}")
            image_path = os.path.join(folder_path, image_file)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_file}")
                failed_images.append(image_file)
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 处理图像
            segmented, pupil_center, iris_center = analyzer.segment_iris(image, image_file)
            if segmented is None:
                failed_images.append(image_file)
                continue
                
            normalized = analyzer.normalize_iris(segmented, pupil_center, iris_center, image_file)
            if normalized is None:
                failed_images.append(image_file)
                continue
                
            iris_code = analyzer.extract_features(normalized)
            iris_codes.append(iris_code)
            print(f"Successfully processed {image_file}")
    
    if len(iris_codes) == 0:
        print("\nNo images were successfully processed.")
        return [], failed_images
    
    # 计算汉明距离
    print("\nCalculating Hamming distances...")
    distances = []
    for i in range(len(iris_codes)):
        for j in range(i+1, len(iris_codes)):
            distance = analyzer.calculate_hamming_distance(iris_codes[i], iris_codes[j])
            distances.append(distance)
    
    distances = np.array(distances)
    
    # 生成统计信息
    print("\nAnalysis Results:")
    print(f"Successfully processed images: {len(iris_codes)}")
    print(f"Failed images: {len(failed_images)}")
    print(f"Total comparisons: {len(distances)}")
    
    if len(distances) > 0:
        print(f"Mean Hamming distance: {np.mean(distances):.4f}")
        print(f"Std Hamming distance: {np.std(distances):.4f}")
        print(f"Percentage HD < 0.4: {(np.sum(distances < 0.4) / len(distances) * 100):.2f}%")
        
        # 绘制分布图
        plt.figure(figsize=(10, 6))
        sns.histplot(data=distances, bins=50, kde=True)
        plt.axvline(x=0.4, color='r', linestyle='--', label='Threshold (0.4)')
        plt.title('Distribution of Hamming Distances')
        plt.xlabel('Hamming Distance')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(output_path, 'hamming_distribution.png'))
        plt.close()
    
    return distances, failed_images

if __name__ == "__main__":
    dataset_path = "/home/saturn/eyes/realeyes/generated"  # 替换为你的图片路径
    output_path = "results"  # 输出和调试文件夹的路径
    distances, failed_images = analyze_iris_dataset(dataset_path, output_path)