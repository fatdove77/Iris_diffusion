import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


class IrisAnalysis:
    def __init__(self):
        # 根据实际图像调整参数
        self.pupil_radius = 63  # 根据输出约62.7
        self.iris_radius = 185  # 根据输出约185.1
        self.pupil_tolerance = 5  # 容差值
        self.iris_tolerance = 5  # 容差值

    def segment_iris(self, image):
        """
        Segment iris using Otsu thresholding and contour detection
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            print(f"Processing grayscale image of shape: {gray.shape}")

            # 应用高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Otsu阈值分割
            _, binary = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # 保存二值化图像用于调试
            # cv2.imwrite('debug_binary.png', binary)

            # 寻找轮廓
            contours, _ = cv2.findContours(
                binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            print(f"Found {len(contours)} contours")

            # 保存带轮廓的图像用于调试
            debug_contours = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
            # cv2.imwrite('debug_contours.png', debug_contours)

            # 分析每个轮廓
            circles = []
            for i, contour in enumerate(contours):
                if len(contour) < 5:  # 需要至少5个点来拟合椭圆
                    continue

                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = float(radius)
                circles.append((center, radius))
                print(f"Contour {i}: center={center}, radius={radius:.1f}")

            # 寻找瞳孔和虹膜
            pupil_center = None
            iris_center = None

            # 根据半径查找匹配的圆
            for center, radius in circles:
                if abs(radius - self.pupil_radius) <= self.pupil_tolerance:
                    pupil_center = center
                    print(f"Found pupil: center={center}, radius={radius:.1f}")
                if abs(radius - self.iris_radius) <= self.iris_tolerance:
                    iris_center = center
                    print(f"Found iris: center={center}, radius={radius:.1f}")

            # 如果没有找到精确匹配，使用最接近的圈
            if pupil_center is None and circles:
                # 找到最接近预期瞳孔半径的圆
                best_pupil = min(circles, key=lambda x: abs(x[1] - self.pupil_radius))
                pupil_center = best_pupil[0]
                print(
                    f"Using closest match for pupil: center={pupil_center}, radius={best_pupil[1]:.1f}"
                )

            if iris_center is None and circles:
                # 找到最接近预期虹膜半径的圆
                best_iris = min(circles, key=lambda x: abs(x[1] - self.iris_radius))
                iris_center = best_iris[0]
                print(
                    f"Using closest match for iris: center={iris_center}, radius={best_iris[1]:.1f}"
                )

            if pupil_center is None or iris_center is None:
                print("Failed to detect pupil or iris")
                return None, None, None

            # 绘制检测结果
            debug_result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if pupil_center:
                cv2.circle(
                    debug_result, pupil_center, int(self.pupil_radius), (0, 0, 255), 2
                )
            if iris_center:
                cv2.circle(
                    debug_result, iris_center, int(self.iris_radius), (0, 255, 0), 2
                )
            # cv2.imwrite('debug_detection.png', debug_result)

            return gray, pupil_center, iris_center

        except Exception as e:
            print(f"Segmentation failed with error: {str(e)}")
            import traceback

            traceback.print_exc()
            return None, None, None

    def normalize_iris(self, image, pupil_center, iris_center):
        """
        Normalize iris to 45x360 polar form
        """
        try:
            normalized = np.zeros((45, 360))
            print(
                f"Normalizing iris with pupil center {pupil_center} and iris center {iris_center}"
            )

            for theta in range(360):
                for r in range(45):
                    # 计算当前半径
                    radius_ratio = r / 45.0
                    current_radius = int(
                        self.pupil_radius
                        + (self.iris_radius - self.pupil_radius) * radius_ratio
                    )

                    # 计算坐标
                    angle = theta * np.pi / 180
                    x = int(pupil_center[0] + current_radius * np.cos(angle))
                    y = int(pupil_center[1] + current_radius * np.sin(angle)) 
                continue

            normalized = analyzer.normalize_iris(segmented, pupil_center, iris_center)
            if normalized is None:
                failed_images.append(image_file)
                continue

            iris_code = analyzer.extract_features(normalized)
            iris_codes.append(iris_code)
            print(f"Successfully processed {image_file}")

    if len(iris_codes) == 0:
        print("\nNo images were successfully processed. Cannot calculate distances.")
        return [], failed_images

    # 计算汉明距离
    print("\nCalculating Hamming distances...")
    distances = []
    for i in range(len(iris_codes)):
        for j in range(i + 1, len(iris_codes)):
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
        print(
            f"Percentage HD < 0.4: {(np.sum(distances < 0.4) / len(distances) * 100):.2f}%"
        )

        # 绘制分布图
        plt.figure(figsize=(10, 6))
        sns.histplot(data=distances, bins=50, kde=True)
        plt.axvline(x=0.4, color="r", linestyle="--", label="Threshold (0.4)")
        plt.title("Distribution of Hamming Distances")
        plt.xlabel("Hamming Distance")
        plt.ylabel("Count")
        plt.legend()
        plt.savefig(os.path.join(output_path, "hamming_distribution_250000_3.png"))
        plt.close()

    return distances, failed_images


if __name__ == "__main__":
    dataset_path = "/home/saturn/eyes/1024_sampling_classified/250000_3/images"  # 替换为你的图片路径
    distances, failed_images = analyze_iris_dataset(dataset_path)
