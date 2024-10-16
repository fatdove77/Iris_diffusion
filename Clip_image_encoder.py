import os
import torch
import clip
from PIL import Image
import argparse

def main(image_dataset_path, similarity_threshold):
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 获取图像文件
    image_files = os.listdir(image_dataset_path)

    # 定义要区分的文本标签
    labels = ["noisy image", "normal iris image"]
    text = clip.tokenize(labels).to(device)

    # 遍历数据集中的每一张图片
    for image_file in image_files:
        # 读取图像
        image_path = os.path.join(image_dataset_path, image_file)
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            # 编码图像和文本
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # 打印相似度结果
        print(f"图像: {image_file}, 噪音图像概率: {probs[0][0]}, 正常虹膜图像概率: {probs[0][1]}")

        # 根据相似度判断并删除噪音图像
        if probs[0][1] < similarity_threshold:  # 如果 "normal iris image" 的概率低于阈值，删除
            print(f"删除噪音图像: {image_file}")
            os.remove(image_path)
        else:
            print(f"保留正常虹膜图像: {image_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images using CLIP and filter based on provided similarity threshold.")
    parser.add_argument('image_dataset_path', type=str, help="The path to the image dataset directory.")
    parser.add_argument('similarity_threshold', type=float, help="The similarity threshold to use as a cutoff for filtering images.")
    args = parser.parse_args()

    main(args.image_dataset_path, args.similarity_threshold)
