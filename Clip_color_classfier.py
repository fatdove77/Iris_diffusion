import os
import torch
import clip
from PIL import Image
import argparse
from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm

def process_images(image_dataset_path, similarity_threshold=None, force_classify=False):
    """
    使用CLIP模型处理图像并进行分类
    
    Args:
        image_dataset_path (str): 图像数据集路径
        similarity_threshold (float, optional): 相似度阈值
        force_classify (bool): 是否强制分类(忽略阈值)
    """
    # 设置输出路径
    output_dir = os.path.join(os.path.dirname(image_dataset_path), "classified_images_2")
    category_dirs = {
        'bluegreygreen': os.path.join(output_dir, "images"),
        'lightbrowndark': os.path.join(output_dir, "images"),
        'uncertain': os.path.join(output_dir, "uncertain")
    }
    
    # 创建输出目录
    for dir_path in category_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 设置设备和加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dataset_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not image_files:
        print("No image files found in the specified directory!")
        return

    # 使用更详细的类别描述
    categories = [
        # 第一类：蓝灰绿色调
        [
            "One class label was for irises whose combined proportion of blue-grey pixels, green, white pixels exceeded 50%"
        ],
        # 第二类：棕色和深色调
        [
           "the other class label was for irises whose combined proportion of brwon  pixels and dark brown pixels exceeded 50%"
        ]
    ]

    # 将所有描述转换为tokens
    text_tokens_list = [clip.tokenize(cat_desc).to(device) for cat_desc in categories]
    
    # 计数器
    counters = {
        'bluegreygreen': 0,
        'lightbrowndark': 0,
        'uncertain': 0
    }

    # 用于记录相似度统计
    all_similarities = []
    
    # 批处理大小
    batch_size = 16  # 可以根据显存大小调整

    # 使用tqdm显示进度
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing images"):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        valid_files = []
        
        # 预处理批次中的所有图像
        for image_file in batch_files:
            try:
                image_path = os.path.join(image_dataset_path, image_file)
                image = preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0)
                batch_images.append(image)
                valid_files.append(image_file)
            except Exception as e:
                print(f"\nError processing {image_file}: {e}")
                continue
        
        if not batch_images:
            continue
            
        # 将批次图像堆叠并移到设备
        batch_tensor = torch.cat(batch_images).to(device)
        
        # 处理批次
        with torch.no_grad():
            # 获取图像特征
            image_features = model.encode_image(batch_tensor)
            
            # 对每个类别计算最大相似度
            max_similarity_per_category = []
            for category_tokens in text_tokens_list:
                text_features = model.encode_text(category_tokens)
                similarity = torch.nn.functional.cosine_similarity(
                    image_features[:, None, :],
                    text_features[None, :, :],
                    dim=-1
                )
                max_similarity_per_category.append(torch.max(similarity, dim=1)[0])
            
            # 堆叠所有类别的相似度
            category_similarities = torch.stack(max_similarity_per_category, dim=1)
            
            # 获取最大相似度值和对应的类别
            max_similarities, categories_idx = category_similarities.max(dim=1)
            
            # 记录相似度值用于统计
            all_similarities.extend(max_similarities.cpu().numpy())
            
            # 处理每张图片
            for idx, (image_file, similarity, category_idx) in enumerate(zip(valid_files, 
                                                                           max_similarities, 
                                                                           categories_idx)):
                similarity_value = similarity.item()
                
                # 确定类别
                if force_classify or similarity_threshold is None or similarity_value >= similarity_threshold:
                    category = 'bluegreygreen' if category_idx.item() == 0 else 'lightbrowndark'
                else:
                    category = 'uncertain'
                
                # 生成新文件名
                counters[category] += 1
                new_filename = f"{category}_{counters[category]:04d}.png"
                
                # 复制并重命名文件
                try:
                    src_path = os.path.join(image_dataset_path, image_file)
                    dst_path = os.path.join(category_dirs[category], new_filename)
                    
                    # 转换为PNG并保存
                    img = Image.open(src_path).convert('RGB')
                    img.save(dst_path, 'PNG')
                    
                    print(f"\nProcessed {image_file} -> {new_filename} ({category}) "
                          f"[similarity: {similarity_value:.3f}]")
                except Exception as e:
                    print(f"\nError saving {image_file}: {e}")
                    continue

    # 打印统计信息
    all_similarities = np.array(all_similarities)
    print("\n" + "="*50)
    print("Processing Complete!")
    print("="*50)
    print("\nClassification Results:")
    print(f"Blue/Grey/Green images: {counters['bluegreygreen']}")
    print(f"Light Brown/Dark images: {counters['lightbrowndark']}")
    print(f"Uncertain images: {counters['uncertain']}")
    
    print("\nSimilarity Statistics:")
    print(f"Mean similarity: {np.mean(all_similarities):.3f}")
    print(f"Median similarity: {np.median(all_similarities):.3f}")
    print(f"Min similarity: {np.min(all_similarities):.3f}")
    print(f"Max similarity: {np.max(all_similarities):.3f}")
    print(f"25th percentile: {np.percentile(all_similarities, 25):.3f}")
    print(f"75th percentile: {np.percentile(all_similarities, 75):.3f}")
    
    print("\nOutput Directories:")
    for category, dir_path in category_dirs.items():
        print(f"{category}: {dir_path}")

def main():
    parser = argparse.ArgumentParser(description="Process images using CLIP for classification.")
    parser.add_argument('image_dataset_path', type=str, 
                      help="Path to the directory containing images")
    parser.add_argument('--similarity_threshold', type=float, default=None,
                      help="Similarity threshold for classification (default: None)")
    parser.add_argument('--force_classify', action='store_true',
                      help="Force classify all images regardless of similarity score")
    
    args = parser.parse_args()
    
    process_images(args.image_dataset_path, 
                  args.similarity_threshold,
                  args.force_classify)

if __name__ == '__main__':
    main()