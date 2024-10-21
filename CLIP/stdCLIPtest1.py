import os
import json
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm
import glob

# ================================
# 初始化 CLIP 模型与设备
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# available models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
model, preprocess = clip.load("ViT-B/32", device=device)

# ================================
# 文件路径配置
# ================================
CSV_PATH = 'dataset/cut_images_annotations.csv'
IMAGE_FOLDER = 'dataset/cut_images'
INTERACTION_JSON_PATH = 'dataset/coco_merged/annotations/interaction.json'

# ================================
# 加载 interaction.json 中的类别标签
# ================================
def load_interaction_categories(json_path):
    """从 interaction.json 文件中加载类别标签"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    categories = [category['name'] for category in data['categories']]
    print(f"Loaded {len(categories)} interaction categories")
    return categories

interaction_categories = load_interaction_categories(INTERACTION_JSON_PATH)

# ================================
# 生成 53 条文本描述
# ================================
def generate_text_descriptions(semantic_category):
    """根据 semantic category 生成 53 条描述文本"""
    # return [
    #     f"Virtual reality users can interact with {semantic_category} by {interaction}"
    #     for interaction in interaction_categories
    # ]
    return [
        f"A picture of {semantic_category} that can be {interaction}"
        for interaction in interaction_categories
    ]

# ================================
# 批量处理分类的核心函数
# ================================
def classify_patches_batch(image_paths, semantic_categories):
    """批量处理图片并返回 CLIP 的分类结果"""
    images = [preprocess(Image.open(img)).unsqueeze(0) for img in image_paths]
    images = torch.cat(images).to(device)  # 将所有图片合并为一个 batch

    # 批量生成所有文本描述的特征
    all_predictions = []
    with torch.no_grad():
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化

        # 对每张图片生成 53 条描述并计算相似度
        for i, semantic_category in enumerate(semantic_categories):
            texts = generate_text_descriptions(semantic_category)
            text_tokens = clip.tokenize(texts).to(device)

            # 编码文本并归一化
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 计算相似度并返回 Top-1 预测
            similarity = image_features[i] @ text_features.T
            top1_index = similarity.argmax().item()
            all_predictions.append(interaction_categories[top1_index])

    return all_predictions

# ================================
# 评估函数：计算 Top-1 Accuracy
# ================================
def compute_top1_accuracy(predictions, ground_truths):
    """计算 Top-1 Accuracy"""
    correct = sum(p == gt for p, gt in zip(predictions, ground_truths))
    accuracy = correct / len(ground_truths)
    print(f"Top-1 Accuracy: {accuracy * 100:.2f}%")
    return accuracy


# ================================
# 批量分类与评估
# ================================
def evaluate_model(df, batch_size=32):
    """按批次处理并评估模型性能"""
    all_predictions = []
    all_ground_truths = []
    error_log = []  # 存储错误信息

    # 按 batch 处理图片
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df.iloc[i:i + batch_size]  # 获取当前 batch 的数据

        image_paths = []
        for _, row in batch.iterrows():
            patch_id = row['patch_id']

            # 使用 glob 匹配图片文件名，以 patch_id 开头
            pattern = os.path.join(IMAGE_FOLDER, f"{patch_id}_*.jpg")
            matched_files = glob.glob(pattern)

            if not matched_files:
                error_log.append(f"未找到图片文件，Patch ID: {patch_id}")
                continue  # 跳过此 patch

            if len(matched_files) > 1:
                error_log.append(f"找到多张图片，Patch ID: {patch_id} - 文件: {matched_files}")
                continue  # 跳过此 patch

            # 取匹配到的第一个文件
            image_paths.append(matched_files[0])

        # 获取语义类别和真实标签
        semantic_categories = batch['semantic_category'].tolist()
        ground_truths = batch['interaction_category'].tolist()

        # 检查图片数量与标签数量是否匹配
        if len(image_paths) != len(semantic_categories):
            error_log.append(f"批次图片数量与语义类别数量不一致，跳过该批次。")
            continue  # 跳过该批次

        # 批量分类并存储预测结果和真实标签
        predictions = classify_patches_batch(image_paths, semantic_categories)
        all_predictions.extend(predictions)
        all_ground_truths.extend(ground_truths)

    # 输出错误日志
    if error_log:
        with open("error_log_CLIPtest1.txt", "w") as f:
            for error in error_log:
                f.write(error + "\n")
        print(f"错误日志已保存至 error_log_CLIPtest1.txt")

    # 计算并返回 Top-1 Accuracy
    return compute_top1_accuracy(all_predictions, all_ground_truths)


# ================================
# 主程序入口
# ================================
if __name__ == "__main__":
    # 读取 CSV 文件
    df = pd.read_csv(CSV_PATH)

    # 运行评估并计算 Top-1 Accuracy
    evaluate_model(df, batch_size=64)
