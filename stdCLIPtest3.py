import csv
import os
import json
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ================================
# 准备 CLIP 模型
# ================================
device ="cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ================================
# 读取 interaction.json 中的 53 个类别
# ================================
INTERACTION_JSON_PATH = 'dataset/sep_test/ingames/interaction_ingame.json'

def load_interaction_categories(json_path):
    """加载 interaction.json 中的类别标签"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    categories = [category['name'] for category in data['categories']]
    print(f"Loaded {len(categories)} interaction categories")
    return categories

interaction_categories = load_interaction_categories(INTERACTION_JSON_PATH)

# ================================
# 为每张图片生成53条描述文本
# ================================
def generate_text_descriptions(semantic_category):
    """生成与每张图片语义类别相关的 53 条描述文本"""
    return [f"A picture of {semantic_category} that can be {interaction}" for interaction in interaction_categories]

# ================================
# 编码图片和文本并计算相似度
# ================================
def calculate_similarity(image_path, semantic_category):
    """计算图像和53条文本描述的相似度，返回置信度分布"""
    # 处理图片
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 生成文本描述
    texts = generate_text_descriptions(semantic_category)
    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        # 编码图片和文本
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        # 归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        similarity = (image_features @ text_features.T).squeeze(0)

    return similarity.cpu().numpy()  # 返回 53 个类别的置信度分布

# ================================
# 评估函数，计算 Top-1 和 Top-5 以及其他分类指标
# ================================
def evaluate_model(df):
    """对数据集进行评估，计算 Top-1 和 Top-5 的准确率以及其他分类指标"""
    total = len(df)
    correct_top1 = 0
    correct_top5 = 0

    all_predictions = []
    all_ground_truths = []

    # 定义保存相似度分布的文件
    SIMILARITY_CSV_FILE = 'dataset/sep_test/ingames/sim_scores_ingame.csv'

    # 初始化 CSV 文件并写入表头
    with open(SIMILARITY_CSV_FILE, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 定义表头，包含 patch_id 和 53 个类别
        header = ['patch_id'] + interaction_categories
        csv_writer.writerow(header)

    for idx, row in tqdm(df.iterrows(), total=total):
        patch_id = row['patch_id']
        image_path = f"dataset/cut_images/{patch_id}.jpg"
        semantic_category = row['semantic_category']
        ground_truth = row['interaction_category']

        # 计算相似度，得到置信度分布
        similarity_scores = calculate_similarity(image_path, semantic_category)

        # 保存相似度分布到同一个 CSV 文件
        with open(SIMILARITY_CSV_FILE, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # 将 patch_id 和 53 个类别的置信度分数写入同一行
            csv_writer.writerow([patch_id] + similarity_scores.tolist())

        # Top-1 和 Top-5 预测
        top1_prediction = similarity_scores.argmax()
        top5_predictions = similarity_scores.argsort()[-5:][::-1]

        # 比较 ground truth 是否在 Top-1 和 Top-5 预测中
        if interaction_categories[top1_prediction] == ground_truth:
            correct_top1 += 1
        if ground_truth in [interaction_categories[i] for i in top5_predictions]:
            correct_top5 += 1

        # 将预测结果和真实标签存储起来，用于计算分类指标
        all_predictions.append(interaction_categories[top1_prediction])
        all_ground_truths.append(ground_truth)

    # 计算准确率
    top1_accuracy = correct_top1 / total * 100
    top5_accuracy = correct_top5 / total * 100

    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

    # 使用 sklearn 计算 Precision, Recall, F1-Score 和 Confusion Matrix
    precision = precision_score(all_ground_truths, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_ground_truths, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_ground_truths, all_predictions, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_ground_truths, all_predictions)

    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)

    # 保存混淆矩阵
    save_confusion_matrix_to_csv(conf_matrix, interaction_categories)

    # 返回分类指标
    return {
        'Top-1 Accuracy': top1_accuracy,
        'Top-5 Accuracy': top5_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': conf_matrix
    }

def save_confusion_matrix_to_csv(conf_matrix, class_names, output_path='confusion_matrix_ingame.csv'):
    """将混淆矩阵保存到 CSV 文件"""
    df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    df.to_csv(output_path)
    print(f"Confusion matrix saved to {output_path}")

def save_metrics_to_csv(metrics, output_path='evaluation_metrics.csv'):
    """将分类指标保存到 CSV 文件"""
    # 将 metrics 字典转换为 DataFrame
    df = pd.DataFrame([metrics])
    df.to_csv(output_path, index=False)
    print(f"Evaluation metrics saved to {output_path}")

# ================================
# 主程序入口
# ================================
if __name__ == "__main__":
    # 读取数据集
    CSV_PATH = 'dataset/sep_test/ingames/cut_images_annotations_ingames.csv'
    df = pd.read_csv(CSV_PATH)
    print(f"Total records in CSV: {len(df)}")

    # 运行评估并计算 Top-1 和 Top-5 Accuracy 以及其他分类指标
    metrics = evaluate_model(df)

    # 保存其他评估指标
    save_metrics_to_csv({
        'Top-1 Accuracy': metrics['Top-1 Accuracy'],
        'Top-5 Accuracy': metrics['Top-5 Accuracy'],
        'Precision': metrics['Precision'] * 100,  # 转换为百分比
        'Recall': metrics['Recall'] * 100,  # 转换为百分比
        'F1-Score': metrics['F1-Score'] * 100  # 转换为百分比
    })
