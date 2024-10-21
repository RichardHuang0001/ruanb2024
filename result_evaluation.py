import json
import csv
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, average_precision_score

# ================================
# 读取标注文件 (COCO格式)
# ================================
ANNOTATION_JSON_PATH = 'dataset/device/interaction_device.json'  # 替换为你的标注文件路径
MODEL_OUTPUT_CSV_PATH = 'dataset/device/GLM_results_device.csv'  # 替换为模型输出CSV文件路径
OUTPUT_RESULT_FILE = 'dataset/device/evaluation_results.txt'  # 结果输出文件


def load_annotation_data(json_path):
    """读取标注文件，返回类别映射和标注列表"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取类别映射
    categories = {category['id']: category['name'] for category in data['categories']}
    # print(categories)

    # 获取标注数据
    annotations = {ann['id']: categories[ann['category_id']] for ann in data['annotations']}

    return categories, annotations


# ================================
# 读取模型输出的CSV文件
# ================================
def load_model_output(csv_path):
    """读取模型输出的CSV文件"""
    model_results = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            model_results[int(row['patch_id'])] = row['predicted_category']
    return model_results


# ================================
# 计算分类指标
# ================================
def calculate_metrics(annotations, model_results):
    """计算分类任务的常见指标"""
    true_labels = []
    pred_labels = []
    # print(model_results)
    # 收集真实标签和模型预测
    for patch_id, true_category in annotations.items():
        if patch_id in model_results:
            true_labels.append(true_category)
            pred_labels.append(model_results[patch_id])
            # print(f"patch id{patch_id}->{true_category}, {model_results[patch_id]}")

    # 计算Top-1 Accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    # 计算Precision, Recall, F1-Score (宏平均)
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    # 计算mAP (平均精度) - 使用每个类别的AP来计算mAP
    labels_binary = [[1 if t == category else 0 for t in true_labels] for category in set(true_labels)]
    preds_binary = [[1 if p == category else 0 for p in pred_labels] for category in set(true_labels)]
    ap_scores = [average_precision_score(l, p) for l, p in zip(labels_binary, preds_binary)]
    mAP = sum(ap_scores) / len(ap_scores)

    # 返回指标
    return {
        'Top-1 Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'mAP': mAP
    }


# ================================
# 保存评估结果到TXT文件
# ================================
def save_results_to_txt(metrics, output_path):
    """将计算结果保存到TXT文件中"""
    with open(output_path, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    print(f"Results saved to {output_path}")


# ================================
# 主程序入口
# ================================
if __name__ == "__main__":
    # 加载标注文件和模型输出文件
    categories, annotations = load_annotation_data(ANNOTATION_JSON_PATH)
    model_results = load_model_output(MODEL_OUTPUT_CSV_PATH)

    # 计算指标
    metrics = calculate_metrics(annotations, model_results)

    # 保存结果
    save_results_to_txt(metrics, OUTPUT_RESULT_FILE)
