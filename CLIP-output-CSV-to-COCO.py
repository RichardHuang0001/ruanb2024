import json
import pandas as pd


# Step 1: 读取 new_groundtruth.json 文件
def load_ground_truth(json_path):
    with open(json_path, 'r') as f:
        ground_truth_data = json.load(f)
    return ground_truth_data


# Step 2: 读取 similarity_score.csv 文件，找到每个 patch_id 的 top-1 类别
def process_similarity_scores(csv_path, category_mapping):
    similarity_scores = pd.read_csv(csv_path)
    patch_predictions = {}

    # 遍历每一行，找到每个 patch_id 的 top-1 类别
    for _, row in similarity_scores.iterrows():
        patch_id = int(row['patch_id'])

        # 找出最高相似度类别
        scores = row[1:]  # 排除掉 patch_id 列
        top_category_name = scores.idxmax()

        # 根据类别名查找 category_id
        if top_category_name in category_mapping:
            top_category_id = category_mapping[top_category_name]
            patch_predictions[patch_id] = top_category_id

    return patch_predictions


# Step 3: 根据模型的预测结果更新 ground truth 的 category_id
def update_ground_truth_categories(ground_truth_data, patch_predictions):
    # 遍历 ground truth 的 annotations，修改对应的 category_id
    for annotation in ground_truth_data['annotations']:
        patch_id = annotation['id']

        if patch_id in patch_predictions:
            # 更新 category_id 为模型预测的值
            annotation['category_id'] = patch_predictions[patch_id]

    return ground_truth_data


# Step 4: 保存更新后的 ground truth 文件
def save_updated_ground_truth(updated_ground_truth, output_path):
    with open(output_path, 'w') as f:
        json.dump(updated_ground_truth, f, indent=4)
    print(f"Updated ground truth saved to {output_path}")


# 主函数：执行所有步骤
def main():
    ground_truth_json_path = 'dataset/new_ground_truth.json'  # 替换为实际路径
    similarity_csv_path = 'dataset/similarity_scores.csv'  # 替换为实际路径
    updated_ground_truth_path = 'dataset/CLIP_results.json'  # 替换为实际保存路径

    # 加载 ground truth 数据
    ground_truth_data = load_ground_truth(ground_truth_json_path)

    # 加载 categories 映射
    category_mapping = {cat["name"]: cat["id"] for cat in ground_truth_data["categories"]}

    # 处理 similarity scores，找到每个 patch_id 的 top-1 预测类别
    patch_predictions = process_similarity_scores(similarity_csv_path, category_mapping)

    # 更新 ground truth 的 categories
    updated_ground_truth = update_ground_truth_categories(ground_truth_data, patch_predictions)

    # 保存更新后的 ground truth
    save_updated_ground_truth(updated_ground_truth, updated_ground_truth_path)


# 运行主函数
if __name__ == "__main__":
    main()
