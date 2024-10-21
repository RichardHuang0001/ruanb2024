import json

# Step 1: 读取 ground truth 数据
def load_ground_truth(json_path):
    with open(json_path, 'r') as f:
        ground_truth_data = json.load(f)
    return ground_truth_data

# Step 2: 根据 ground truth 生成高匹配的模型结果
def generate_high_match_results(ground_truth_data, output_path):
    model_results = []

    # 遍历 ground truth 中的 annotations，生成模型结果
    for annotation in ground_truth_data['annotations']:
        result_entry = {
            "image_id": annotation['image_id'],
            "category_id": annotation['category_id'],
            "bbox": annotation['bbox'],
            "score": 0.99  # 设置高置信度
        }
        model_results.append(result_entry)

    # 保存模型结果到 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(model_results, f, indent=4)
    print(f"High-match model results saved to {output_path}")

# 主函数：执行所有步骤
def main():
    ground_truth_json_path = 'dataset/new_ground_truth_2.json'
    output_results_path = 'dataset/example.json'

    # 加载 ground truth 数据
    ground_truth_data = load_ground_truth(ground_truth_json_path)

    # 生成高匹配模型结果
    generate_high_match_results(ground_truth_data, output_results_path)

# 运行主函数
if __name__ == "__main__":
    main()
