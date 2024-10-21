import json
import csv

# 文件路径
annotation_file = 'dataset/device/interaction_device.json'  # 标注文件路径
csv_result_file = 'dataset/device/GLM_results_device.csv'  # 新测试结果的CSV文件路径
output_ground_truth_file = 'dataset/device/glm_interaction_device.json'  # 输出的Ground Truth文件
output_prediction_file = 'dataset/device/GLM_results_device.json'  # 输出的Predictions文件

# 1. 读取标注文件和类别映射
with open(annotation_file, 'r') as f:
    annotation_data = json.load(f)

# 从 'categories' 中创建类别映射
category_mapping = {category['name']: category['id'] for category in annotation_data['categories']}

# 2. 读取CSV文件中的测试结果
test_results = []
with open(csv_result_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        test_results.append({'patch_id': int(row['patch_id']), 'predicted_category': row['predicted_category']})

# 3. 创建用于评估的Ground Truth和Predictions
ground_truth = []
predictions = []

# 遍历所有测试结果并生成对应的Ground Truth和预测结果
for result in test_results:
    patch_id = result['patch_id']
    predicted_category = result['predicted_category']

    # 在标注文件中寻找匹配的patch_id
    for annotation in annotation_data['annotations']:
        if annotation['id'] == patch_id:
            # 创建Ground Truth条目
            ground_truth.append({
                "image_id": annotation['image_id'],
                "category_id": annotation['category_id'],
                "bbox": annotation['bbox'],
                "iscrowd": annotation.get('iscrowd', 0)
            })

            # 创建Prediction条目，使用从JSON读取的类别映射
            predictions.append({
                "image_id": annotation['image_id'],
                "category_id": category_mapping.get(predicted_category, 1),  # 使用类别映射获取ID
                "bbox": annotation['bbox'],  # 假设bbox从标注中保留
                "score": 1.0  # 假设预测分数为1.0
            })
            break  # 找到对应的patch_id后跳出循环

# 4. 将Ground Truth和Predictions写入文件
with open(output_ground_truth_file, 'w') as f:
    json.dump(ground_truth, f, indent=4)

with open(output_prediction_file, 'w') as f:
    json.dump(predictions, f, indent=4)

print(f"Ground Truth文件已生成: {output_ground_truth_file}")
print(f"预测结果文件已生成: {output_prediction_file}")
