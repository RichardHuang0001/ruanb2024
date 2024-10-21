import json
import pandas as pd


# Step 1: 读取 interaction.json 文件，建立 categories id 和类别的映射
def load_interaction_data(json_path):
    with open(json_path, 'r') as f:
        interaction_data = json.load(f)

    # 建立 category_id 和类别名的映射
    category_mapping = {cat["name"]: cat["id"] for cat in interaction_data["categories"]}
    id_to_category = {cat["id"]: cat["name"] for cat in interaction_data["categories"]}

    return interaction_data, category_mapping, id_to_category


# Step 2: 读取 similarity_score.csv 文件，并根据相似度得分选择 top-1 类别
def process_similarity_scores(csv_path, category_mapping):
    similarity_scores = pd.read_csv(csv_path)
    results = []

    # 遍历每行，找出每个 patch_id 的 top-1 类别
    for index, row in similarity_scores.iterrows():
        patch_id = row['patch_id']

        # 找出最高相似度类别
        scores = row[1:]  # 排除掉 patch_id 列
        top_category_name = scores.idxmax()
        top_score = scores.max()

        # 根据类别名查找 category_id
        if top_category_name in category_mapping:
            top_category_id = category_mapping[top_category_name]

            results.append({
                "patch_id": patch_id,
                "category_id": top_category_id,
                "category_name": top_category_name,
                "score": top_score
            })

    return results


# Step 3: 根据相似度得分生成测试结果的 COCO 格式 .json 文件
def generate_model_results(interaction_data, results, output_path):
    # 用于生成 annotations
    model_annotations = []

    # 创建一个查找表，以便快速找到 patch_id 对应的 interaction annotation 信息
    interaction_dict = {
        (annotation["id"], annotation["image_id"]): annotation
        for annotation in interaction_data["annotations"]
    }

    # 遍历相似度结果，生成模型结果文件
    for result in results:
        patch_id = result["patch_id"]
        category_id = result["category_id"]
        score = result["score"]

        # 找到原始 annotation 中对应的 patch_id 以获取 image_id 和 bbox
        for key, interaction in interaction_dict.items():
            if interaction["id"] == patch_id:
                model_annotation = {
                    "image_id": interaction["image_id"],
                    "category_id": category_id,
                    "bbox": interaction["bbox"],
                    "score": score,
                    "id": patch_id  # 重新使用 patch_id 作为标识符
                }
                model_annotations.append(model_annotation)
                break

    # 准备 COCO 格式的模型结果 JSON 文件
    model_results = {
        "images": interaction_data["images"],
        "categories": interaction_data["categories"],
        "annotations": model_annotations
    }

    # 保存到指定路径
    with open(output_path, 'w') as f:
        json.dump(model_results, f, indent=4)
    print(f"Model results saved to {output_path}")


# 主函数：执行所有步骤
def main():
    interaction_json_path = 'dataset/coco_merged/annotations/interaction.json'
    similarity_csv_path = 'dataset/similarity_scores.csv'
    model_results_path = 'dataset/CLIP_model_results.json'

    # 加载 interaction 数据
    interaction_data, category_mapping, id_to_category = load_interaction_data(interaction_json_path)

    # 处理 similarity scores，选择 top-1 类别
    similarity_results = process_similarity_scores(similarity_csv_path, category_mapping)

    # 生成模型结果文件
    generate_model_results(interaction_data, similarity_results, model_results_path)


# 运行主函数
if __name__ == "__main__":
    main()
