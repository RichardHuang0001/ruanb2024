import json
import pandas as pd


# Step 1: 读取 interaction.json 文件，建立 categories id 和类别的映射
def load_interaction_data(json_path):
    with open(json_path, 'r') as f:
        interaction_data = json.load(f)

    # 建立 category_id 和类别名的映射
    category_mapping = {cat["id"]: cat["name"] for cat in interaction_data["categories"]}

    # 创建一个字典，用于快速查找 annotation
    interaction_dict = {
        (annotation["id"], annotation["image_id"]): annotation
        for annotation in interaction_data["annotations"]
    }

    return interaction_data, category_mapping, interaction_dict


# Step 2: 读取 cut_images_annotations.csv，并匹配 interaction.json 的数据
def generate_new_ground_truth(interaction_data, interaction_dict, csv_path):
    # 加载 CSV 文件
    cut_images_annotations = pd.read_csv(csv_path)

    new_annotations = []

    count = 0

    # 遍历每一行，匹配 interaction 数据
    for index, row in cut_images_annotations.iterrows():
        patch_id = row["patch_id"]
        image_id = row["original_image_id"]

        # 查找匹配的 interaction 数据
        interaction_key = (patch_id, image_id)
        if interaction_key in interaction_dict:
            interaction = interaction_dict[interaction_key]
            # 生成新的 annotation
            new_annotation = {
                "id": interaction["id"],
                "image_id": interaction["image_id"],
                "category_id": interaction["category_id"],
                "bbox": interaction["bbox"],
                "area": interaction["area"],
                "iscrowd": interaction.get("iscrowd", 0),
                "segmentation": interaction.get("segmentation", {})
            }
            new_annotations.append(new_annotation)
            count += 1


    # 生成新的 Ground Truth 文件
    new_ground_truth = {
        "images": interaction_data["images"],
        "categories": interaction_data["categories"],
        "annotations": new_annotations
    }

    print(f"共生成{count}条数据")

    return new_ground_truth


# Step 3: 保存新的 Ground Truth 文件
def save_new_ground_truth(new_ground_truth, output_path):
    with open(output_path, 'w') as f:
        json.dump(new_ground_truth, f, indent=4)
    print(f"New ground truth saved to {output_path}")


# 主函数：执行所有步骤
def main():
    interaction_json_path = 'dataset/coco_merged/annotations/interaction.json'
    cut_images_csv_path = 'dataset/cut_images_annotations.csv'
    new_ground_truth_path = 'dataset/new_ground_truth.json'

    # 读取并加载数据
    interaction_data, category_mapping, interaction_dict = load_interaction_data(interaction_json_path)

    # 生成新的 Ground Truth
    new_ground_truth = generate_new_ground_truth(interaction_data, interaction_dict, cut_images_csv_path)

    # 保存生成的文件
    save_new_ground_truth(new_ground_truth, new_ground_truth_path)


# 运行主函数
if __name__ == "__main__":
    main()
