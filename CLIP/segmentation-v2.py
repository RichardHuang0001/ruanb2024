import os
import json
import cv2
import csv
import pandas as pd  # 导入 pandas 库

# 文件路径配置
INTERACTION_ANNOTATION_PATH = 'dataset/coco_merged/annotations/interaction.json'
SEMANTIC_ANNOTATION_PATH = 'dataset/coco_merged/annotations/semantics.json'
IMAGE_FOLDER = 'dataset/coco_merged/images/interaction'
OUTPUT_FOLDER = 'dataset/cut_images'
CSV_OUTPUT_FILE = 'dataset/cut_images_annotations.csv'

# 创建输出文件夹
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 加载标注文件
def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        return json.load(f)

interaction_data = load_annotations(INTERACTION_ANNOTATION_PATH)
semantic_data = load_annotations(SEMANTIC_ANNOTATION_PATH)

# 建立 categories 映射：id -> name
def build_category_map(data):
    return {category['id']: category['name'] for category in data['categories']}

interaction_categories = build_category_map(interaction_data)
semantic_categories = build_category_map(semantic_data)

# 将所有 semantic 标注按照 image_id 建立索引，方便快速查找
semantic_annotations_by_image = {}
for annotation in semantic_data['annotations']:
    image_id = annotation['image_id']
    if image_id not in semantic_annotations_by_image:
        semantic_annotations_by_image[image_id] = []
    semantic_annotations_by_image[image_id].append(annotation)

# 创建并初始化 CSV 文件
with open(CSV_OUTPUT_FILE, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # 写入 CSV 文件的表头
    csv_writer.writerow([
        'patch_id', 'original_image_id', 'interaction_category', 'semantic_category',
        'interaction_bbox'
    ])

    # 遍历每张图片的 interaction 标注
    for image_info in interaction_data['images']:
        image_id = image_info['id']
        image_path = os.path.join(IMAGE_FOLDER, image_info['file_name'])

        # 加载原始图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法加载图像: {image_path}")
            continue

        # 查找该图片的 semantic 标注
        semantic_annotations = semantic_annotations_by_image.get(image_id, [])

        # 遍历这张图片的所有 interaction 标注
        for interaction_annotation in interaction_data['annotations']:
            if interaction_annotation['image_id'] != image_id:
                continue

            # 获取 interaction 的类别名和 bbox
            interaction_category_id = interaction_annotation['category_id']
            interaction_category_name = interaction_categories[interaction_category_id]
            bbox = interaction_annotation['bbox']
            x, y, width, height = map(int, bbox)

            # 在原图上切割出 bbox 区域
            patch = image[y:y + height, x:x + width]

            # 处理该图片的所有 semantic 标注
            for semantic_annotation in semantic_annotations:
                semantic_category_id = semantic_annotation['category_id']
                semantic_category_name = semantic_categories[semantic_category_id]

                # 生成切割后的图片文件名
                patch_id = f"{interaction_annotation['id']}_{image_id}_{interaction_category_name}_{semantic_category_name}.jpg"
                patch_path = os.path.join(OUTPUT_FOLDER, patch_id)

                # 保存切割后的图片
                cv2.imwrite(patch_path, patch)

                # 写入 CSV 文件中的一行数据
                csv_writer.writerow([
                    interaction_annotation['id'], image_id,
                    interaction_category_name, semantic_category_name,
                    bbox
                ])

# 去重逻辑：使用 pandas 读取并去重
df = pd.read_csv(CSV_OUTPUT_FILE)  # 读取生成的 CSV 文件
df = df.drop_duplicates()  # 去除重复行
df.to_csv(CSV_OUTPUT_FILE, index=False)  # 将去重后的数据覆盖写回原 CSV 文件

print(f"所有图片已切割并保存至 {OUTPUT_FOLDER}")
print(f"CSV 文件已生成并去重：{CSV_OUTPUT_FILE}")
