import os
import json
import cv2
import csv
import pandas as pd  # 导入 pandas 库

# 文件路径配置
INTERACTION_ANNOTATION_PATH = 'dataset/sep_test/ingames/interaction_ingame.json'
SEMANTIC_ANNOTATION_PATH = 'dataset/coco_merged/annotations/semantics.json'
IMAGE_FOLDER = 'dataset/coco_merged/images/interaction'
OUTPUT_FOLDER = 'dataset/sep_test/ingames/cut_images_ingames'
CSV_OUTPUT_FILE = 'dataset/sep_test/ingames/cut_images_annotations_ingames.csv'
ERROR_LOG_FILE = 'dataset/sep_test/ingames/ingames_error_log.txt'  # 错误日志文件路径

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

# 初始化错误计数器和错误信息列表
error_count = 0
error_log = []

# 用于模糊比较的函数
def is_bbox_similar(bbox1, bbox2, threshold=0.1):
    """
    比较两个 bbox 是否相似。
    :param bbox1: 第一个 bbox, 格式为 [x, y, width, height]
    :param bbox2: 第二个 bbox, 格式为 [x, y, width, height]
    :param threshold: 允许的最大差异百分比 (默认 10%)
    :return: 如果相似，返回 True，否则返回 False
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # 计算每个维度上的差异
    x_diff = abs(x1 - x2)
    y_diff = abs(y1 - y2)
    w_diff = abs(w1 - w2)
    h_diff = abs(h1 - h2)

    # 判断是否在允许的阈值范围内
    if (x_diff / w1 <= threshold and y_diff / h1 <= threshold and
            w_diff / w1 <= threshold and h_diff / h1 <= threshold):
        return True
    return False

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
            interaction_bbox = interaction_annotation['bbox']
            x, y, width, height = map(int, interaction_bbox)

            # 在原图上切割出 bbox 区域
            patch = image[y:y + height, x:x + width]

            # 初始化匹配标志为 False
            found_matching_semantic = False

            # 遍历该图片的所有 semantic 标注，并进行 bbox 匹配判断
            for semantic_annotation in semantic_annotations:
                semantic_bbox = semantic_annotation['bbox']

                # 使用模糊判断函数比较 bbox 是否相似
                if is_bbox_similar(semantic_bbox, interaction_bbox):
                    found_matching_semantic = True  # 找到匹配的标注

                    # 获取 semantic 的类别名
                    semantic_category_id = semantic_annotation['category_id']
                    semantic_category_name = semantic_categories[semantic_category_id]

                    # 生成切割后的图片文件名
                    patch_id = f"{interaction_annotation['id']}.jpg"
                    patch_path = os.path.join(OUTPUT_FOLDER, patch_id)

                    # 保存切割后的图片
                    cv2.imwrite(patch_path, patch)

                    # 写入 CSV 文件中的一行数据
                    csv_writer.writerow([
                        interaction_annotation['id'], image_id,
                        interaction_category_name, semantic_category_name,
                        interaction_bbox
                    ])
                    break  # 找到匹配的 semantic 标注后退出循环

            # 如果遍历完所有 semantic 标注仍未找到匹配的标注
            if not found_matching_semantic:
                error_count += 1  # 增加错误计数
                error_log.append(
                    f"图片 ID: {image_id}, Interaction 标注: {interaction_annotation}\n"
                )

# 将错误日志写入文件
with open(ERROR_LOG_FILE, 'w') as f:
    f.write(f"总共找不到匹配的 patch 数量: {error_count}\n\n")
    f.writelines(error_log)

# 去重逻辑：使用 pandas 读取并去重
df = pd.read_csv(CSV_OUTPUT_FILE)
before_row_count = len(df)
df = df.drop_duplicates()
after_row_count = len(df)
print(f'csv标注文件长度:{after_row_count}')
# 输出去除的重复行数
print(f"Number of duplicate rows removed: {before_row_count - after_row_count}")
# 将去重后的数据覆盖写回原 CSV 文件
df.to_csv(CSV_OUTPUT_FILE, index=False)


print(f"所有图片已切割并保存至 {OUTPUT_FOLDER}")
print(f"CSV 文件已生成并去重：{CSV_OUTPUT_FILE}")
print(f"错误日志已保存至：{ERROR_LOG_FILE}")
