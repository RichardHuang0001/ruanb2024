import os
import json
import cv2

# 文件路径设置
ANNOTATION_PATH = 'dataset/coco_merged/annotations/interaction.json'
IMAGE_FOLDER = 'dataset/coco_merged/images/interaction'
OUTPUT_FOLDER = 'dataset/cut_images'
NEW_ANNOTATION_FILE = 'dataset/cut_annotations.json'

# 确保输出文件夹存在
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 加载COCO格式的标注文件
with open(ANNOTATION_PATH, 'r') as f:
    data = json.load(f)

# 初始化新的COCO标注结构
new_annotations = {
    "images": [],
    "annotations": [],
    "categories": data["categories"]  # 保留原始类别信息
}

# 遍历每张图片的信息
for image_info in data['images']:
    image_id = image_info['id']
    image_path = os.path.join(IMAGE_FOLDER, image_info['file_name'])

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        continue

    # 遍历该图片的所有标注
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            # 获取bounding box信息并进行切割
            bbox = annotation['bbox']  # [x, y, width, height]
            x, y, width, height = map(int, bbox)
            cropped_image = image[y:y+height, x:x+width]

            # 生成新图像文件名并保存
            cropped_filename = f"{image_info['file_name'].split('.')[0]}_bbox_{annotation['id']}.jpg"
            cropped_filepath = os.path.join(OUTPUT_FOLDER, cropped_filename)
            cv2.imwrite(cropped_filepath, cropped_image)

            # 添加新的图像信息到新标注文件
            new_image_info = {
                "id": annotation["id"],
                "file_name": cropped_filename,
                "width": width,
                "height": height
            }
            new_annotations["images"].append(new_image_info)

            # 添加新的标注信息到新标注文件
            new_annotation = {
                "id": annotation["id"],
                "image_id": annotation["id"],
                "category_id": annotation["category_id"],
                "bbox": [0, 0, width, height],  # 新bbox从左上角开始
                "area": width * height,
                "iscrowd": annotation["iscrowd"]
            }
            new_annotations["annotations"].append(new_annotation)

# 保存新的COCO格式标注文件
with open(NEW_ANNOTATION_FILE, 'w') as f:
    json.dump(new_annotations, f)

print(f"所有图片已切割并保存至 {OUTPUT_FOLDER}")
print(f"新的COCO格式标注文件已保存至 {NEW_ANNOTATION_FILE}")
