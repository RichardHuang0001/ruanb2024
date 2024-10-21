import os
import json
import base64
import csv
from zhipuai import ZhipuAI
from zhipuai.core._errors import APIRequestFailedError

def sort_files_by_number(filename):
    """ Extracts numbers from filename and converts to integer for sorting. """
    num_part = ''.join(filter(str.isdigit, filename))
    return int(num_part) if num_part.isdigit() else float('inf')

# Set up dataset directory and CSV output path
img_directory_path = "dataset/device/cut_images_device"
output_csv_path = "dataset/device/GLM_results_device.csv"
error_log_path = "dataset/device/error_log.csv"
progress_file_path = "dataset/device/progress_log.txt"
annotation_file = 'dataset/device/interaction_device.json'  # JSON 文件路径

# 读取类别映射并生成 interaction_categories 列表
with open(annotation_file, 'r') as f:
    annotation_data = json.load(f)
    interaction_categories = [category['name'].lower() for category in annotation_data['categories']]

# Initialize ZhipuAI client
client = ZhipuAI(api_key="3ccec0493319b6538bc901a76ae85ed9.3yCz7b4Y36Hp3yn0")

# Open or create CSV file in append mode
mode = 'a' if os.path.exists(output_csv_path) else 'w'
with open(output_csv_path, mode=mode, newline='') as csv_file, open(error_log_path, mode='a', newline='') as error_log_file:
    csv_writer = csv.writer(csv_file)
    error_log_writer = csv.writer(error_log_file)

    if mode == 'w':
        csv_writer.writerow(["patch_id", "predicted_category"])
        error_log_writer.writerow(["patch_id", "unexpected_category"])

    # Get all images and sort them by numerical order in filenames
    image_files = sorted(
        [img for img in os.listdir(img_directory_path) if img.endswith(('.jpg', '.png', '.jpeg'))],
        key=sort_files_by_number
    )

    # Check if there's a progress log and read it
    last_index = 0
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r') as progress_file:
            last_index = int(progress_file.read().strip())

    # Loop through each image starting from the last index
    total_count = len(image_files)
    for index, img_name in enumerate(image_files[last_index:], start=last_index):
        patch_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_directory_path, img_name)

        # Read the image and convert it to Base64
        with open(img_path, 'rb') as img_file:
            img_base = base64.b64encode(img_file.read()).decode('utf-8')

        try:
            # Call the model, sending the image and task description
            response = client.chat.completions.create(
                model="glm-4v",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img_base
                                }
                            },
                            {
                                "type": "text",
                                "text": "This is a picture of an interactable object in an XR Game. Predict the interaction category for the given image and return the category name only. The output must be one of the following categories:\
                                         {" + ",".join(interaction_categories) + "}\
                                         Example return 1: shoot,Example return 2:knock"
                            }
                        ]
                    }
                ]
            )

            # Extract and save the predicted category
            predicted_category = response.choices[0].message.content.strip().lower()
            if predicted_category in interaction_categories:
                csv_writer.writerow([patch_id, predicted_category])
                print(f"({index}/{total_count})Successfully categorized patch_id {patch_id}: {predicted_category}")
                with open(progress_file_path, 'w') as progress_file:
                    progress_file.write(str(index + 1))
            else:
                error_log_writer.writerow([patch_id, predicted_category])
                print(f"Unexpected category for patch_id {patch_id}: {predicted_category}")

        except APIRequestFailedError as e:
            # Log the error and skip the image
            print(f"APIRequestFailedError for patch_id {patch_id}: {e}")
            with open(progress_file_path, 'w') as progress_file:
                progress_file.write(str(index + 1))
            continue  # Skip this image and move to the next one

        except Exception as e:
            # Catch all other exceptions and log the error
            print(f"An error occurred with patch_id {patch_id}: {e}")
            with open(progress_file_path, 'w') as progress_file:
                progress_file.write(str(index + 1))
            continue  # Skip this image and move to the next one

print(f"Predictions and errors have been logged. Results saved to {output_csv_path}. Errors logged to {error_log_path}.")

# Function to read and analyze the CSV for duplicates and total count
def analyze_csv(csv_path):
    with open(csv_path, mode='r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        data = list(reader)
    total_rows = len(data)
    unique_rows = len(set(tuple(row) for row in data))
    duplicate_rows = total_rows - unique_rows
    print(f"Total rows: {total_rows}, Duplicate rows: {duplicate_rows}")

analyze_csv(output_csv_path)
