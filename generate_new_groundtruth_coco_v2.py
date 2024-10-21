import json
import pandas as pd

# Step 1: Load interaction data and establish category ID mapping
def load_interaction_data(json_path):
    with open(json_path, 'r') as f:
        interaction_data = json.load(f)

    # Create a mapping of category_id to category name
    category_mapping = {cat["id"]: cat["name"] for cat in interaction_data["categories"]}

    # Create a dictionary for quick lookup of annotations by (id, image_id)
    interaction_dict = {
        (annotation["id"], annotation["image_id"]): annotation
        for annotation in interaction_data["annotations"]
    }

    return interaction_data, category_mapping, interaction_dict

# Step 2: Generate new ground truth based on cut images annotations
def generate_new_ground_truth(interaction_data, interaction_dict, csv_path):
    # Load the CSV file
    cut_images_annotations = pd.read_csv(csv_path)

    new_annotations = []
    count = 0

    # Iterate over each row, matching with interaction data
    for index, row in cut_images_annotations.iterrows():
        patch_id = int(row["patch_id"])  # Ensure patch_id is treated as an integer
        image_id = int(row["original_image_id"])  # Ensure image_id is treated as an integer

        # Find matching interaction data
        interaction_key = (patch_id, image_id)
        if interaction_key in interaction_dict:
            interaction = interaction_dict[interaction_key]
            # Generate new annotation
            new_annotation = {
                "id": interaction["id"],
                "image_id": interaction["image_id"],
                "category_id": interaction["category_id"],
                "bbox": interaction["bbox"],
                "area": interaction["area"],
                "iscrowd": interaction.get("iscrowd", 0),  # Default to 0 if not present
                "segmentation": interaction.get("segmentation", {})  # Include segmentation if available
            }
            new_annotations.append(new_annotation)
            count += 1

    # Create new ground truth structure
    new_ground_truth = {
        "images": interaction_data["images"],
        "categories": interaction_data["categories"],
        "annotations": new_annotations
    }

    print(f"Total {count} annotations generated")

    return new_ground_truth

# Step 3: Save the new ground truth JSON file
def save_new_ground_truth(new_ground_truth, output_path):
    with open(output_path, 'w') as f:
        json.dump(new_ground_truth, f, indent=4)
    print(f"New ground truth saved to {output_path}")

# Main function to execute all steps
def main():
    interaction_json_path = 'dataset/coco_merged/annotations/interaction.json'  # Adjust the path as needed
    cut_images_csv_path = 'dataset/cut_images_annotations.csv'  # Adjust the path as needed
    new_ground_truth_path = 'dataset/new_ground_truth_2.json'  # Adjust the path as needed

    # Load interaction data
    interaction_data, category_mapping, interaction_dict = load_interaction_data(interaction_json_path)

    # Generate the new ground truth
    new_ground_truth = generate_new_ground_truth(interaction_data, interaction_dict, cut_images_csv_path)

    # Save the generated ground truth file
    save_new_ground_truth(new_ground_truth, new_ground_truth_path)

# Run the main function
if __name__ == "__main__":
    main()
