import json
import pandas as pd

# Step 1: Load the existing new_ground_truth.json file
def load_ground_truth(json_path):
    with open(json_path, 'r') as f:
        ground_truth_data = json.load(f)
    return ground_truth_data

# Step 2: Read the similarity_score.csv file to find the top-1 category and score for each patch_id
def process_similarity_scores(csv_path, category_mapping):
    similarity_scores = pd.read_csv(csv_path)
    patch_predictions = {}

    # Find top-1 category and score for each patch_id based on similarity scores
    for _, row in similarity_scores.iterrows():
        patch_id = int(row['patch_id'])

        # Find the highest similarity score category
        scores = row[1:]  # Exclude patch_id column
        top_category_name = scores.idxmax()
        top_score = scores.max()  # Get the highest similarity score

        # Look up the category ID from the category mapping
        if top_category_name in category_mapping:
            top_category_id = category_mapping[top_category_name]
            patch_predictions[patch_id] = (top_category_id, float(top_score))

    return patch_predictions

# Step 3: Generate model results in the correct COCO format
def generate_model_results(ground_truth_data, patch_predictions, output_path):
    model_results = []

    # Iterate over ground truth annotations to create model prediction entries
    for annotation in ground_truth_data['annotations']:
        patch_id = annotation['id']
        image_id = annotation['image_id']

        if patch_id in patch_predictions:
            # Prepare the prediction result with the required fields
            predicted_category_id, score = patch_predictions[patch_id]
            result_entry = {
                "image_id": image_id,
                "category_id": predicted_category_id,
                "bbox": annotation['bbox'],
                "score": score  # Use the actual similarity score
            }
            model_results.append(result_entry)

    # Save the model results to a JSON file
    with open(output_path, 'w') as f:
        json.dump(model_results, f, indent=4)
    print(f"Model results saved to {output_path}")

# Main function: Execute all steps
def main():
    ground_truth_json_path = 'dataset/sep_test/ingames/interaction_ingame.json'  # Replace with actual path
    similarity_csv_path = 'dataset/sep_test/ingames/sim_scores_ingame.csv'  # Replace with actual path
    model_results_path = 'dataset/sep_test/ingames/CLIP_results_ingame.json'  # Replace with actual save path

    # Load ground truth data
    ground_truth_data = load_ground_truth(ground_truth_json_path)

    # Create category name-to-ID mapping
    category_mapping = {cat["name"]: cat["id"] for cat in ground_truth_data["categories"]}

    # Process similarity scores to get top-1 predicted categories and scores for each patch_id
    patch_predictions = process_similarity_scores(similarity_csv_path, category_mapping)

    # Generate model results in the required format
    generate_model_results(ground_truth_data, patch_predictions, model_results_path)

# Run the main function
if __name__ == "__main__":
    main()
