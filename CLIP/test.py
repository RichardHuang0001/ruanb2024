import json

# Load the ground truth and model results
with open('dataset/new_ground_truth_2.json', 'r') as f:
    gt_data = json.load(f)

with open('dataset/CLIP_results_2.json', 'r') as f:
    model_results = json.load(f)

# Extract all category IDs from both files
gt_category_ids = {cat['id'] for cat in gt_data['categories']}
model_category_ids = {res['category_id'] for res in model_results}

# Check for inconsistencies
missing_in_gt = model_category_ids - gt_category_ids
print(f"Category IDs missing in ground truth categories: {missing_in_gt}")

# Check for any discrepancies
if missing_in_gt:
    print("Please check the categories mapping between ground truth and model results.")
else:
    print("All category IDs in model results match ground truth categories.")
