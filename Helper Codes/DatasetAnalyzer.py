import os

# Define the base directory
base_dir = "archive/train"

# Initialize variables
folder_stats = {}

# Iterate through all subfolders in the base directory
for root, _, files in os.walk(base_dir):
    if root == base_dir:
        continue  # Skip the base directory itself
    
    # Count valid image files in the folder
    image_count = sum(1 for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')))
    
    # Store the count for this folder
    folder_stats[root] = image_count

# Total number of subfolders
total_folders = len(folder_stats)

# Average number of images per folder
average_images = sum(folder_stats.values()) / total_folders if total_folders > 0 else 0

# Count folders with less than 100 and 150 images
folders_below_100 = sum(1 for count in folder_stats.values() if count < 100)
folders_below_150 = sum(1 for count in folder_stats.values() if count < 200)

# Print the results
print(f"Total subfolders: {total_folders}")
print(f"Average images per folder: {average_images:.2f}")
print(f"Folders with <100 images: {folders_below_100}")
print(f"Folders with <150 images: {folders_below_150}")
