import os
import shutil
import random

# Define paths
train_dir = "archive/train"
test_dir = "archive/test"

# Create the test directory if it doesn't exist
os.makedirs(test_dir, exist_ok=True)

# Iterate over all subdirectories in train
for subfolder in os.listdir(train_dir):
    train_subdir = os.path.join(train_dir, subfolder)
    test_subdir = os.path.join(test_dir, subfolder)

    # Ensure it's a directory
    if os.path.isdir(train_subdir):
        os.makedirs(test_subdir, exist_ok=True)  # Create corresponding test subdir

        # Get all image files in the subdirectory
        images = [f for f in os.listdir(train_subdir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # Select 20 random images (or all if there are fewer than 20)
        images_to_move = random.sample(images, min(20, len(images)))

        # Move selected images from train to test
        for img in images_to_move:
            src_path = os.path.join(train_subdir, img)
            dest_path = os.path.join(test_subdir, img)
            shutil.move(src_path, dest_path)  # Move the file

        print(f"Moved {len(images_to_move)} images from {train_subdir} to {test_subdir}")

print("Dataset split complete!")
