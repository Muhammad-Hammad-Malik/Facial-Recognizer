import os
import cv2
import numpy as np
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import imgaug.augmenters as iaa

# Define paths
train_dir = "archive/train"
TARGET_IMAGES = 500  # Target images per class

# Define augmentations with increased variety
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50% chance of horizontal flip
    iaa.Flipud(0.2),  # 20% chance of vertical flip
    iaa.Affine(
        rotate=(-15, 15),  # Rotation (-15° to +15°)
        shear=(-8, 8),  # Shearing (-8° to +8°)
        scale=(0.8, 1.2)  # Random scaling (80% to 120%)
    ),
    iaa.PerspectiveTransform(scale=(0.02, 0.05)),  # Slight perspective distortion
    iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.25),  # Warping effect
    iaa.AddToHueAndSaturation((-20, 20)),  # Random hue/saturation shifts
    iaa.AddToBrightness((-40, 40)),  # Vary brightness
    iaa.contrast.LinearContrast((0.5, 2.0)),  # Contrast variation
    iaa.GaussianBlur(sigma=(0, 1.5)),  # Slight blur
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # Noise injection
    iaa.Cutout(nb_iterations=(1, 3), size=0.2, squared=True),  # Simulated occlusion
    iaa.CoarseDropout(0.02, size_percent=0.1)  # Random pixel dropouts
])

# Function to augment and save an image
def augment_and_save(image_path, save_dir, index):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return

        # Apply augmentation
        augmented_img = augmenters(image=img)

        # Generate new filename
        new_filename = f"aug_{index}.jpg"
        save_path = os.path.join(save_dir, new_filename)

        # Save the augmented image
        cv2.imwrite(save_path, augmented_img)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Process each subfolder
for subfolder in os.listdir(train_dir):
    subfolder_path = os.path.join(train_dir, subfolder)

    if os.path.isdir(subfolder_path):
        # Get all existing images
        images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        image_count = len(images)

        if image_count < TARGET_IMAGES:
            needed = TARGET_IMAGES - image_count
            print(f"Generating {needed} images for {subfolder}")

            # Use multi-threading to speed up augmentation
            with ThreadPoolExecutor(max_workers=16) as executor:  # Increased workers for faster processing
                futures = []
                for i in range(needed):
                    image_path = os.path.join(subfolder_path, random.choice(images))  # Pick a random image
                    futures.append(executor.submit(augment_and_save, image_path, subfolder_path, i + image_count))

                # Wait for all tasks to complete
                for future in tqdm(futures, desc=f"Augmenting {subfolder}", unit="img"):
                    future.result()

print("✅ Dataset balancing complete! Each class has 250 images.")
