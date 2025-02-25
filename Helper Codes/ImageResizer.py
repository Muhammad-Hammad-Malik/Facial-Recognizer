import os
import cv2
import threading
from tqdm import tqdm

# Define the base directory
base_dir = "input"

# Target image size
TARGET_SIZE = (112, 112)

# List to store image paths for progress tracking
image_paths = []

# Collect all image file paths
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(root, file))

# Progress bar
progress_bar = tqdm(total=len(image_paths), desc="Resizing Images", unit="img")

# Function to resize an image
def resize_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping (invalid image): {image_path}")
            return
        
        # Resize and overwrite the image
        resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, resized_img)

        # Update progress bar
        progress_bar.update(1)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Function to process images in multiple threads
def process_images():
    threads = []

    for image_path in image_paths:
        thread = threading.Thread(target=resize_image, args=(image_path,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    progress_bar.close()

# Run the function
if __name__ == "__main__":
    process_images()
