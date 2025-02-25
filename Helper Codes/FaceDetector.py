import os
import cv2
import numpy as np
from tqdm import tqdm
from retinaface import RetinaFace

# Define the base directory
base_dir = "input"

# Collect all image file paths
image_paths = []
for root, _, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(root, file))

# Progress bar
progress_bar = tqdm(total=len(image_paths), desc="Processing Images", unit="img")

# Function to detect and crop faces
def process_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping (invalid image): {image_path}")
            return
        
        # Detect faces using RetinaFace
        faces = RetinaFace.detect_faces(image_path)

        if isinstance(faces, dict) and len(faces) > 0:
            # Extract the first detected face
            face_info = list(faces.values())[0]
            facial_area = face_info['facial_area']  # (x1, y1, x2, y2)

            x1, y1, x2, y2 = facial_area

            # Crop the face region
            face_crop = img[y1:y2, x1:x2]

            # Ensure the face is not empty after cropping
            if face_crop.size > 0:
                cv2.imwrite(image_path, face_crop)
                print(f"Cropped and saved: {image_path}")
            else:
                os.remove(image_path)
                print(f"Deleted (empty crop): {image_path}")
        else:
            os.remove(image_path)
            print(f"Deleted (no face detected): {image_path}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Process images one by one with progress bar
for image_path in image_paths:
    process_image(image_path)
    progress_bar.update(1)

# Close progress bar after completion
progress_bar.close()
