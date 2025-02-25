import os

# Define the base directory
base_dir = "archive/train"

# Maximum allowed images per subdirectory
MAX_IMAGES = 200

# Function to enforce the image limit in each subdirectory
def enforce_image_limit(directory):
    for root, _, files in os.walk(directory):
        # Filter only image files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # If images exceed the limit, sort by creation time and delete the oldest
        if len(image_files) > MAX_IMAGES:
            full_paths = [os.path.join(root, f) for f in image_files]

            # Sort images by creation time (oldest first)
            full_paths.sort(key=os.path.getctime)

            # Calculate how many need to be deleted
            excess_count = len(full_paths) - MAX_IMAGES

            # Delete excess images
            for i in range(excess_count):
                os.remove(full_paths[i])
                print(f"Deleted: {full_paths[i]} (Exceeded limit)")

# Run the function
if __name__ == "__main__":
    enforce_image_limit(base_dir)
