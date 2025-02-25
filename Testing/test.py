import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn.functional as F
import pickle
import io
from torch import nn
from facenet_pytorch import InceptionResnetV1  # ArcFace backbone

# --- Custom Unpickler for CPU ---
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

# --- Define SiameseArcFace Class ---
class SiameseArcFace(nn.Module):
    def __init__(self):
        super(SiameseArcFace, self).__init__()
        self.arcface = InceptionResnetV1(pretrained='vggface2').eval()
        self.dropout = nn.Dropout(p=0.3)  # Helps prevent overfitting
        self.fc = nn.Linear(512, 1)

    def forward(self, img1, img2):
        emb1 = self.arcface(img1)
        emb2 = self.arcface(img2)
        distance = torch.abs(emb1 - emb2)
        distance = self.dropout(distance)  # Apply dropout before classification
        output = self.fc(distance)
        return output

# --- Load Model from Pickle (CPU Safe) ---
with open("siamese_model.pkl", "rb") as f:
    model = CPU_Unpickler(f).load()  # Use custom unpickler to force CPU

model.to("cpu")  # Ensure model is on CPU
model.eval()  # Set to evaluation mode

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Paths ---
input_folder = "input"
options_folder = "options"

# Get input image
input_images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not input_images:
    print("No images found in input folder!")
    exit()

input_image_path = os.path.join(input_folder, input_images[0])
input_image = Image.open(input_image_path).convert("RGB")
input_tensor = transform(input_image).unsqueeze(0)

# Get option images
option_images = [f for f in os.listdir(options_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not option_images:
    print("No images found in options folder!")
    exit()

# --- Compute Similarity ---
print(f"\nComparing input image: {input_images[0]}\n")
with torch.no_grad():
    for img_name in option_images:
        option_image_path = os.path.join(options_folder, img_name)
        option_image = Image.open(option_image_path).convert("RGB")
        option_tensor = transform(option_image).unsqueeze(0)

        # Compute embeddings on CPU
        input_embedding = model.arcface(input_tensor)
        option_embedding = model.arcface(option_tensor)

        # Calculate cosine similarity
        similarity = F.cosine_similarity(input_embedding, option_embedding).item()

        print(f"Similarity with {img_name}: {similarity:.4f}")

print("\n✅ Comparison Complete!")
