import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn.functional as F
import pickle
import io
from torch import nn
from facenet_pytorch import InceptionResnetV1
import numpy as np
from tqdm import tqdm
import cv2
from ultralytics import YOLO

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
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512, 1)

    def forward(self, img1, img2):
        emb1 = self.arcface(img1)
        emb2 = self.arcface(img2)
        distance = torch.abs(emb1 - emb2)
        distance = self.dropout(distance)
        output = self.fc(distance)
        return output

# --- Load Model from Pickle ---
with open("siamese_model.pkl", "rb") as f:
    model = CPU_Unpickler(f).load()

model.to("cpu")
model.eval()

# Load YOLO face detection model
yolo_model = YOLO("yolov8n-face.pt")

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

options_folder = "options"

st.title("Face Matching App")
st.write("Upload an image to find similar faces from the options folder.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Detect face using YOLO
    results = yolo_model(img_cv)
    if not results or len(results[0].boxes) == 0:
        st.error("No face detected! Please upload a clear image with a visible face.")
    else:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        face_crop = img_cv[y1:y2, x1:x2]
        face_crop = cv2.resize(face_crop, (112, 112))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_crop)
        input_tensor = transform(face_pil).unsqueeze(0)

        # Compute embedding
        with torch.no_grad():
            input_embedding = model.arcface(input_tensor)

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Face detected! Now comparing...")

        matching_images = []

        # Compare with options
        option_images = [f for f in os.listdir(options_folder) if f.lower().endswith((".png", "jpg", "jpeg"))]
        for img_name in tqdm(option_images, desc="Comparing Faces"):
            option_image_path = os.path.join(options_folder, img_name)
            option_image = Image.open(option_image_path).convert("RGB")
            option_tensor = transform(option_image).unsqueeze(0)

            with torch.no_grad():
                option_embedding = model.arcface(option_tensor)
                similarity = F.cosine_similarity(input_embedding, option_embedding).item()

            if similarity > 0.5:
                matching_images.append((img_name, similarity))

        if matching_images:
            st.write("### Matching Faces:")
            for img_name, similarity in matching_images:
                st.image(os.path.join(options_folder, img_name), caption=f"Match: {similarity:.2f}", use_column_width=True)
        else:
            st.write("No matching faces found with similarity > 65%.")