import torch

# Load the checkpoint
model_path = "78good.pth"
checkpoint = torch.load(model_path, map_location="cpu")

# Print the keys in the checkpoint
print("Keys inside .pth file:", checkpoint.keys())

# Check if it contains model weights or full model
if "model_state_dict" in checkpoint:
    print("✅ This file contains a full model state dict (e.g., SiameseArcFace).")
elif isinstance(checkpoint, dict) and any(k.startswith("fc") or k.startswith("arcface") for k in checkpoint.keys()):
    print("✅ This file contains model weights (e.g., for InceptionResnetV1).")
elif isinstance(checkpoint, torch.nn.Module):
    print("✅ This file contains the entire model object.")
else:
    print("⚠️ Unknown format. You might need to check how the model was saved.")
