import torch

model_path = "Notebook/best_model.pth"  # Ensure this is correct
checkpoint = torch.load(model_path, map_location="cpu")

print("Saved Model Keys:\n", checkpoint.keys())  # Check layer names
