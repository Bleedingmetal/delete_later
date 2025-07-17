# delete_later


import os
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# ViM model import (adjusted to match the ViM GitHub repo structure)
from vim.models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 as create_model

# === Configuration ===
model_path = "checkpoints/vim_t_midclstok_76p1acc.pth"  # Your model checkpoint
img_folder = "test_images"  # Folder with input images
output_folder = "results"   # Folder to save annotated outputs
os.makedirs(output_folder, exist_ok=True)

img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Optional: Load ImageNet class labels ===
try:
    with open("imagenet_classes.txt", "r") as f:
        classes = [line.strip() for line in f]
except FileNotFoundError:
    print("⚠️ Warning: imagenet_classes.txt not found — class IDs will be shown instead.")
    classes = None

# === Load model ===
model = create_model()
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

# === Define image transformation ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

# === Inference and Annotation Loop ===
with torch.no_grad():
    for file_name in os.listdir(img_folder):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        input_path = os.path.join(img_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        image = Image.open(input_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        label = classes[pred_class] if classes else f"class {pred_class}"

        # === Annotate image ===
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), label, fill="red", font=font)

        annotated.save(output_path)
        print(f"{file_name} → {label}")
