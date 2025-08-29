import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def extract_patch_embeddings(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.vision_model(**inputs)
        patch_tokens = outputs.last_hidden_state.squeeze(0)[1:]  # remove CLS token
    return patch_tokens.cpu().numpy()

def process_images(img_dir, out_dir):
    img_dir = Path(img_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))):
        try:
            patch_embeds = extract_patch_embeddings(image_path)
            name = image_path.stem  # e.g., 01_0123
            np.save(out_dir / f"{name}.npy", patch_embeds)
        except Exception as e:
            print(f"Failed {image_path}: {e}")

if __name__ == "__main__":
    process_images(
        img_dir="data/images/frieren/01",
        out_dir="data/embeddings/patches/vol_01"
    )
