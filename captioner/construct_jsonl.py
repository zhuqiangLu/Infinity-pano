import json
import os
from PIL import Image
from tqdm import tqdm

# Load the captions
with open('structured3d_short_captions_fixed.json', 'r') as f:
    captions = json.load(f)

# Create dataset entries
dataset = []

for img_path, caption in tqdm(captions.items()):
    # Verify image exists and can be opened
    try:
        img = Image.open(img_path)
        img.close()
        
        # Create entry
        entry = {
            "image_path": f"data/rename_structured3d/{img_path}",
            "long_caption": caption,
            "long_caption_type": "blip3",
            "h_div_w": 0.5,
        }
        dataset.append(entry)
    except:
        print(f"Could not process {img_path}")
        continue

# Save as JSONL
with open('structured3d_dataset.jsonl', 'w') as f:
    for entry in dataset:
        json.dump(entry, f)
        f.write('\n')

print(f"Created dataset with {len(dataset)} entries")
