import base64
from pathlib import Path
import pandas as pd

# Assuming you have filenames and matching fan names
image_folder = Path("real_fan_images")  # folder containing real fan images
df = pd.read_excel("usha_fans_dataset.xlsx")

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Match image filenames to fan names (by order or mapping)
for i in range(len(df)):
    fan_name = df.at[i, "Fan Name"]
    image_path = image_folder / f"{fan_name.lower().replace(' ', '_')}.jpg"
    if image_path.exists():
        df.at[i, "Image (base64)"] = image_to_base64(image_path)
    else:
        print(f"Image not found for: {fan_name}")

df.to_excel("usha_fans_dataset_with_real_images.xlsx", index=False)
