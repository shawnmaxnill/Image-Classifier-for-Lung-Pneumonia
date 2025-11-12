import os
import shutil
import pandas as pd
from dotenv import load_dotenv

# Loading paths
load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
METADATA_FILE = os.getenv("METADATA_FILE")
TRAIN_IMAGE_DIR = os.getenv("TRAIN_IMAGE_DIR")
TEST_IMAGE_DIR = os.getenv("TEST_IMAGE_DIR")
ORG_IMAGE_DIR = os.getenv("OUTPUT_IMAGE_DIR")

# Loading in metadata
metadata = pd.read_csv(METADATA_FILE)
print("Metadata Loaded...")

# Creating output folders
for dataset_type in metadata['Dataset_type'].unique():
    for label in metadata['Label'].unique():
        folder_path = os.path.join(ORG_IMAGE_DIR, dataset_type, label) # Creating path names
        os.makedirs(folder_path, exist_ok=True) # Creating directory itself
        print(f"New directory created at: {folder_path}")

# Function to return source image path based on metadata type
def get_image_path(row):
    if row['Dataset_type'].upper() == "TRAIN":
        return os.path.join(TRAIN_IMAGE_DIR, row['X_ray_image_name'])
    elif row['Dataset_type'].upper() == "TEST":
        return os.path.join(TEST_IMAGE_DIR, row['X_ray_image_name'])
    else:
        return None
    
# Move images to their respective folders
print("Moving files... This might take a while")

for _, row in metadata.iterrows():
    src_path = get_image_path(row)
    if src_path is None or not os.path.exists(src_path):
        print(f"Skipping missing file: {row['X_ray_image_name']}")
        continue

    dst_path = os.path.join(ORG_IMAGE_DIR, row['Dataset_type'], row['Label'], row['X_ray_image_name'])
    shutil.move(src_path, dst_path)

print("File processing pipeline completed")

