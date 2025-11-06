# Import required libraries
import os
import cv2
import numpy as np
from tqdm import tqdm

# Define paths
input_path = "./EuroSAT"
output_path = "./Processed_EuroSAT_64x64"

# Create output directory
os.makedirs(output_path)

# Get all class folders
class_folders = os.listdir(input_path)

# Process each class
for class_folder in class_folders:
    print(f"Processing {class_folder}")
    
    # Create class folder in output directory
    os.makedirs(os.path.join(output_path, class_folder))
    
    # Get all images in class folder
    images = os.listdir(os.path.join(input_path, class_folder))
    
    # Process each image
    for img_name in tqdm(images):
        # Read image
        img = cv2.imread(os.path.join(input_path, class_folder, img_name))
        
        # Normalize to [0,1]
        img_normalized = img.astype(np.float32) / 255.0
        
        # Convert back to uint8 for saving (0-255)
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        
        # Save as regular image
        cv2.imwrite(
            os.path.join(output_path, class_folder, img_name),
            img_uint8
        )

print("DONE")