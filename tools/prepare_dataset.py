# prepare_drone_dataset.py
import os
import json
from pathlib import Path

def prepare_drone_dataset(image_dir, output_dir):
    """
    Prepare drone dataset in COCO format.
    
    Args:
        image_dir: Directory with drone images
        output_dir: Output directory for annotations
    """
    print("Preparing drone dataset...")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images")
    
    # Create COCO structure
    coco_data = {
        "info": {
            "description": "Drone Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "Your Name"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add your categories here
    categories = [
        {"id": 1, "name": "person", "supercategory": "object"},
        {"id": 2, "name": "vehicle", "supercategory": "object"},
        {"id": 3, "name": "building", "supercategory": "object"},
        {"id": 4, "name": "road", "supercategory": "object"},
        {"id": 5, "name": "tree", "supercategory": "object"},
        {"id": 6, "name": "water", "supercategory": "object"},
        {"id": 7, "name": "animal", "supercategory": "object"},
        {"id": 8, "name": "drone", "supercategory": "object"},
        {"id": 9, "name": "obstacle", "supercategory": "object"},
        {"id": 10, "name": "field", "supercategory": "object"},
    ]
    
    coco_data["categories"] = categories
    
    # Save
    output_file = os.path.join(output_dir, "annotations.json")
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Template saved: {output_file}")
    print("Add your image metadata and annotations to this file")

if __name__ == "__main__":
    prepare_drone_dataset("data/images/train", "data/annotations")
