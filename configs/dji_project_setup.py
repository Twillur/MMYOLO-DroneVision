# dji_project_setup.py
import os

print("=== Setting Up DJI Drone Vision Project ===")

# Create project structure
directories = [
    'configs/dji',
    'data/images/train',
    'data/images/val', 
    'data/annotations',
    'data/videos',
    'tools',
    'work_dirs',
    'models'
]

for dir_path in directories:
    os.makedirs(dir_path, exist_ok=True)
    print(f"📁 Created: {dir_path}")

# Create README
readme = """# DJI Drone Vision Project

This project uses MMYOLO (OpenMMLab) for drone object detection, 
following DJI's computer vision pipeline.

## Project Structure
- `configs/dji/` - Configuration files
- `data/images/` - Drone images
- `data/annotations/` - COCO format annotations  
- `data/videos/` - Drone video footage
- `tools/` - Custom scripts
- `work_dirs/` - Training outputs
- `models/` - Trained models

## Getting Started
1. Add your drone images to `data/images/train/` and `data/images/val/`
2. Create COCO format annotations in `data/annotations/`
3. Update config file paths
4. Train with: `mim train mmyolo configs/dji/drone_detector.py`

## Drone Object Classes (Example)
1. person
2. vehicle  
3. building
4. road
5. tree
6. water
7. animal
8. drone
9. obstacle
10. field
"""

with open('README.md', 'w') as f:
    f.write(readme)
print("📄 Created: README.md")

# Create a sample annotation template
sample_annotation = '''{
  "info": {
    "description": "Drone Dataset",
    "version": "1.0",
    "year": 2024,
    "contributor": "Your Name"
  },
  "licenses": [],
  "images": [
    {
      "id": 1,
      "file_name": "drone_image_001.jpg",
      "height": 1080,
      "width": 1920
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 50, 50],
      "area": 2500,
      "segmentation": [],
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "object"
    }
  ]
}'''

with open('data/annotations/sample_template.json', 'w') as f:
    f.write(sample_annotation)
print("📄 Created: data/annotations/sample_template.json")

# Create dataset preparation script
dataset_script = '''# prepare_drone_dataset.py
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
'''

with open('tools/prepare_dataset.py', 'w') as f:
    f.write(dataset_script)
print("📄 Created: tools/prepare_dataset.py")

print("\n" + "=" * 60)
print("✅ DJI PROJECT SETUP COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("1. Add drone images to data/images/train/ and data/images/val/")
print("2. Run: python tools/prepare_dataset.py")
print("3. Annotate images (use LabelImg, CVAT, etc.)")
print("4. Update configs/dji/minimal_drone_detector.py")
print("5. When ready, train with MMYOLO")