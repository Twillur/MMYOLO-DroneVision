# setup_trash_yolo.py
import os
import yaml
import shutil
from pathlib import Path
import json

print("=== Setting Up TRASH-UAVVaste YOLOv8 Dataset ===")

# Path to dataset
dataset_path = Path("datasets/trash_uavvaste")

if not dataset_path.exists():
    print(f"❌ Dataset not found: {dataset_path}")
    exit(1)

print(f"✅ Found dataset at: {dataset_path}")

# Read the data.yaml file
yaml_file = dataset_path / "data.yaml"
if not yaml_file.exists():
    print(f"❌ data.yaml not found: {yaml_file}")
    exit(1)

with open(yaml_file, 'r') as f:
    data_config = yaml.safe_load(f)

print(f"\nDataset configuration:")
print(f"  Path: {data_config.get('path', 'Not specified')}")
print(f"  Train: {data_config.get('train', 'Not specified')}")
print(f"  Val: {data_config.get('val', 'Not specified')}")
print(f"  Test: {data_config.get('test', 'Not specified')}")
print(f"  Classes: {len(data_config.get('names', []))}")

# Show classes
names = data_config.get('names', {})
if names:
    print("\nWaste categories:")
    for class_id, class_name in names.items():
        print(f"  {class_id}: {class_name}")

# Check folder structure
print("\nChecking folder structure...")

# Check train folder
train_path = dataset_path / "train"
if train_path.exists():
    images_path = train_path / "images"
    labels_path = train_path / "labels"
    
    if images_path.exists():
        images = list(images_path.glob("*"))
        print(f"  Train images: {len(images)}")
    
    if labels_path.exists():
        labels = list(labels_path.glob("*"))
        print(f"  Train labels: {len(labels)}")

# Check valid folder
val_path = dataset_path / "valid"
if val_path.exists():
    images_path = val_path / "images"
    labels_path = val_path / "labels"
    
    if images_path.exists():
        images = list(images_path.glob("*"))
        print(f"  Val images: {len(images)}")
    
    if labels_path.exists():
        labels = list(labels_path.glob("*"))
        print(f"  Val labels: {len(labels)}")

# Convert to COCO format for MMYOLO
print("\n" + "=" * 60)
print("Converting to COCO format for MMYOLO...")

def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """Convert YOLO bbox [x_center, y_center, width, height] to COCO [x, y, width, height]"""
    x_center, y_center, bbox_width, bbox_height = yolo_bbox
    
    # Convert from normalized to absolute coordinates
    x = (x_center - bbox_width / 2) * img_width
    y = (y_center - bbox_height / 2) * img_height
    width = bbox_width * img_width
    height = bbox_height * img_height
    
    return [float(x), float(y), float(width), float(height)]

def convert_split_to_coco(split_path, output_json, split_name):
    """Convert YOLO format split to COCO format"""
    from PIL import Image
    
    images_path = split_path / "images"
    labels_path = split_path / "labels"
    
    if not images_path.exists() or not labels_path.exists():
        print(f"❌ Missing images or labels for {split_name}")
        return False
    
    coco_data = {
        "info": {
            "description": f"TRASH-UAVVaste {split_name} split",
            "version": "1.0",
            "year": 2024,
            "contributor": "Roboflow"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for class_id, class_name in names.items():
        coco_data["categories"].append({
            "id": int(class_id),
            "name": class_name,
            "supercategory": "waste"
        })
    
    # Process images
    image_files = list(images_path.glob("*"))
    annotation_id = 1
    
    for img_idx, img_file in enumerate(image_files[:50]):  # Process first 50 for testing
        # Get image info
        try:
            with Image.open(img_file) as img:
                width, height = img.size
        except:
            print(f"  Warning: Could not open {img_file}")
            continue
        
        # Add to images list
        image_id = img_idx + 1
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_file.name,
            "width": width,
            "height": height
        })
        
        # Corresponding label file
        label_file = labels_path / f"{img_file.stem}.txt"
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:  # class_id, x_center, y_center, width, height
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    
                    # Convert YOLO bbox to COCO
                    coco_bbox = yolo_to_coco_bbox(bbox, width, height)
                    
                    # Add annotation
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id + 1,  # COCO uses 1-indexed
                        "bbox": coco_bbox,
                        "area": coco_bbox[2] * coco_bbox[3],
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    
                    annotation_id += 1
    
    # Save COCO JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Converted {split_name}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    return True

# Create data directory for MMYOLO
data_dir = Path("data/drone")
data_dir.mkdir(parents=True, exist_ok=True)

# Create images directory
images_dir = data_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

train_images_dir = images_dir / "train"
val_images_dir = images_dir / "val"
train_images_dir.mkdir(exist_ok=True)
val_images_dir.mkdir(exist_ok=True)

# Create annotations directory
ann_dir = data_dir / "annotations"
ann_dir.mkdir(parents=True, exist_ok=True)

# Convert train split
train_success = convert_split_to_coco(
    train_path,
    ann_dir / "train.json",
    "train"
)

# Convert val split  
val_success = convert_split_to_coco(
    val_path,
    ann_dir / "val.json", 
    "val"
)

# Copy some sample images
print("\nCopying sample images...")
def copy_sample_images(src_dir, dest_dir, max_images=10):
    src_images = list((src_dir / "images").glob("*"))
    for i, img_file in enumerate(src_images[:max_images]):
        dest_file = dest_dir / img_file.name
        shutil.copy2(img_file, dest_file)

copy_sample_images(train_path, train_images_dir, 5)
copy_sample_images(val_path, val_images_dir, 5)

print(f"✅ Copied sample images to {images_dir}")

# Create MMYOLO config for this dataset
print("\n" + "=" * 60)
print("Creating MMYOLO config...")

config_content = f'''# TRASH-UAVVaste Drone Waste Detection - YOLOv8
# Converted from YOLO format to COCO for MMYOLO

_base_ = [
    '../_base_/default_runtime.py',
]

# Model
num_classes = {len(names)}
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,  # YOLOv8n
        widen_factor=0.5),
    neck=dict(type='YOLOv8PAFPN'),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        assigner=dict(type='BatchTaskAlignedAssigner')),
    test_cfg=dict(
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7)))

# Dataset
dataset_type = 'CocoDataset'
data_root = 'data/drone/'

# Image augmentation
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# Data Loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = val_dataloader

# Validation metrics
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False)

test_evaluator = val_evaluator

# Training
train_cfg = dict(max_epochs=100, val_interval=5)

# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9))

# Classes in this dataset:
'''

# Add class information
for class_id, class_name in names.items():
    config_content += f"#   {class_id}: {class_name}\n"

# Save config
config_dir = Path("configs/dji")
config_dir.mkdir(parents=True, exist_ok=True)

config_file = config_dir / "trash_uavvaste_yolo_converted.py"
with open(config_file, 'w') as f:
    f.write(config_content)

print(f"✅ Config saved: {config_file}")

print("\n" + "=" * 60)
print("SETUP COMPLETE!")
print("=" * 60)
print(f"\nDataset converted to COCO format in: data/drone/")
print(f"  - Images: data/drone/images/")
print(f"  - Annotations: data/drone/annotations/")
print(f"  - Config: configs/dji/trash_uavvaste_yolo_converted.py")
print("\nTo use all images (not just samples):")
print("1. Copy all images from datasets/trash_uavvaste/train/images/ to data/drone/images/train/")
print("2. Copy all images from datasets/trash_uavvaste/valid/images/ to data/drone/images/val/")
print("3. Run conversion on full dataset")
print("\nWhen mmcv-full is available, train with:")
print("  mim train mmyolo configs/dji/trash_uavvaste_yolo_converted.py")