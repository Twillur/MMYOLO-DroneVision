# MMYOLO Drone Waste Detection

## Quick Start:
1. Clone this repo
2. `pip install -r requirements.txt`
3. Download dataset from Roboflow
4. Run `python tools/train_yolo.py`

## Dataset:
Download from: https://universe.roboflow.com/trash-northeastern/trash-uavvaste

Place images in:
- `data/drone/images/train/` (617 images)
- `data/drone/images/val/` (116 images)
- `data/drone/images/test/` (39 images)

Annotations are already included in `data/drone/annotations/`

## Files Included:
- ✅ All Python scripts (`tools/`)
- ✅ All config files (`configs/`)
- ✅ All annotations (JSON files)
- ❌ Dataset images (download separately)
