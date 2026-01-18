# validate_dataset.py
import os
from pathlib import Path
import cv2

def validate_dataset(dataset_path='datasets/trash_uavvaste'):
    """Validate the TRASH+UAVVaste dataset structure and integrity"""
    dataset_path = Path(dataset_path)
    
    print("="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    
    for split in ['train', 'valid', 'test']:
        img_dir = dataset_path / split / 'images'
        lbl_dir = dataset_path / split / 'labels'
        
        print(f"\n🔍 Checking '{split}' split:")
        
        # Check if directories exist
        if not img_dir.exists():
            print(f"   ❌ ERROR: Missing images directory: {img_dir}")
            continue
        if not lbl_dir.exists():
            print(f"   ❌ ERROR: Missing labels directory: {lbl_dir}")
            continue
        
        # Count files
        img_files = list(img_dir.glob('*.*'))
        lbl_files = list(lbl_dir.glob('*.txt'))
        
        img_exts = {f.suffix.lower() for f in img_files}
        print(f"   Found {len(img_files)} images with extensions: {img_exts}")
        print(f"   Found {len(lbl_files)} label files")
        
        # Get filenames without extensions
        img_stems = {f.stem for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']}
        lbl_stems = {f.stem for f in lbl_files}
        
        # Check for matches
        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems
        
        if missing_labels:
            print(f"   ⚠️  WARNING: {len(missing_labels)} images have no labels")
            if len(missing_labels) <= 3:
                for stem in list(missing_labels)[:3]:
                    print(f"      - {stem}")
        
        if missing_images:
            print(f"   ⚠️  WARNING: {len(missing_images)} labels have no images")
            if len(missing_images) <= 3:
                for stem in list(missing_images)[:3]:
                    print(f"      - {stem}")
        
        # Perfect match check
        if not missing_labels and not missing_images and img_stems:
            print(f"   ✅ PERFECT: All {len(img_stems)} images have corresponding labels!")
        
        # Sample check of first image
        if img_stems:
            sample_stem = next(iter(img_stems))
            sample_img = next(img_dir.glob(f"{sample_stem}.*"))
            sample_lbl = lbl_dir / f"{sample_stem}.txt"
            
            img = cv2.imread(str(sample_img))
            if img is None:
                print(f"   ❌ ERROR: Could not read sample image: {sample_img.name}")
            else:
                print(f"   Sample image: {sample_img.name} - Shape: {img.shape}")
                
            if sample_lbl.exists():
                with open(sample_lbl, 'r') as f:
                    lines = f.readlines()
                print(f"   Sample label: {len(lines)} objects found")
                
                # Show class distribution in sample
                if lines:
                    classes = [int(line.split()[0]) for line in lines if line.strip()]
                    from collections import Counter
                    class_counts = Counter(classes)
                    print(f"   Class IDs in sample: {dict(class_counts)}")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    validate_dataset()