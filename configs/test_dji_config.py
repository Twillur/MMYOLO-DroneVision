# test_dji_config.py
print("=== Testing DJI Config Creation ===")

# Only import what definitely works
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    
    # Try to import Config without mmcv ops
    from mmengine.config import Config
    
    # Create a minimal config
    config_dict = {
        'model': {
            'type': 'YOLODetector',
            'num_classes': 10,
        },
        'data': {
            'train': {'type': 'CocoDataset', 'ann_file': 'train.json'},
            'val': {'type': 'CocoDataset', 'ann_file': 'val.json'},
        },
        'optimizer': {'type': 'SGD', 'lr': 0.01},
        'train_cfg': {'max_epochs': 100},
    }
    
    cfg = Config(config_dict)
    print(f"✅ Config created: {cfg.model.type}")
    print(f"   Classes: {cfg.model.num_classes}")
    print(f"   Epochs: {cfg.train_cfg.max_epochs}")
    
    # Save it
    import os
    os.makedirs('configs/dji', exist_ok=True)
    
    config_text = f'''
# DJI Drone Detector Config
model = dict(
    type='YOLODetector',
    num_classes={cfg.model.num_classes},
)

data = dict(
    train=dict(type='CocoDataset', ann_file='train.json'),
    val=dict(type='CocoDataset', ann_file='val.json'),
)

optimizer = dict(type='SGD', lr=0.01)
train_cfg = dict(max_epochs={cfg.train_cfg.max_epochs})
'''
    
    with open('configs/dji/drone_detector.py', 'w') as f:
        f.write(config_text)
    
    print(f"📁 Config saved: configs/dji/drone_detector.py")
    
    print("\n" + "=" * 60)
    print("✅ DJI CONFIG DEVELOPMENT READY!")
    print("=" * 60)
    print("\nYou can now:")
    print("1. Edit configs/dji/drone_detector.py")
    print("2. Add your drone dataset paths")
    print("3. Customize model architecture")
    print("4. Design training schedule")
    
    print("\nFor actual training, you'll need mmcv-full")
    print("But config development works perfectly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    
    # Try even simpler
    print("\nTrying minimal approach...")
    try:
        # Just test PyTorch
        import torch
        print(f"✅ At least PyTorch works: {torch.__version__}")
        print("\nYou can still:")
        print("1. Prepare drone dataset in COCO format")
        print("2. Write config files manually")
        print("3. Use MMYOLO tools later")
    except:
        print("❌ Basic imports failed")