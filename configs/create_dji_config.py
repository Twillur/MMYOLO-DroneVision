# create_dji_config.py
print("=== Creating DJI Drone Vision Config ===")

from mmengine.config import Config

# DJI-style YOLOv8 config for drone detection
config_text = '''
# DJI DroneVision Config - YOLOv8 Nano
# Based on MMYOLO YOLOv8 configs

# Model Configuration
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True,
        pad_mask=False,
        pad_size_divisor=32),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=1024,
        deepen_factor=0.33,    # nano: 0.33, small: 0.67, medium: 1.0
        widen_factor=0.5,      # nano: 0.25, small: 0.5, medium: 0.75
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN',
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        deepen_factor=0.33,
        widen_factor=0.5,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        use_cspnext_block=False,
        expand_ratio=0.5,
        num_csp_blocks=1,
        upsample_feats_cat_first=False),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=10,  # Drone-specific classes
            in_channels=[256, 512, 1024],
            widen_factor=0.5,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32]),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator',
            offset=0.5,
            strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False),
        # Since we don't have Objectness loss in YOLOv8, assign loss_weight=0
        # to loss_obj to avoid raising error
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=1.0),
        obj_level_weights=[4.0, 1.0, 0.4]),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=10,
            use_ciou=True,
            topk=10,
            alpha=0.5,
            beta=6.0,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

# Drone Dataset Configuration
dataset_type = 'CocoDataset'
data_root = 'data/drone/'

# Image augmentation for aerial views
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
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
        pipeline=train_pipeline,
        backend_args=None))

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
        pipeline=val_pipeline,
        backend_args=None))

test_dataloader = val_dataloader

# Validation metrics
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None)

test_evaluator = val_evaluator

# Training Configuration
max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=None)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True),
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning Rate Schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        milestones=[60, 80],
        by_epoch=True,
        begin=0,
        end=max_epochs,
        gamma=0.1)
]

# Runtime
default_scope = 'mmyolo'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
'''

# Parse and save config
cfg = Config.fromstring(config_text, '.py')
print(f"✅ Config parsed: {cfg.model.type}")
print(f"   Backbone: {cfg.model.backbone.type}")
print(f"   Classes: {cfg.model.bbox_head.head_module.num_classes}")

# Save to file
with open('configs/dji/drone_detector_yolov8n.py', 'w') as f:
    f.write(config_text)

print("📁 Config saved: configs/dji/drone_detector_yolov8n.py")

print("\n" + "=" * 60)
print("DJI DRONE VISION CONFIG READY!")
print("=" * 60)

print("\nNext steps:")
print("1. Add your drone images to data/images/train/ and data/images/val/")
print("2. Create COCO format annotations in data/annotations/")
print("3. Update config paths to match your data")
print("4. Change num_classes to match your drone object categories")
print("5. When ready for training, install mmcv-full with C++ ops")
print("6. Train with: mim train mmyolo configs/dji/drone_detector_yolov8n.py")

print("\nDrone object categories (example):")
print("  1. person")
print("  2. vehicle")
print("  3. building") 
print("  4. road")
print("  5. tree")
print("  6. water")
print("  7. animal")
print("  8. drone")
print("  9. obstacle")
print("  10. field")