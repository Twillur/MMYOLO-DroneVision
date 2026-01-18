# configs/trash_uavvaste_yolov8.py

# Since your yolov8_s_config.py is empty, we need a full config
# We'll create a minimal working config for YOLOv8 on your dataset

# Model settings
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=1024,
        deepen_factor=0.33,
        widen_factor=0.5),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=0.33,
        widen_factor=0.5,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024]),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=1,  # ← CRITICAL: 1 class for 'rubbish'
            in_channels=[256, 512, 1024],
            widen_factor=0.5),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='YOLOv5BBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=1.5)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=1,
            topk=10,
            alpha=0.5,
            beta=6.0)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

# Dataset settings
dataset_type = 'YOLOv5CocoDataset'
data_root = 'datasets/trash_uavvaste/'

# Class information from data.yaml
metainfo = dict(
    classes=('rubbish',),  # ← SINGLE CLASS: 'rubbish'
    palette=[(220, 20, 60)])

# Image size (your images are 416x416)
img_scale = (416, 416)

# Data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='mmdet.PackDetInputs')
]

# Data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/labels',  # ← YOLO format uses folder, not file
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/labels',
        data_prefix=dict(img='valid/images/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# Validation metrics
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'valid/labels',  # Will be auto-converted
    metric='bbox',
    proposal_nums=(100, 1, 10),
    classwise=True)
test_evaluator = val_evaluator

# Training schedule
max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5)  # Validate every 5 epochs

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        bypass_duplicate=True))

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Runtime settings
default_scope = 'mmyolo'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='auto',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None  # Start from scratch
resume = False