# configs/dji/trash_uavvaste_coco_config.py
from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *

from mmengine.dataset.sampler import DefaultSampler
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.transforms import RandomFlip, Resize
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmdet.models.detectors.yolo import YOLODetector
from mmdet.models.task_modules.assigners.task_aligned_assigner import TaskAlignedAssigner
from mmdet.models.losses.gfocal_loss import QualityFocalLoss
from mmdet.models.losses.iou_loss import GIoULoss
from mmyolo.models.dense_heads.yolov5_head import YOLOv5Head
from mmyolo.models.layers.yolov5_pafpn import YOLOv5PAFPN
from mmyolo.models.backbones.yolov5_csp import YOLOv5CSPDarknet

# Dataset configuration
dataset_type = 'CocoDataset'
data_root = 'data/drone/'

# Class names (from your dataset)
class_name = ('rubbish',)  # Single class - waste detection
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

# Image size
img_scale = (640, 640)  # Training image size

# Train pipeline
train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),
    dict(type=Resize, scale=img_scale, keep_ratio=False),
    dict(type=RandomFlip, prob=0.5),
]

# Test/Val pipeline
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),
    dict(type=Resize, scale=img_scale, keep_ratio=False),
]

# Dataset configuration
train_dataloader = dict(
    batch_size=8,  # Reduced for CPU training
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# Evaluation metrics
val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    format_only=False,
    classwise=True)

test_evaluator = val_evaluator

# Model configuration
model = dict(
    type=YOLODetector,
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type=YOLOv5CSPDarknet,
        arch='P5',
        out_indices=(3, 4, 5),
        deepen_factor=0.33,
        widen_factor=0.5),
    neck=dict(
        type=YOLOv5PAFPN,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        deepen_factor=0.33,
        widen_factor=0.5),
    bbox_head=dict(
        type=YOLOv5Head,
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=0.5),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[[(10, 13), (16, 30), (33, 23)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(116, 90), (156, 198), (373, 326)]],
            strides=[8, 16, 32]),
        loss_cls=dict(
            type=QualityFocalLoss,
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type=GIoULoss, loss_weight=0.05),
        loss_obj=dict(
            type=QualityFocalLoss,
            use_sigmoid=True,
            beta=1.0,
            loss_weight=1.0),
        assigner=dict(
            type=TaskAlignedAssigner,
            topk=13,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'))),
    train_cfg=dict(
        assigner=dict(
            type=TaskAlignedAssigner,
            topk=13,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'))),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

# Training schedule
max_epochs = 100
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=5)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2))

# Learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[60, 80],
        gamma=0.1)
]

# Runtime
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='auto'))
