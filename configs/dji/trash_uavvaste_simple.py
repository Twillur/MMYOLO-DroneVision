# configs/dji/trash_uavvaste_simple.py
from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *

from mmdet.datasets.coco import CocoDataset
from mmdet.evaluation.metrics import CocoMetric

# Basic dataset config
dataset_type = 'CocoDataset'
data_root = 'data/drone/'

# Single class - waste
class_name = ('rubbish',)
num_classes = len(class_name)
metainfo = dict(classes=class_name)

# Simple train dataloader (for testing)
train_dataloader = dict(
    batch_size=4,  # Small batch for CPU
    num_workers=0,  # 0 for Windows CPU
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False)))

# Simple validation
val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=metainfo,
        test_mode=True))

test_dataloader = val_dataloader

# Evaluation
val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'valid/_annotations.coco.json',
    metric='bbox',
    format_only=False)

test_evaluator = val_evaluator

# Simple model (YOLOv5 nano for CPU testing)
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.]),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        arch='P5',
        deepen_factor=0.33,
        widen_factor=0.25),  # Nano size
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=0.33,
        widen_factor=0.25),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=num_classes,
            widen_factor=0.25),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=0.05,
            return_iou=False),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0)))

# Training config (short for testing)
max_epochs = 10
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=2)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9))

# Learning rate
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(type='MultiStepLR', milestones=[6, 8], gamma=0.1)
]

# Checkpoint
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))
