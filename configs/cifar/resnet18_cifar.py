from mmengine.config import read_base

# Dataset config
dataset_type = 'CIFAR10Dataset'
data_root = 'data/cifar10/'

# Pipeline
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.4914, 0.4822,
         0.4465], std=[0.2023, 0.1994, 0.2010]),
]

test_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.4914, 0.4822,
         0.4465], std=[0.2023, 0.1994, 0.2010]),
]

# DataLoader
train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train'),
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test'),
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False))

# Models
teacher_cfg = dict(
    type='ResNet',
    depth=34,
    num_classes=10,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34'))

student_cfg = dict(
    type='ResNet',
    depth=18,
    num_classes=10,
    style='pytorch')

# Distillation config
model = dict(
    type='KnowledgeDistiller',
    teacher=teacher_cfg,
    student=student_cfg,
    distill_cfg=dict(
        temperature=4.0,
        alpha=0.5,
        kd_loss='kl_div'))

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005))

# Learning rate scheduler
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=200)
]

# Training config
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Checkpoint config
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook'))

# Runtime config
default_scope = 'mmcls'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
