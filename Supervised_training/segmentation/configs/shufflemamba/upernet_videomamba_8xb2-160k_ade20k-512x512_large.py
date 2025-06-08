_base_ = ['../_base_/datasets/ade20k.py', 
            '../_base_/default_runtime.py',
        '../_base_/schedules/schedule_160k.py']

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)

embed_dims = 1024
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    type='EncoderDecoder',
    backbone=dict(
        type='MM_VSSM',
        pretrained='your_model_path',
        img_size=(224, 224),
        patch_size=16,
        channels=3,
        embed_dim=embed_dims,
        depth=40,
        out_indices=[9, 19, 29, 39],
        drop_rate=0.,
        drop_path_rate=0.2,
        shuffle_rate=0.,
        residual_in_fp32=True,),
    neck=None,
    decode_head=dict(
        type='UPerHead',
        in_channels=[embed_dims, embed_dims, embed_dims, embed_dims],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=embed_dims,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))  # yapf: disable

# AdamW optimizer, no weight decay for position embedding & layer norm
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.01),
    # constructor='MyLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        # num_layers=32, layer_decay_rate=0.9,
        custom_keys={
            '.pos_embed': dict(decay_mult=0.),
            '.cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            '.A_log': dict(decay_mult=0.0),
            '.D': dict(decay_mult=0.0),
        }),
    clip_grad=dict(max_norm=3.0))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader