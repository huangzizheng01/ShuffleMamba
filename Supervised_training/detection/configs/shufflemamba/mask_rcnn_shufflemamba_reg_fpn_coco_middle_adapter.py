_base_ = [
    './template.py',
    '../_base_/models/mask-rcnn_r50_fpn.py',
]

embed_dims = 576
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='MM_VSSM_ADT',
        pretrained='your_model_path',
        img_size=(224, 224),
        patch_size=16,
        channels=3,
        embed_dim=embed_dims,
        depth=32,
        out_indices=[7, 15, 23, 31],
        drop_rate=0.,
        drop_path_rate=0.4,
        shuffle_rate=0.,
        residual_in_fp32=True,
        fused_add_norm=False,
        rms_norm=False,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 7], [8, 15], [16, 23], [24, 31]],),
    neck=dict(
        type='FPN',
        in_channels=[embed_dims, embed_dims, embed_dims, embed_dims],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    )


# AdamW optimizer, no weight decay for position embedding & layer norm
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1.e-4, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='MyLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=32, layer_decay_rate=0.9,
        custom_keys={
            '.pos_embed': dict(decay_mult=0.),
            '.cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            '.A_log': dict(decay_mult=0.0),
            '.D': dict(decay_mult=0.0),
        }))