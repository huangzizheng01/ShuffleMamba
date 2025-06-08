_base_ = [
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

# AdamW optimizer, no weight decay for position embedding & layer norm
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    # constructor='MyLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        # num_layers=32, layer_decay_rate=0.9,
        custom_keys={
            '.pos_embed': dict(decay_mult=0.),
            '.cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            '.A_log': dict(decay_mult=0.0),
            '.D': dict(decay_mult=0.0),
        }))

max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,)

