config = dict(
    model=dict(
        type='Deeplabv3',
        params=dict(
            encoder_config=dict(
                resnet_encoder=dict(
                    resnet_type='resnet101',
                    include_conv5=True,
                    batchnorm_trainable=True,
                    pretrained=True,
                    freeze_at=0,
                    # 8, 16 or 32
                    output_stride=16,
                    with_cp=(True, True, True, True),
                    stem3_3x3=False,
                ),
            ),
            decoder_config=dict(
                in_channel=2048,
                decoder_dim=256,
                # feat os 16, (4, 8)
                # feat os 8,  (2, 4)
                multi_grid_dilations=(4, 8),
            ),
            aspp_config=dict(
                in_channel=256,
                aspp_dim=256,
                atrous_rates=(6, 12, 18),
                add_image_level=True,
                use_bias=False,
                use_batchnorm=True,
                norm_type='batchnorm'
            ),
            upsample_ratio=16.0,
            loss_config=dict(
                cls_weight=1.0,
            ),
            num_classes=21,
        )
    ),
    data=dict(
        train=dict(
            type='DistVOCDataLoader',
            params=dict(
                crop_size=(512, 512),
                training=True,
                scale_range=[0.5, 2],
                batch_size=16,
                num_workers=2,
                img_dir='./voc2012/aug_image',
                mask_dir='./voc2012/aug_mask',
                train_txt_path='./voc2012/aug_train.txt',
                val_txt_path='./voc2012/aug_val.txt',
            ),
        ),
        test=dict(
            type='VOCDataLoader',
            params=dict(
                training=False,
                batch_size=12,
                num_workers=2,
                img_dir='./voc2012/aug_image',
                mask_dir='./voc2012/aug_mask',
                train_txt_path='./voc2012/aug_train.txt',
                val_txt_path='./voc2012/aug_val.txt',
            ),
        ),
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.0001
        ),
        grad_clip=dict(
            max_norm=35,
            norm_type=2,
        )
    ),
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=0.007,
            power=0.9,
            max_iters=60000,
        )),
    train=dict(
        forward_times=1,
        num_iters=60000,
        eval_per_epoch=False,
        summary_grads=False,
        summary_weights=False,
        distributed=True,
        apex_sync_bn=True,
    ),
    test=dict(
    ),
)
