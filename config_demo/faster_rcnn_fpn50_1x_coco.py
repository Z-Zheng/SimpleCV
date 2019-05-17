from rsdet.module.backbone import fpn

config = dict(
    model=dict(
        type='FasterRCNN',
        params=dict(
            backbone=dict(
                resnet=dict(
                    resnet_type='resnet50',
                    include_conv5=True,
                    batchnorm_trainable=False,
                    pretrained=True,
                    freeze_at=0,
                    # 8, 16 or 32
                    output_stride=32,
                    with_cp=(False, False, False, False),
                    stem3_3x3=False,
                ),
                fpn=dict(
                    in_channels_list=(256, 512, 1024, 2048),
                    out_channels=256,
                    conv_block=fpn.default_conv_block,
                    top_blocks=fpn.LastLevelMaxPool,
                ),
                size_divisible=32,
            ),
            rpn=dict(
                rpn_only=False,
                in_channels=256,
                anchor_sizes=(32, 64, 128, 256, 512),
                aspect_ratios=(0.5, 1.0, 2.0),
                # Stride of the feature map that RPN is attached.
                # For FPN, number of strides should match number of scales
                anchor_stride=(4, 8, 16, 32, 64),
                # Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
                # Set to -1 or a large value, e.g. 100000, to disable pruning anchors
                straddle_thresh=0,
                fpn_post_nms_top_n_train=2000,
                fpn_post_nms_top_n_test=2000,
                pre_nms_top_n_train=2000,
                pre_nms_top_n_test=1000,
                post_nms_top_n_train=2000,
                post_nms_top_n_test=1000,
                # Apply the post NMS per batch (default) or per image during training
                fpn_post_nms_per_batch=True,
                nms_thresh=0.7,
                min_size=0,
                use_fpn=True,
                fg_iou_threshold=0.7,
                bg_iou_threshold=0.3,
                batch_size_per_image=256,
                positive_fraction=0.5,
            ),
            roi_box_head=dict(
                num_classes=81,
                cls_agnostic_bbox_reg=False,
                feature_extractor=dict(
                    type='FPN2MLPFeatureExtractor',
                    in_channels=256,
                    pooler_resolution=7,
                    pooler_scales=(0.25, 0.125, 0.0625, 0.03125),
                    pooler_sampling_ratio=2,
                    # Hidden layer dimension when using an MLP for the RoI box head
                    mlp_head_dim=1024,
                ),
                loss=dict(
                    # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
                    fg_iou_threshold=0.5,
                    # Overlap threshold for an RoI to be considered background
                    # (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
                    bg_iou_threshold=0.5,
                    # Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
                    # These are empirically chosen to approximately lead to unit variance targets
                    bbox_reg_weights=(10., 10., 5., 5.),
                    # RoI minibatch size *per image* (number of regions of interest [ROIs])
                    # Total number of RoIs per training minibatch =
                    #   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
                    # E.g., a common configuration is: 512 * 2 * 8 = 8192
                    batch_size_per_image=512,
                    # Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
                    positive_fraction=0.25,
                ),
                post_processor=dict(
                    # Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
                    # balance obtaining high recall with not having too many low precision
                    # detections that will slow down inference post processing steps (like NMS)
                    score_thresh=0.05,
                    # Overlap threshold used for non-maximum suppression (suppress boxes with
                    # IoU >= this threshold),
                    nms_thresh=0.5,
                    # Maximum number of detections to return per image (100 is based on the limit
                    # established for the COCO dataset)
                    detections_per_img=100,
                )
            )
        )
    ),
    data=dict(
        train=dict(
            type='COCODatasetLoader',
            params=dict(
                images_per_batch=2,
                num_workers=0,
                training=True,
                distributed=True,
                image_dir='./coco2017/train2017',
                ann_file='./coco2017/annotations/instances_train2017.json',
                size_divisibility=32,
                transforms=dict(
                    color_jitter=dict(
                        brightness=0.,
                        contrast=0.,
                        saturation=0.,
                        hue=0.,
                    ),
                    resize_to_range=dict(
                        min_size=800,
                        max_size=1333,
                    ),
                    random_flip=dict(
                        prob=0.5
                    ),
                    normalize=dict(
                        mean=(123.675, 116.28, 103.53),
                        std=(58.395, 57.12, 57.375),
                        to_255=True,
                    )
                ),
            ),
        ),
        test=dict(
            type='COCODatasetLoader',
            params=dict(
                images_per_batch=2,
                num_workers=2,
                training=False,
                distributed=True,
                image_dir='./coco2017/val2017',
                ann_file='./coco2017/annotations/instances_val2017.json',
                size_divisibility=32,
                transforms=dict(
                    color_jitter=dict(
                        brightness=0.,
                        contrast=0.,
                        saturation=0.,
                        hue=0.,
                    ),
                    resize_to_range=dict(
                        min_size=800,
                        max_size=1333,
                    ),
                    random_flip=dict(
                        prob=0.
                    ),
                    normalize=dict(
                        mean=(123.675, 116.28, 103.53),
                        std=(58.395, 57.12, 57.375),
                        to_255=True,
                    )
                ),
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
        type='multistep',
        params=dict(
            base_lr=0.02,
            steps=(60000, 80000),
            gamma=0.1,
            warmup_step=500,
            warmup_init_lr=0.02 / 3, ),
    ),
    train=dict(
        forward_times=1,
        num_iters=90000,
        eval_per_epoch=False,
        summary_grads=False,
        summary_weights=False,
        distributed=True,
        apex_sync_bn=False,
        sync_bn=False,
        eval_after_train=False,
        log_interval_step=50,
    ),
    test=dict(
        box_only=False,
        iou_types=('bbox',),
        expected_results=[],
        expected_results_sigma_tol=4,
        output_folder='./log'
    ),
)
