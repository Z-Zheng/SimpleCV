config = dict(
    model=dict(
        type='deeplabv3',
        params=dict(

        )
    ),
    data=dict(
        train=dict(
            type='detdataloader',
            params=dict(

            ),
        ),
        test=dict(
            type='detdataloader',
            params=dict(

            ),
        )
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0001
        )
    ),
    learning_rate=dict(
        type='multistep',
        params=dict(
            base_lr=0.01,
            steps=(60000, 80000),
            gamma=0.1,
            warmup_step=500,
            warmup_init_lr=0.01 / 3, ),
    ),
    train=dict(
        batch_size_per_device=2,
        num_gpu=0,
        forward_times=1,
        num_iters=90000,
    ),
    test=dict(
        batch_size_per_device=2,
        num_gpu=0
    ),
)
