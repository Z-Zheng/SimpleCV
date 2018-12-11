config = dict(
    model=dict(),
    data=dict(),
    optimizer=dict(
        type='',
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
    test=dict(),
)
