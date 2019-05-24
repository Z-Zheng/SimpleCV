import torch.nn as nn

dict(
    resnet_type='resnet50',
    include_conv5=True,
    batchnorm_trainable=True,
    pretrained=False,
    freeze_at=0,
    # 16 or 32
    output_stride=32,
    with_cp=(False, False, False, False),
    norm_layer=lambda num_channels: nn.GroupNorm(num_groups=16, num_channels=num_channels),
)
