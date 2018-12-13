from simplecv.interface import CVModule
from simplecv import registry


@registry.MODEL.register('deeplab_encoder')
class DeeplabEncoder(CVModule):
    def __init__(self, config):
        super(DeeplabEncoder, self).__init__(config)
        self.resnet_encoder = registry.MODEL['resnet_encoder'](self.config['resnet_encoder'])
        self.aspp = registry.OP['aspp'](**self.config['aspp'])

    def forward(self, x):
        feat_list = self.resnet_encoder(x)

        aspp_feat = self.aspp(feat_list[-1])

        return feat_list[0], aspp_feat

    def set_defalut_config(self):
        self.config.update(dict(
            resnet_encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnoram_trainable=False,
                pretrained=False,
                freeze_at=0,
            ),
            aspp=dict(
                in_channel=2048,
                aspp_dim=256,
                atrous_rates=(6, 12, 18),
                add_image_level=True,
                use_bias=True,
                use_batchnorm=False,
                norm_type='batchnorm'
            ),
        ))


if __name__ == '__main__':
    model = DeeplabEncoder({})
    import torch

    im = torch.ones([1, 3, 256, 256])

    a, b = model(im)
    print(a.shape)
    print(b.shape)
