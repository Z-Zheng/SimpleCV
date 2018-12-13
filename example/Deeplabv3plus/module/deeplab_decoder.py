import torch
import torch.nn as nn
import torch.nn.functional as F

from simplecv.interface import CVModule
from simplecv import registry
from simplecv.module import SeparableConv2D

@registry.MODEL.register('deeplab_decoder')
class DeeplabDecoder(CVModule):
    """
    This module is a reimplemented version in the following paper.
    Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous
    separable convolution for semantic image segmentation[J],
    """

    def __init__(self, config):
        super(DeeplabDecoder, self).__init__(config)
        for k, v in self.config.items():
            self.__dict__[k] = v
        norm_fn = registry.OP[self.norm_fn]

        layers = [nn.Conv2d(self.low_level_feature_channel, self.reduction_dim, 1)]
        if self.use_batchnorm:
            layers.append(norm_fn(self.reduction_dim))
        layers.append(nn.ReLU(inplace=True))
        self.conv1x1_low_level = nn.Sequential(*layers)

        layers = [nn.Conv2d(self.encoder_feature_channel, self.reduction_dim, 1)]
        if self.use_batchnorm:
            layers.append(norm_fn(self.reduction_dim))
        layers.append(nn.ReLU(inplace=True))
        self.conv1x1_encoder = nn.Sequential(*layers)

        layers = [SeparableConv2D(self.reduction_dim * 2, self.decoder_dim, 3, 1, padding=1, dilation=1,
                                  bias=self.use_bias,
                                  norm_fn=norm_fn)] + [
                     SeparableConv2D(self.decoder_dim, self.decoder_dim, 3, 1, padding=1, dilation=1,
                                     bias=self.use_bias,
                                     norm_fn=norm_fn) for i in
                     range(self.num_3x3conv - 1)]
        self.stack_conv3x3 = nn.Sequential(*layers)

    def forward(self, low_level_feature, encoder_feature):
        low_feat = self.conv1x1_low_level(low_level_feature)
        encoder_feat = self.conv1x1_encoder(encoder_feature)

        feat_upx = F.interpolate(encoder_feat, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)

        concat_feat = torch.cat([low_feat, feat_upx], dim=1)

        out = self.stack_conv3x3(concat_feat)

        return out

    def set_defalut_config(self):
        self.config.update(
            low_level_feature_channel=256,
            encoder_feature_channel=256,
            reduction_dim=48,
            decoder_dim=256,
            num_3x3conv=2,
            scale_factor=4.0,
            use_bias=True,
            use_batchnorm=False,
            norm_fn='batchnorm'
        )
