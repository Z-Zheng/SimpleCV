import torch.nn as nn
from simplecv.interface import ConfigurableMixin
from simplecv.module._hrnet import hrnetv2_w18
from simplecv.module._hrnet import hrnetv2_w32
from simplecv.module._hrnet import hrnetv2_w40
from simplecv import registry
from torch.utils import checkpoint as cp
from simplecv.util import logger

_logger = logger.get_logger()
registry.MODEL.register('hrnetv2_w18', hrnetv2_w18)
registry.MODEL.register('hrnetv2_w32', hrnetv2_w32)
registry.MODEL.register('hrnetv2_w40', hrnetv2_w40)

defalut_config = dict(
    hrnet_type='hrnetv2_w18',
    pretrained=False,
    norm_eval=False,
    frozen_stages=-1,
    with_cp=False
)


@registry.MODEL.register('HRNetEncoder')
class HRNetEncoder(nn.Module, ConfigurableMixin):
    def __init__(self, config=defalut_config):
        super(HRNetEncoder, self).__init__()
        ConfigurableMixin.__init__(self, config)
        self.hrnet = registry.MODEL[self.config.hrnet_type](self.config.pretrained,
                                                            self.config.norm_eval,
                                                            self.config.frozen_stages)
        _logger.info('HRNetEncoder: pretrained = {}'.format(self.config.pretrained))

    def forward(self, x):
        if self.config.with_cp and not self.training:
            return cp.checkpoint(self.hrnet, x)
        return self.hrnet(x)

    def set_defalut_config(self):
        self.config.update(defalut_config)

    def output_channels(self):
        if self.config.hrnet_type == 'hrnetv2_w18':
            return 18, 36, 72, 144
        elif self.config.hrnet_type == 'hrnetv2_w32':
            return 32, 64, 128, 256
        elif self.config.hrnet_type == 'hrnetv2_w40':
            return 40, 80, 160, 320
        else:
            raise NotImplementedError('{} is not implemented.'.format(self.config.hrnet_type))
