import torch.nn as nn
from simplecv.interface.configurable import ConfigurableMixin


class CVModule(nn.Module, ConfigurableMixin):
    def __init__(self, config):
        super(CVModule, self).__init__()
        ConfigurableMixin.__init__(self, config)

    def forward(self, *input):
        raise NotImplementedError

    def set_defalut_config(self):
        raise NotImplementedError
