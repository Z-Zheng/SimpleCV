import torch.nn as nn
from simplecv.core.config import AttrDict


class CVModule(nn.Module):
    def __init__(self, config):
        super(CVModule, self).__init__()
        self._cfg = AttrDict(

        )
        self.set_defalut_config()
        self._cfg.update(config)

    def forward(self, *input):
        raise NotImplementedError

    def set_defalut_config(self):
        raise NotImplementedError

    @property
    def config(self):
        return self._cfg
