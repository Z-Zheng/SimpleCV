import torch.nn as nn


class CVModule(nn.Module):
    def __init__(self, config):
        super(CVModule, self).__init__()
        self._cfg = config

    @property
    def config(self):
        return self._cfg

    def forward(self, *input):
        raise NotImplementedError


class Loss(CVModule):
    def __init__(self, config):
        super(Loss, self).__init__(config)

    def forward(self, *input):
        raise NotImplementedError
