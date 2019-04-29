import torch.nn as nn
from simplecv.core import AttrDict


class CVModule(nn.Module):
    def __init__(self, config):
        super(CVModule, self).__init__()
        self._cfg = AttrDict(

        )
        self.set_defalut_config()
        self._update_config(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

    def forward(self, *input):
        raise NotImplementedError

    def set_defalut_config(self):
        raise NotImplementedError

    def _update_config(self, new_config):
        self._cfg.update(new_config)

    @property
    def config(self):
        return self._cfg

    # def __repr__(self):
    #     s = '[config]\n'
    #     for k, v in self.config.items():
    #         s += '{name} = {value}\n'.format(name=k, value=v)
    #     s += '[------]'
    #     return s


class Loss(CVModule):
    def __init__(self, config):
        super(Loss, self).__init__(config)

    def forward(self, *input):
        raise NotImplementedError


class LearningRateBase(object):
    def __init__(self, base_lr):
        self._base_lr = base_lr

    @property
    def base_lr(self):
        return self._base_lr

    def step(self, global_step, optimizer):
        raise NotImplementedError
