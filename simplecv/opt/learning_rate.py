import numpy as np
from simplecv.util import registry


def make_learningrate(config):
    lr_type = config['type']
    if lr_type in registry.LR:
        lr_module = registry.LR[lr_type]
        return lr_module(**config['params'])
    else:
        raise ValueError('{} is not support now.'.format(lr_type))


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class LearningRateBase(object):
    def __init__(self):
        pass

    def step(self, global_step, optimizer):
        raise NotImplementedError


@registry.LR.register('multistep')
class MultiStepLearningRate(LearningRateBase):
    def __init__(self,
                 steps,
                 base_lr=0.1,
                 gamma=0.1,
                 warmup_step=None,
                 warmup_init_lr=None):
        super(MultiStepLearningRate, self).__init__()
        self._steps = np.array(steps)
        self._base_lr = base_lr
        self._gamma = gamma
        self._warmup_step = warmup_step
        self._warmup_init_lr = warmup_init_lr

        self._check()

    def _check(self):
        if self._warmup_step is not None:
            assert self._warmup_init_lr < self._base_lr
            assert self._warmup_step < self._steps[0]

        assert np.all(np.diff(self._steps) > 0)

    def step(self, global_step, optimizer):
        cur_step = global_step

        if self._warmup_step is not None:
            if cur_step <= self._warmup_step:
                lr = self._compute_warmup_lr(cur_step)
                set_lr(optimizer, lr)
                return

        lr = self._compute_lr(cur_step)

        set_lr(optimizer, lr)

    def _compute_lr(self, cur_step):
        return self._base_lr * (self._gamma ** int((cur_step > self._steps).sum(dtype=np.int32)))

    def _compute_warmup_lr(self, cur_step):
        lr = cur_step * (self._base_lr - self._warmup_init_lr) / self._warmup_step + self._warmup_init_lr
        return lr
