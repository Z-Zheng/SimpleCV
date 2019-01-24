from simplecv.util.logger import get_logger
from functools import reduce

logger = get_logger(__name__)


def trainable_parameters(module):
    ret = []
    total = 0
    for idx, p in enumerate(module.parameters()):
        if p.requires_grad:
            ret.append(p)
        total = idx + 1
    logger.info('[trainable params] {}/{}'.format(len(ret), total))
    return ret


def count_model_parameters(module):
    cnt = 0
    for p in module.parameters():
        cnt += reduce(lambda x, y: x * y, list(p.shape))
    logger.info('#params: {}, {} M'.format(cnt, round(cnt / float(1e6), 3)))

    return cnt


def freeze_params(module):
    for name, p in module.named_parameters():
        p.requires_grad = False
        # todo: show complete name
        # logger.info('[freeze params] {name}'.format(name=name))


def freeze_modules(module, specific_class=None):
    for m in module.modules():
        if specific_class is not None:
            if not isinstance(m, specific_class):
                continue
        freeze_params(m)

