import torch.optim
from simplecv.util import registry

registry.OPT.register('sgd', torch.optim.SGD)
registry.OPT.register('adam', torch.optim.Adam)


def make_optimizer(config, params):
    opt_type = config['type']
    if opt_type in registry.OPT:
        opt = registry.OPT[opt_type](params=params, **config['params'])
    else:
        raise ValueError('{} is not support now.'.format(opt_type))
    return opt
