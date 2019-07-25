import torch.optim
from simplecv.util import registry

registry.OPT.register('sgd', torch.optim.SGD)
registry.OPT.register('adam', torch.optim.Adam)
try:
    from apex.optimizers import FusedAdam
    registry.OPT.register('fused_adam', FusedAdam)
except ImportError:
    print('Please install apex for FusedAdam.')

def make_optimizer(config, params):
    opt_type = config['type']
    if opt_type in registry.OPT:
        opt = registry.OPT[opt_type](params=params, **config['params'])
        opt.simplecv_config = config
    else:
        raise ValueError('{} is not support now.'.format(opt_type))
    return opt
