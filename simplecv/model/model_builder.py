from simplecv.util import registry


def make_model(config):
    model_type = config['type']
    if model_type in registry.MODEL:
        model = registry.MODEL[model_type](config['params'])
    else:
        raise ValueError('{} is not support now.'.format(model_type))
    return model
