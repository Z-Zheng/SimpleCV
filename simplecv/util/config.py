import importlib


def import_config(config_name, prefix='configs'):
    m = importlib.import_module(name='{}.{}'.format(prefix, config_name))
    return m.config
