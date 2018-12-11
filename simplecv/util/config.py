import importlib


def read_file(config_path):
    raise NotImplementedError


def import_config(config_name):
    m = importlib.import_module(name='configs.{}'.format(config_name))
    return m.config
