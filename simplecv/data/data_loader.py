from simplecv.util import registry


def make_dataloader(config):
    dataloader_type = config['type']
    if dataloader_type in registry.DATALOADER:
        data_loader = registry.DATALOADER[dataloader_type](config['params'])
    else:
        raise ValueError('{} is not support now.'.format(dataloader_type))

    return data_loader
