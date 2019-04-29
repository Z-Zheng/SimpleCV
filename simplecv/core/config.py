class AttrDict(dict):
    def __init__(self, **kwargs):
        super(AttrDict, self).__init__(**kwargs)
        self.update(kwargs)

    def __setitem__(self, key: str, value):
        if isinstance(value, dict):
            value = AttrDict(**value)
        super(AttrDict, self).__setitem__(key, value)
        super(AttrDict, self).__setattr__(key, value)

    def update(self, config):
        for k, v in config.items():
            self[k] = v
