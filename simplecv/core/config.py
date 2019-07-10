from ast import literal_eval
import warnings


class AttrDict(dict):
    def __init__(self, **kwargs):
        super(AttrDict, self).__init__(**kwargs)
        self.update(kwargs)

    @staticmethod
    def from_dict(dict):
        ad = AttrDict()
        ad.update(dict)
        return ad

    def __setitem__(self, key: str, value):
        super(AttrDict, self).__setitem__(key, value)
        super(AttrDict, self).__setattr__(key, value)

    def update(self, config: dict):
        for k, v in config.items():
            if k not in self:
                self[k] = AttrDict()
            if isinstance(v, dict):
                self[k].update(v)
            else:
                self[k] = v

    def update_from_list(self, str_list: list):
        assert len(str_list) % 2 == 0
        for key, value in zip(str_list[0::2], str_list[1::2]):
            key_list = key.split('.')
            item = None
            last_key = key_list.pop()
            for sub_key in key_list:
                item = self[sub_key] if item is None else item[sub_key]
            try:
                item[last_key] = literal_eval(value)
            except ValueError:
                item[last_key] = value
                warnings.warn('a string value is set to {}'.format(key))
