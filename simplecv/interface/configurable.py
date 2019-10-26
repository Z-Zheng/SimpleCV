from simplecv.core.config import AttrDict


class ConfigurableMixin(object):
    def __init__(self, config):
        self._cfg = AttrDict(

        )
        self.set_defalut_config()
        self._cfg.update(config)

    def set_defalut_config(self):
        raise NotImplementedError

    @property
    def config(self):
        return self._cfg
