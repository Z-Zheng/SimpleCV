# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging

logging.basicConfig(level=logging.INFO)


def _register_generic(module_dict, module_name, module, override=False):
    module_name = module_name if module_name else module.__name__
    if not override:
        if module_name in module_dict:
            logging.warning('{} has been in module_dict.'.format(module_name))
    module_dict[module_name] = module


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...
    3): used as decorator when declaring the module named via __name__:
        @some_registry.register()
        def foo():
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name=None, module=None, override=False):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module, override)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn, override)
            return fn

        return register_fn


LR = Registry()
OPT = Registry()
DATALOADER = Registry()
MODEL = Registry()
LOSS = Registry()
OP = Registry()
