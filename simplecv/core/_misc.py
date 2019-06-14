from functools import wraps


class Callback(object):
    def __init__(self, before_callbacks=None, after_callbacks=None):
        self.before_callbacks = before_callbacks
        self.after_callbacks = after_callbacks

    def __call__(self, func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if self.before_callbacks is not None:
                for callback in self.before_callbacks:
                    callback(*args, **kwargs)

            out = func(*args, **kwargs)

            if self.after_callbacks is not None:
                for callback in self.after_callbacks:
                    callback(out)
            return out

        return wrapped_func()


def merge_dict(dict1: dict, dict2: dict):
    # check whether redundant key
    redundant_keys = [key for key in dict1 if key in dict2]
    if len(redundant_keys) > 0:
        raise ValueError('Duplicate keys: {}'.format(redundant_keys))

    merged = dict1.copy()
    merged.update(dict2)

    return merged
