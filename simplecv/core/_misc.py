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
