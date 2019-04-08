import warnings

try:
    from apex import amp
except ImportError:
    # warnings.warn("If you want to use apex, please install apex from https://www.github.com/nvidia/apex")
    pass

__all__ = [
    'default_backward',
    'amp_backward'
]


def default_backward(self, total_loss, optimizer):
    total_loss.backward()


def amp_backward(self, total_loss, optimizer):
    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
        scaled_loss.backward()
