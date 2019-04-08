try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex")

__all__ = [
    'default_backward',
    'amp_backward'
]


def default_backward(total_loss, optimizer):
    total_loss.backward()


def amp_backward(total_loss, optimizer):
    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
        scaled_loss.backward()
