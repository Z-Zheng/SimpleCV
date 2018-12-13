from simplecv import registry
import torch.nn as nn

registry.LOSS.register('cross_entropy', nn.CrossEntropyLoss)
