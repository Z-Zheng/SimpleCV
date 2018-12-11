from util.logger import get_logger

logger = get_logger(__name__)


def trainable_parameters(module):
    ret = []
    total = 0
    for idx, p in enumerate(module.parameters()):
        if p.requires_grad:
            ret.append(p)
        total = idx + 1
    logger.info('[trainable params] {}/{}'.format(len(ret), total))
    return ret


if __name__ == '__main__':
    from torchvision.models.resnet import resnet18

    model = resnet18(False)
    trainable_parameters(model)
