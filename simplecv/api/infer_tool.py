import torch
from simplecv.util import config
from simplecv.module.model_builder import make_model
from simplecv.util import checkpoint
from simplecv.util.logger import get_logger

logger = get_logger(__name__)


def build_from_file(config_path):
    cfg = config.import_config(config_path)
    model = make_model(cfg['model'])
    return model


def build_and_load_from_file(config_path, checkpoint_path):
    model = build_from_file(config_path)
    model_state_dict = checkpoint.load_model_state_dict_from_ckpt(checkpoint_path)
    global_step = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)[
        checkpoint.CheckPoint.GLOBALSTEP]
    model.load_state_dict(model_state_dict)
    model.eval()
    logger.info('[Load params] from {}'.format(checkpoint_path))
    return model, global_step
