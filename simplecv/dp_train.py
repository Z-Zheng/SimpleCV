import argparse
import torch
import torch.nn as nn
from simplecv.module.model_builder import make_model
from simplecv.data.data_loader import make_dataloader
from simplecv.opt.optimizer import make_optimizer
from simplecv.opt.learning_rate import make_learningrate
from simplecv.util import config
from simplecv.core import trainer
from simplecv.core._misc import merge_dict

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default=None, type=str,
                    help='path to config file')
parser.add_argument('--model_dir', default=None, type=str,
                    help='path to model directory')
parser.add_argument('--cpu', action='store_true', default=False, help='')


def run(config_path, model_dir, cpu_mode=False, after_construct_launcher_callbacks=None):
    # 0. config
    cfg = config.import_config(config_path)

    # 1. model
    model = make_model(cfg['model'])

    if not cpu_mode:
        if torch.cuda.is_available():
            model.to(torch.device('cuda'))
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # 2. data
    traindata_loader = make_dataloader(cfg['data']['train'])
    testdata_loader = make_dataloader(cfg['data']['test']) if 'test' in cfg['data'] else None

    # 3. optimizer
    lr_schedule = make_learningrate(cfg['learning_rate'])
    cfg['optimizer']['params']['lr'] = lr_schedule.base_lr
    optimizer = make_optimizer(cfg['optimizer'], params=model.parameters())

    tl = trainer.Launcher(
        model_dir=model_dir,
        model=model,
        optimizer=optimizer,
        lr_schedule=lr_schedule)

    if after_construct_launcher_callbacks is not None:
        for f in after_construct_launcher_callbacks:
            f(tl)

    tl.train_by_config(traindata_loader, config=merge_dict(cfg['train'],cfg['test']), test_data_loader=testdata_loader)


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.config_path is not None, 'The config file is needed.'
    assert args.model_dir is not None, 'The model dir is needed.'
    run(config_path=args.config_path,
        model_dir=args.model_dir,
        cpu_mode=args.cpu)
