import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from simplecv.module.model_builder import make_model
from simplecv.data.data_loader import make_dataloader
from simplecv.opt.optimizer import make_optimizer
from simplecv.opt.learning_rate import make_learningrate
from simplecv.util import config
from simplecv.core import trainer
from simplecv.util import param_util
from simplecv.core import default_backward

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex")

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument('--config_path', default=None, type=str,
                    help='path to config file')
parser.add_argument('--model_dir', default=None, type=str,
                    help='path to model directory')
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu')
parser.add_argument('--opt_level', type=str, default='O0', help='O0, O1, O2, O3')
parser.add_argument('--keep_batchnorm_fp32', type=bool, default=None, help='')

OPT_LEVELS = ['O0', 'O1', 'O2', 'O3']


def run(local_rank,
        config_path,
        model_dir,
        opt_level='O0',
        cpu_mode=False,
        after_construct_launcher_callbacks=None):
    # 0. config
    cfg = config.import_config(config_path)

    # 1. model
    model = make_model(cfg['model'])
    if cfg['train'].get('apex_sync_bn', False):
        model = apex.parallel.convert_syncbn_model(model)
    # 2. optimizer
    lr_schedule = make_learningrate(cfg['learning_rate'])
    cfg['optimizer']['params']['lr'] = lr_schedule.base_lr
    optimizer = make_optimizer(cfg['optimizer'], params=model.parameters())

    if not cpu_mode:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl", init_method="env://"
            )
        model.to(torch.device('cuda'))
        if dist.is_available():
            # if OPT_LEVELS.index(opt_level) < 2:
            #     keep_batchnorm_fp32 = None
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level=opt_level,
                                              )
            model = DDP(
                model, delay_allreduce=True,
            )
    # 3. data
    traindata_loader = make_dataloader(cfg['data']['train'])
    testdata_loader = make_dataloader(cfg['data']['test']) if 'test' in cfg['data'] else None
    tl = trainer.Launcher(
        model_dir=model_dir,
        model=model,
        optimizer=optimizer,
        lr_schedule=lr_schedule)
    # log dist train info
    tl.logger.info('[NVIDIA/apex] amp optimizer. opt_level = {}'.format(opt_level))
    tl.logger.info('apex sync bn: {}'.format('on' if cfg['train'].get('apex_sync_bn', False) else 'off'))
    tl.override_backward(default_backward.amp_backward)

    if after_construct_launcher_callbacks is not None:
        for f in after_construct_launcher_callbacks:
            f(tl)

    tl.train_by_config(traindata_loader, config=cfg['train'], test_data_loader=testdata_loader)


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.config_path is not None, 'The config file is needed.'
    assert args.model_dir is not None, 'The model dir is needed.'
    run(local_rank=args.local_rank,
        config_path=args.config_path,
        model_dir=args.model_dir,
        opt_level=args.opt_level,
        cpu_mode=args.cpu)
