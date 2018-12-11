import argparse
import torch
import torch.nn as nn
from model.model_builder import make_model
from data.data_loader import make_dataloader
from opt.optimizer import make_optimizer
from util import config
from api import trainer

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument('--config_path', default=None, type=str,
                    help='path to config file')
parser.add_argument('--model_dir', default=None, type=str,
                    help='path to model directory')

def main():
    local_rank = args.local_rank
    config_path = args.config_path
    # 0. config
    cfg = config.read_file(config_path)

    torch.cuda.set_device(local_rank)
    # 1. data
    data_loader = make_dataloader(cfg)
    # 2. model
    model = make_model(cfg)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank,
    )
    # 3. optimizer
    optimizer = make_optimizer(cfg)

    tl = trainer.Launcher(
        model_dir=args.model_dir,
        model=model,
        optimizer=optimizer)

    tl.train_by_config(data_loader, config=cfg)


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.config_path is not None, 'The config file is needed.'
    assert args.model_dir is not None, 'The model dir is needed.'
    main()
