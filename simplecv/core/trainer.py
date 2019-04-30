import os
import torch.distributed as dist
from simplecv.util.logger import Logger
from simplecv.util.checkpoint import CheckPoint
from simplecv.data.iterator import Iterator
from simplecv.util import tensor_util
import time
from torch.optim.lr_scheduler import _LRScheduler
from simplecv.opt.learning_rate import LearningRateBase
from simplecv.util import param_util
import functools
import types
import torch
from torch.nn.utils import clip_grad
from simplecv.core import default_backward


def get_rank():
    try:
        if not dist.is_initialized():
            return 0
    except AttributeError:
        return 0
    return dist.get_rank()


class Launcher(object):
    def __init__(self,
                 model_dir,
                 model,
                 optimizer,
                 lr_schedule):
        self._model_dir = model_dir
        self._model = model
        self._optimizer = optimizer
        self._lr_schedule = lr_schedule
        self._master = get_rank() == 0
        self._logger = Logger('SimpleCV', use_tensorboard=self._master, tensorboard_logdir=model_dir)
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self._master:
            self._logger.on()
        else:
            self._logger.off()
        self._ckpt = CheckPoint(self)
        self.init()

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def checkpoint(self):
        return self._ckpt

    @property
    def lr(self):
        return self._optimizer.param_groups[0]['lr']

    @property
    def logger(self):
        return self._logger

    def compute_loss_gradient(self, data):
        """

        Args:
            data:

        Returns:

        """
        if not isinstance(data, list):
            data = [data]

        loss_dict = {'total_loss': 0.0}

        for d in data:
            d = tensor_util.to_device(d, self._device)
            msg_dict = self._model(*d)

            losses = {k: v for k, v in msg_dict.items() if k.endswith('loss')}

            # scale losses by 1. / forward times
            if len(data) != 1:
                losses = scale_dict(losses, 1. / len(data))

            losses = average_dict(losses)
            total_loss = sum([e for e in losses.values()])

            self.backward(total_loss, self.optimizer)

            # log losses
            with torch.no_grad():
                for name, value in losses.items():
                    if name not in loss_dict:
                        loss_dict[name] = 0.0
                    loss_dict[name] += value.item()
                loss_dict['total_loss'] += total_loss.item()
            # extra log message
            log_dict = {k: v for k, v in msg_dict.items() if not k.endswith('loss')}
            with torch.no_grad():
                if len(log_dict) != 0:
                    if len(data) != 1:
                        log_dict = scale_dict(log_dict, 1. / len(data))
                    avg_log_dict = average_dict(log_dict)
                    for name, value in avg_log_dict.items():
                        if name not in loss_dict:
                            loss_dict[name] = 0.0
                        loss_dict[name] += value.item() if isinstance(value, torch.Tensor) else value

        return loss_dict

    def apply_gradient(self):
        self._optimizer.step()
        self._optimizer.zero_grad()

        self._update_lr()
        self._ckpt.step()

    def _update_lr(self):
        if isinstance(self._lr_schedule, LearningRateBase):
            self._lr_schedule.step(self._ckpt.global_step, self._optimizer)
        elif isinstance(self._lr_schedule, _LRScheduler):
            self._lr_schedule.step()
        else:
            raise NotImplementedError()

    def train_iters(self, train_data_loader, test_data_loader=None,
                    **kwargs):
        num_iters = kwargs.get('num_iters', -1)
        forward_times = kwargs.get('forward_times', 1)
        eval_per_epoch = kwargs.get('eval_per_epoch', True)

        iterator = Iterator(train_data_loader)
        call_backs = [self._ckpt.save]
        if eval_per_epoch:
            call_backs.append(functools.partial(self.evaluate, test_data_loader))
        while self._ckpt.global_step < num_iters:
            start = time.time()
            if kwargs.get('distributed', False):
                iterator.set_seed_for_dist_sampler(self._ckpt.global_step)
            data_list = iterator.next(forward_times,
                                      call_backs=call_backs,
                                      is_master=self._master)
            self._model.train()
            loss_dict = self.compute_loss_gradient(data_list)
            # clip gradient
            grad_clip_config = self._optimizer.simplecv_config.get('grad_clip', dict(max_norm=35, norm_type=2))
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.module.parameters()),
                                      **grad_clip_config)
            if self._master:
                if kwargs.get('summary_grads', True):
                    self._logger.summary_grads(module=self.model.module, step=self._ckpt.global_step)
            self.apply_gradient()
            if self._master:
                time_cost = time.time() - start

                self._logger.train_log(step=self._ckpt.global_step, loss_dict=loss_dict,
                                       time_cost=time_cost, lr=self.lr, num_iters=num_iters)
                if kwargs.get('summary_weights', True):
                    self._logger.summary_weights(module=self.model.module, step=self._ckpt.global_step)

    def train_epochs(self, train_data_loader, test_data_loader=None, **kwargs):
        num_epochs = kwargs.get('num_epochs', -1)
        forward_times = kwargs.get('forward_times', 1)
        iterator = Iterator(train_data_loader)
        for i in range(num_epochs):
            self._model.train()
            if kwargs.get('distributed', False):
                iterator.set_seed_for_dist_sampler(self._ckpt.global_step)
            for data_list in iterator.iter(forward_times=forward_times):
                start = time.time()
                loss_dict = self.compute_loss_gradient(data_list)
                # clip gradient
                grad_clip_config = self._optimizer.simplecv_config.get('grad_clip', dict(max_norm=35, norm_type=2))
                clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.module.parameters()),
                                          **grad_clip_config)
                if self._master:
                    if kwargs.get('summary_grads', True):
                        self._logger.summary_grads(module=self.model.module, step=self._ckpt.global_step)
                self.apply_gradient()
                if self._master:
                    time_cost = time.time() - start

                    self._logger.train_log(step=self._ckpt.global_step, loss_dict=loss_dict,
                                           time_cost=time_cost, lr=self.lr)
                    if kwargs.get('summary_weights', True):
                        self._logger.summary_weights(module=self.model.module, step=self._ckpt.global_step)

            if self._master:
                self._ckpt.save()

    def train_by_config(self, train_data_loader, config, test_data_loader=None, ):
        self.model.train()
        forward_times = config['forward_times'] if 'forward_times' in config else 1

        if self._master:
            param_util.trainable_parameters(self.model)
            param_util.count_model_parameters(self.model)
            self._logger.equation('batch_size', train_data_loader.batch_sampler.batch_size)
            self._logger.forward_times(forward_times)
        if 'num_epochs' in config and 'num_iters' not in config:
            if self._master:
                self._logger.equation('num_epochs', config['num_epochs'])
                self._logger.equation('num_iters', config['num_epochs'] * len(train_data_loader))
            self.train_epochs(train_data_loader, test_data_loader=test_data_loader, **config)

        elif 'num_epochs' not in config and 'num_iters' in config:
            if self._master:
                self._logger.approx_equation('num_epochs',
                                             round(config['num_iters'] * forward_times / len(train_data_loader), 1))
                self._logger.equation('num_iters', config['num_iters'])
            self.train_iters(train_data_loader, test_data_loader=test_data_loader, **config)

        else:
            raise ValueError('`num_epochs` is mutually exclusive `num_iters`. Please only use one of them')
        if self._master:
            self._ckpt.save()
            if config.get('eval_after_train', True):
                self.evaluate(test_data_loader)

    def init(self):
        if self._master:
            self.init_model_dir()
        self._ckpt.try_resume()

    def init_model_dir(self):
        os.makedirs(self._model_dir, exist_ok=True)

    def evaluate(self, data_loader):
        raise NotImplementedError

    def backward(self, total_loss, optimizer, **kwargs):
        total_loss.backward()

    def override_evaluate(self, fn):
        self.evaluate = types.MethodType(fn, self)

    def override_backward(self, fn):
        self.backward = types.MethodType(fn, self)


def scale_dict(input_dict, scale):
    for k, v in input_dict.items():
        input_dict[k] = v * scale
    return input_dict


def average_dict(input_dict):
    for k, v in input_dict.items():
        input_dict[k] = v.mean()
    return input_dict
