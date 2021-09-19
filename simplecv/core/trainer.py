import os
from simplecv.util.logger import Logger
from simplecv.util.checkpoint import CheckPoint
from simplecv.data.iterator import get_iterator
from simplecv.util import tensor_util
import time
from torch.optim.lr_scheduler import _LRScheduler
from simplecv.interface.learning_rate import LearningRateBase
from simplecv.util import param_util
import functools
import types
import torch
from torch.nn.utils import clip_grad
from simplecv.util.dist import reduce_loss_dict, get_rank

__all__ = ['Launcher',
           'LauncherPlugin']


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
        if self._master:
            self.init_model_dir()
            self._logger = Logger('SimpleCV', use_tensorboard=self._master, tensorboard_logdir=model_dir)
            self._logger.on()
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self._ckpt = CheckPoint(self)
        self._training = False

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
        class _FakeLogger(object):
            def info(self, value):
                pass

        if self._master:
            return self._logger
        else:
            return _FakeLogger()

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
                losses = reduce_loss_dict(losses)
                for name, value in losses.items():
                    if name not in loss_dict:
                        loss_dict[name] = 0.0
                    loss_dict[name] += value.item()
                loss_dict['total_loss'] += sum(list(loss_dict.values()))
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
        eval_per_epoch = kwargs.get('eval_per_epoch', False)
        tensorboard_interval_step = kwargs.get('tensorboard_interval_step', 100)
        log_interval_step = kwargs.get('log_interval_step', 1)
        distributed = kwargs.get('distributed', False)
        summary_grads = kwargs.get('summary_grads', False)
        summary_weights = kwargs.get('summary_weights', False)
        iterator_type = kwargs.get('iterator_type', 'normal')
        save_ckpt_interval_epoch = kwargs.get('save_ckpt_interval_epoch', 1)
        eval_interval_epoch = kwargs.get('eval_interval_epoch', 1)

        iterator = get_iterator(iterator_type)(train_data_loader)

        call_backs = [(self._ckpt.save, save_ckpt_interval_epoch)]
        signal_loss_dict = dict()
        if eval_per_epoch:
            call_backs.append((functools.partial(self.evaluate, test_data_loader, kwargs), eval_interval_epoch))
        while self._ckpt.global_step < num_iters:
            start = time.time()
            if distributed:
                iterator.set_seed_for_dist_sampler(self._ckpt.global_step)
            data_list = iterator.next(forward_times,
                                      call_backs=call_backs,
                                      is_master=self._master)
            data_time = time.time() - start
            self._model.train()
            loss_dict = self.compute_loss_gradient(data_list)
            signal_loss_dict = loss_dict.copy()
            # clip gradient
            grad_clip_config = self._optimizer.simplecv_config.get('grad_clip', dict(max_norm=35, norm_type=2))
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                      **grad_clip_config)
            if self._master:
                if summary_grads:
                    self._logger.summary_grads(module=self.model.module, step=self._ckpt.global_step)
            self.apply_gradient()
            if self._master:
                time_cost = time.time() - start

                self._logger.train_log(step=self._ckpt.global_step, loss_dict=loss_dict,
                                       data_time=data_time,
                                       time_cost=time_cost, lr=self.lr, num_iters=num_iters,
                                       tensorboard_interval_step=tensorboard_interval_step,
                                       log_interval_step=log_interval_step)
                if summary_weights:
                    self._logger.summary_weights(module=self.model.module, step=self._ckpt.global_step)
        return signal_loss_dict

    def train_epochs(self, train_data_loader, test_data_loader=None, **kwargs):
        num_epochs = kwargs.get('num_epochs', -1)
        forward_times = kwargs.get('forward_times', 1)
        tensorboard_interval_step = kwargs.get('tensorboard_interval_step', 100)
        log_interval_step = kwargs.get('log_interval_step', 1)

        iterator_type = kwargs.get('iterator_type', 'normal')

        iterator = get_iterator(iterator_type)(train_data_loader)
        signal_loss_dict = dict()
        for i in range(num_epochs):
            self._model.train()
            if kwargs.get('distributed', False):
                iterator.set_seed_for_dist_sampler(self._ckpt.global_step)
            for data_list in iterator.iter(forward_times=forward_times):
                start = time.time()
                loss_dict = self.compute_loss_gradient(data_list)
                signal_loss_dict = loss_dict.copy()
                # clip gradient
                grad_clip_config = self._optimizer.simplecv_config.get('grad_clip', dict(max_norm=35, norm_type=2))
                clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          **grad_clip_config)
                if self._master:
                    if kwargs.get('summary_grads', False):
                        self._logger.summary_grads(module=self.model.module, step=self._ckpt.global_step)
                self.apply_gradient()
                if self._master:
                    time_cost = time.time() - start

                    self._logger.train_log(step=self._ckpt.global_step, loss_dict=loss_dict,
                                           time_cost=time_cost, lr=self.lr, num_iters=None,
                                           tensorboard_interval_step=tensorboard_interval_step,
                                           log_interval_step=log_interval_step)
                    if kwargs.get('summary_weights', False):
                        self._logger.summary_weights(module=self.model.module, step=self._ckpt.global_step)

            if self._master:
                self._ckpt.save()
        return signal_loss_dict

    def train_by_config(self, train_data_loader, config, test_data_loader=None, ):
        self._training = True
        if config.get('resume_from_last', True):
            self.init()
        self.model.train()
        forward_times = config['forward_times'] if 'forward_times' in config else 1

        if self._master:
            param_util.trainable_parameters(self.model, self._logger)
            param_util.count_model_parameters(self.model, self._logger)
            self._logger.equation('batch_size_per_gpu', train_data_loader.batch_sampler.batch_size)
            self._logger.forward_times(forward_times)
        if 'num_epochs' in config and 'num_iters' not in config:
            if self._master:
                self._logger.equation('num_epochs', config['num_epochs'])
                self._logger.equation('num_iters', config['num_epochs'] * len(train_data_loader))
            signal_loss_dict = self.train_epochs(train_data_loader, test_data_loader=test_data_loader, **config)

        elif 'num_epochs' not in config and 'num_iters' in config:
            if self._master:
                self._logger.approx_equation('num_epochs',
                                             round(config['num_iters'] * forward_times / len(train_data_loader), 1))
                self._logger.equation('num_iters', config['num_iters'])
            signal_loss_dict = self.train_iters(train_data_loader, test_data_loader=test_data_loader, **config)

        else:
            raise ValueError('`num_epochs` is mutually exclusive `num_iters`. Please only use one of them')
        if self._master:
            self._ckpt.save()
            if config.get('eval_after_train', True):
                self.evaluate(test_data_loader, config)
        return signal_loss_dict

    def init(self):
        if self._master:
            self.init_model_dir()
        self._ckpt.try_resume()

    def init_model_dir(self):
        os.makedirs(self._model_dir, exist_ok=True)

    def evaluate(self, data_loader, config=None):
        if not self._training:
            self.init()
        self._evaluate_fn(data_loader, config)

    def evaluate_last_ckpt(self, data_loader):
        self.init()
        self._evaluate_fn(data_loader)

    def _evaluate_fn(self, data_loader, config=None):
        raise NotImplementedError

    def backward(self, total_loss, optimizer, **kwargs):
        total_loss.backward()

    def override_evaluate(self, fn):
        self._evaluate_fn = types.MethodType(fn, self)

    def override_backward(self, fn):
        self.backward = types.MethodType(fn, self)

    def invoke_plugin(self, plugin_name, *args, **kwargs):
        if hasattr(self, plugin_name):
            getattr(self, plugin_name)(*args, **kwargs)
        else:
            raise ModuleNotFoundError('plugin: {} is not found.'.format(plugin_name))


class LauncherPlugin(object):
    def __init__(self, name):
        self.plugin_name = name

    def register(self, launcher: Launcher):
        assert isinstance(launcher, Launcher)
        if hasattr(launcher, self.plugin_name):
            raise ValueError('plugin_name: {} has existed.'.format(self.plugin_name))
        launcher.__setattr__(self.plugin_name, types.MethodType(self.function, launcher))

    def function(self, launcher: Launcher):
        raise NotImplementedError


def scale_dict(input_dict, scale):
    for k, v in input_dict.items():
        input_dict[k] = v * scale
    return input_dict


def average_dict(input_dict):
    for k, v in input_dict.items():
        input_dict[k] = v.mean() if v.ndimension() != 0 else v
    return input_dict
