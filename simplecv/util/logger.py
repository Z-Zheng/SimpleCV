import logging
import time
import tensorboardX
import numpy as np

logging.basicConfig(level=logging.INFO)


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    return logger


class Logger(object):
    def __init__(self,
                 name,
                 level=logging.INFO,
                 use_tensorboard=False,
                 tensorboard_logdir=None):
        self._logger = logging.getLogger(name)
        self._level = level
        self._logger.setLevel(level)
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard and tensorboard_logdir is None:
            raise ValueError('logdir is not None if you use tensorboard')
        if self.use_tensorboard:
            self.summary_w = tensorboardX.SummaryWriter(log_dir=tensorboard_logdir)

    def info(self, value):
        self._logger.info(value)

    def on(self):
        self._logger.setLevel(self._level)
        self.use_tensorboard = True

    def off(self):
        self._logger.setLevel(100)
        self.use_tensorboard = False

    def train_log(self, step, loss_dict, time_cost, lr, metric_dict=None):
        loss_info = ''.join(
            ['{name} = {value}\t'.format(name=name, value=str(round(value, 6)).ljust(6, '0')) for name, value in
             loss_dict.items()])
        step_info = 'step: {}\t'.format(int(step))
        time_cost_info = '({} sec / step)'.format(round(time_cost, 3))

        if metric_dict:
            metric_info = ''.join(
                ['[Train] {name} = {value}\t'.format(name=name, value=np.round(value, 6)) for name, value in
                 metric_dict.items()])
        else:
            metric_info = ''
        lr_info = 'lr = {}'.format(str(round(lr, 6)))
        msg = '{loss}\t{metric}\t{lr}\t{step}\t{time}'.format(loss=loss_info, metric=metric_info, step=step_info,
                                                              lr=lr_info,
                                                              time=time_cost_info)
        self._logger.info(msg)

        if self.use_tensorboard and step % 100 == 0:
            self.train_summary(step, loss_dict, time_cost, lr, metric_dict)

    def train_summary(self, step, loss_dict, time_cost, lr, metric_dict=None):
        for name, value in loss_dict.items():
            self.summary_w.add_scalar('loss/{}'.format(name), float(value), global_step=step)
        if metric_dict:
            for name, value in metric_dict.items():
                if isinstance(value, float):
                    self.summary_w.add_scalar('train/{}'.format(name), value, global_step=step)
                elif isinstance(value, np.ndarray):
                    for idx, nd_v in enumerate(value):
                        self.summary_w.add_scalar('train/{}_{}'.format(name, idx), float(nd_v), global_step=step)

        self.summary_w.add_scalar('sec_per_step', float(time_cost), global_step=step)
        self.summary_w.add_scalar('learning_rate', float(lr), global_step=step)

    def eval_log(self, metric_dict, step=None):
        for name, value in metric_dict.items():
            self._logger.info('[Eval] {name} = {value}'.format(name=name, value=np.round(value, 6)))
        if self.use_tensorboard:
            self.eval_summary(metric_dict, step)

    def eval_summary(self, metric_dict, step):
        if step is None:
            step = 1
        for name, value in metric_dict.items():
            if isinstance(value, float):
                self.summary_w.add_scalar('eval/{}'.format(name), value, global_step=step)
            elif isinstance(value, np.ndarray):
                for idx, nd_v in enumerate(value):
                    self.summary_w.add_scalar('eval/{}_{}'.format(name, idx), float(nd_v), global_step=step)

    def forward_times(self, forward_times):
        self._logger.info('use {} forward and 1 backward mode.'.format(forward_times))

    def equation(self, name, value):
        self._logger.info('{name} = {value}'.format(name=name, value=value))

    def approx_equation(self, name, value):
        self._logger.info('{name} ~= {value}'.format(name=name, value=value))


def save_log(logger, checkpoint_name):
    logger.info('{} has been saved.'.format(checkpoint_name))


def restore_log(logger, checkpoint_name):
    logger.info('{} has been restored.'.format(checkpoint_name))


def eval_start(logger):
    logger.info('Start evaluation at {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


def eval_progress(logger, cur, total):
    logger.info('[Eval] {}/{}'.format(cur, total))


def speed(logger, sec, unit='im'):
    logger.info('[Speed] {} s/{}'.format(sec, unit))
