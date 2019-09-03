from simplecv.core import trainer
from simplecv.core.trainer import LauncherPlugin
from simplecv.opt.learning_rate import set_lr


class LRSearchBase(LauncherPlugin):
    def __init__(self, data_loader, loss_key='total_loss'):
        super(LRSearchBase, self).__init__('lr_search_base')
        self.data_loader = data_loader
        self.loss_key = loss_key

    def function(self, launcher: trainer.Launcher):
        raise NotImplementedError


class LinearSearch(LRSearchBase):
    def __init__(self,
                 data_loader,
                 loss_key='total_loss',
                 lr_search_space=(0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1),
                 search_iters=1000,
                 ):
        super(LinearSearch, self).__init__(data_loader, loss_key)
        self.search_space = lr_search_space
        self.search_results = list()
        self.search_iters = search_iters

    def function(self, launcher: trainer.Launcher):
        launcher.logger.info('Start LR linear search...')
        original_lr = launcher.lr
        best_lr, best_loss = None, 1e6
        for lr in self.search_space:
            set_lr(launcher.optimizer, lr)
            # don't save ckpt during searching, simply set `save_ckpt_interval_epoch` to large enough value
            train_config = dict(num_iters=self.search_iters, save_ckpt_interval_epoch=999999)
            loss_dict = launcher.train_iters(self.data_loader, None, **train_config)
            launcher._ckpt._global_step = 0
            if self.loss_key in loss_dict:
                pass
            loss = loss_dict[self.loss_key]
            self.search_results.append((lr, loss))
            if loss < best_loss:
                best_lr = lr
                best_loss = loss

        for idx, result in enumerate(self.search_results):
            launcher.logger.info('[LR search][{}] lr: {}, loss: {}'.format(idx + 1, *result))
        launcher.logger.info('[LR search] best_lr: {}, best_loss: {}'.format(best_lr, best_loss))
        # recover status
        set_lr(launcher.optimizer, original_lr)
