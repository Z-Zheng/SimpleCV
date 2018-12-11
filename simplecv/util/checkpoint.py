import os
from collections import OrderedDict
import torch
import json
from simplecv.util.logger import get_logger, save_log, restore_log

logger = get_logger(__name__)


class CheckPoint(object):
    MODEL = 'model'
    OPTIMIZER = 'opt'
    GLOBALSTEP = 'global_step'
    LASTCHECKPOINT = 'last'

    def __init__(self, launcher=None):
        self._launcher = launcher
        self._global_step = 0
        self._json_log = {CheckPoint.LASTCHECKPOINT: dict(
            step=0,
            name=''),
        }
        self.init_json_log_from_launcher()

    def set_global_step(self, value):
        if value >= 0:
            self._global_step = value
        else:
            raise ValueError('The global step must be larger than zero.')

    @property
    def global_step(self):
        return self._global_step

    def step(self):
        self._global_step += 1

    def set_launcher(self, launcher):
        self._launcher = launcher
        self.init_json_log_from_launcher()

    def save(self):
        ckpt = OrderedDict({
            CheckPoint.MODEL: self._launcher.model,
            CheckPoint.OPTIMIZER: self._launcher.optimizer,
            CheckPoint.GLOBALSTEP: self.global_step
        })
        filename = self.get_checkpoint_name(self.global_step)
        filepath = os.path.join(self._launcher.model_dir, filename)
        torch.save(ckpt, filepath)
        self._json_log[self.global_step] = filename
        if self.global_step > self._json_log[CheckPoint.LASTCHECKPOINT]['step']:
            self._json_log[CheckPoint.LASTCHECKPOINT]['step'] = self.global_step
            self._json_log[CheckPoint.LASTCHECKPOINT]['name'] = filename
        # log
        save_log(logger, filename)

    def load(self, filepath):
        ckpt = torch.load(filepath)

        return ckpt

    def try_resume(self):
        """ json -> ckpt_path -> ckpt -> launcher

        Returns:

        """
        if self._launcher is None:
            return
        # 1. json
        model_dir = self._launcher.model_dir
        json_log = self.load_json_log(model_dir)
        if json_log is None:
            return
        # 2. ckpt path
        last_path = json_log[CheckPoint.LASTCHECKPOINT]
        # 3. ckpt
        ckpt = self.load(last_path)
        # 4. resume
        self._launcher.model.load_state_dict(ckpt[CheckPoint.MODEL])
        self._launcher.optimizer.load_state_dict(ckpt[CheckPoint.OPTIMIZER])
        self._launcher.checkpoint.set_global_step(ckpt[CheckPoint.GLOBALSTEP])
        # log
        restore_log(logger, last_path)

    def init_json_log_from_launcher(self):
        if self._launcher is None:
            return

        model_dir = self._launcher.model_dir
        json_file = self.load_json_log(model_dir)

        self._json_log = json_file

    @staticmethod
    def load_json_log(model_dir):
        json_path = os.path.join(model_dir, 'checkpoint.json')
        if not os.path.exists(json_path):
            return None
        with open(json_path, 'r') as f:
            json_file = json.load(f)
        return json_file

    @staticmethod
    def get_checkpoint_name(global_step):
        return 'model-{}.pth'.format(global_step)
