from simplecv.interface import CVModule
from simplecv import registry
import torch
import torch.nn as nn
import torch.nn.functional as F


@registry.MODEL.register('deeplabv3plus')
class Deeplabv3plus(CVModule):
    def __init__(self, config):
        super(Deeplabv3plus, self).__init__(config)
        self.encoder = registry.MODEL[config['encoder']['type']](config['encoder']['params'])

        self.decoder = registry.MODEL[config['decoder']['type']](config['decoder']['params'])

        self.cls_pred_conv = nn.Conv2d(self.decoder.decoder_dim, self.config['other']['num_classes'], 1)

        self.loss_fn_dict = {}
        for name, item in self.config['loss'].items():
            self.loss_fn_dict[name] = registry.LOSS[item['type']](**item['params'])

    def forward(self, x, y=None, **kwargs):
        feat_list = self.encoder(x)

        out = self.decoder(*feat_list)

        out = self.cls_pred_conv(out)
        out = F.interpolate(out, scale_factor=self.config['other']['scale_factor'], mode='bilinear', align_corners=True)
        if self.training:
            ret_loss_dict = {}
            flat_y_true = torch.reshape(y, (-1,))
            flat_y_pred = torch.reshape(out, (-1,))

            ret_loss_dict['cls_loss'] = self.loss_fn_dict['cls_loss'](input=flat_y_pred, target=flat_y_true)
            return ret_loss_dict
        else:
            if self.config['other']['use_softmax']:
                out = torch.softmax(out, dim=1)
            else:
                out = torch.sigmoid(out)

        return out

    def set_defalut_config(self):
        pass
