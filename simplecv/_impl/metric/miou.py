import torch
import simplecv._impl.metric.function as mF
from simplecv.util.logger import get_console_file_logger
import logging
import prettytable as pt


class MeanIntersectionOverUnion(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._total_cm = torch.zeros(num_classes, num_classes).to_sparse()

    def __call__(self, y_true, y_pred):
        sparse_cm = mF.th_confusion_matrix(y_true, y_pred, self.num_classes, to_dense=False)
        self._total_cm += sparse_cm

    def summary(self, log_dir=None):
        iou_per_class = mF.intersection_over_union_per_class(self._total_cm.to_dense())
        miou = iou_per_class.mean()

        tb = pt.PrettyTable()
        tb.field_names = ['class', 'iou']
        for idx, iou in enumerate(iou_per_class):
            tb.add_row([idx, iou])
        tb.add_row(['mIoU', miou])
        if log_dir is not None:
            logger = get_console_file_logger('mIoU', logging.INFO, log_dir)
            logger.info('\n' + tb.get_string())
        else:
            print(tb)
        return iou_per_class, miou
