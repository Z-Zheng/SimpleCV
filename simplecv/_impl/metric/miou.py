import torch
import simplecv._impl.metric.function as mF
from simplecv.util.logger import get_console_file_logger
import logging
import prettytable as pt
import numpy as np
from scipy import sparse


class THMeanIntersectionOverUnion(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._total_cm = torch.zeros(num_classes, num_classes).to_sparse()

    def __call__(self, y_true, y_pred):
        sparse_cm = mF.th_confusion_matrix(y_true.view(-1), y_pred.view(-1), self.num_classes, to_dense=False)
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


class NPMeanIntersectionOverUnion(object):
    def __init__(self, num_classes, logdir=None):
        self.num_classes = num_classes
        self._total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)
        self.logdir = logdir

    @staticmethod
    def compute_iou_per_class(confusion_matrix):
        """
        Args:
            confusion_matrix: numpy array [num_classes, num_classes] row - gt, col - pred
        Returns:
            iou_per_class: float32 [num_classes, ]
        """
        sum_over_row = np.sum(confusion_matrix, axis=0)
        sum_over_col = np.sum(confusion_matrix, axis=1)
        diag = np.diag(confusion_matrix)
        denominator = sum_over_row + sum_over_col - diag

        iou_per_class = diag / denominator

        return iou_per_class

    def forward(self, y_true, y_pred):
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()

        y_pred = y_pred.reshape((-1,))
        y_true = y_true.reshape((-1,))

        v = np.ones_like(y_pred)
        cm = sparse.coo_matrix((v, (y_true, y_pred)), shape=(self.num_classes, self.num_classes), dtype=np.float32)
        self._total += cm

    def summary(self):
        dense_cm = self._total.toarray()
        iou_per_class = NPMeanIntersectionOverUnion.compute_iou_per_class(dense_cm)
        miou = iou_per_class.mean()

        tb = pt.PrettyTable()
        tb.field_names = ['class', 'iou']
        for idx, iou in enumerate(iou_per_class):
            tb.add_row([idx, iou])
        tb.add_row(['mIoU', miou])
        if self.logdir is not None:
            logger = get_console_file_logger('mIoU', logging.INFO, self.logdir)
            logger.info('\n' + tb.get_string())
        else:
            print(tb)
        return iou_per_class, miou
