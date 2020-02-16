from simplecv.util.logger import get_console_file_logger
import logging
import prettytable as pt
import numpy as np
from scipy import sparse
import torch
import os
import time

EPS = 1e-7


class NPPixelMetric(object):
    def __init__(self, num_classes, logdir=None, logger=None):
        self.num_classes = num_classes
        self._total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)
        if logdir is not None:
            os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir
        if logdir is not None and logger is None:
            self._logger = get_console_file_logger('PixelMertic', logging.INFO, self.logdir)
        elif logger is not None:
            self._logger = logger
        else:
            self._logger = None

    @property
    def logger(self):
        return self._logger

    def reset(self):
        num_classes = self.num_classes
        self._total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)

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

        iou_per_class = diag / (denominator + EPS)

        return iou_per_class

    @staticmethod
    def compute_recall_per_class(confusion_matrix):
        sum_over_row = np.sum(confusion_matrix, axis=1)
        diag = np.diag(confusion_matrix)
        recall_per_class = diag / (sum_over_row + EPS)
        return recall_per_class

    @staticmethod
    def compute_precision_per_class(confusion_matrix):
        sum_over_col = np.sum(confusion_matrix, axis=0)
        diag = np.diag(confusion_matrix)
        precision_per_class = diag / (sum_over_col + EPS)
        return precision_per_class

    @staticmethod
    def compute_overall_accuracy(confusion_matrix):
        diag = np.diag(confusion_matrix)
        return np.sum(diag) / (np.sum(confusion_matrix) + EPS)

    @staticmethod
    def compute_F_measure_per_class(confusion_matrix, beta=1.0):
        precision_per_class = NPPixelMertic.compute_precision_per_class(confusion_matrix)
        recall_per_class = NPPixelMertic.compute_recall_per_class(confusion_matrix)
        F1_per_class = (1 + beta ** 2) * precision_per_class * recall_per_class / (
                (beta ** 2) * precision_per_class + recall_per_class + EPS)

        return F1_per_class

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

    def _log_summary(self, table, dense_cm):
        if self.logger is not None:
            self.logger.info('\n' + table.get_string())
            if self.logdir is not None:
                np.save(os.path.join(self.logdir, 'confusion_matrix-{time}.npy'.format(time=time.time())), dense_cm)
        else:
            print(table)

    def summary_iou(self):
        dense_cm = self._total.toarray()
        iou_per_class = NPPixelMertic.compute_iou_per_class(dense_cm)
        miou = iou_per_class.mean()

        tb = pt.PrettyTable()
        tb.field_names = ['class', 'iou']
        for idx, iou in enumerate(iou_per_class):
            tb.add_row([idx, iou])
        tb.add_row(['mIoU', miou])

        self._log_summary(tb, dense_cm)

        return tb

    def summary_all(self):
        dense_cm = self._total.toarray()
        iou_per_class = NPPixelMertic.compute_iou_per_class(dense_cm)
        miou = iou_per_class.mean()
        F1_per_class = NPPixelMertic.compute_F_measure_per_class(dense_cm, beta=1.0)
        mF1 = F1_per_class.mean()
        overall_accuracy = NPPixelMertic.compute_overall_accuracy(dense_cm)
        precision_per_class = NPPixelMertic.compute_precision_per_class(dense_cm)
        mprec = precision_per_class.mean()
        recall_per_class = NPPixelMertic.compute_recall_per_class(dense_cm)
        mrecall = recall_per_class.mean()

        tb = pt.PrettyTable()
        tb.field_names = ['class', 'iou', 'f1', 'precision', 'recall']

        for idx, (iou, f1, precision, recall) in enumerate(
                zip(iou_per_class, F1_per_class, precision_per_class, recall_per_class)):
            tb.add_row([idx, iou, f1, precision, recall])

        tb.add_row(['mean', miou, mF1, mprec, mrecall])
        tb.add_row(['OA', overall_accuracy, '-', '-', '-'])

        self._log_summary(tb, dense_cm)

        return tb


NPPixelMertic = NPPixelMetric
