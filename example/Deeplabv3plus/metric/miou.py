from scipy import sparse
import numpy as np
import torch
import time
from simplecv.util.logger import eval_progress, speed


class IoU(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)

    def update(self, y_pred, y_true):
        """

        Args:
            y_pred: 1-D
            y_true: 1-D

        Returns:

        """
        v = np.ones_like(y_pred)
        cm = sparse.coo_matrix((v, (y_true, y_pred)), shape=(self.num_classes, self.num_classes), dtype=np.float32)
        self._total += cm

    def value(self):
        dense_cm = self._total.toarray()
        return compute_iou_per_class(dense_cm)


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


def evaluate_cls_fn(self, test_dataloader):
    torch.cuda.empty_cache()
    self.model.eval()
    total_time = 0.
    iou_metric = IoU(self.model.module.num_classes)
    with torch.no_grad():
        for idx, (ret, ret_gt) in enumerate(test_dataloader):
            start = time.time()
            y = self.model(ret)

            cls = y.argmax(dim=1).cpu()
            cls = cls.numpy()
            cls_gt = ret_gt['cls']
            cls_gt = cls_gt.numpy()
            y_true = cls_gt.ravel()
            y_pred = cls.ravel()
            valid_inds = np.where(y_true != 255)[0]
            y_true = y_true[valid_inds]
            y_pred = y_pred[valid_inds]

            iou_metric.update(y_pred, y_true)
            time_cost = round(time.time() - start, 3)

            total_time += time_cost
            speed(self._logger, time_cost, 'batch')
            eval_progress(self._logger, idx + 1, len(test_dataloader))
    torch.cuda.empty_cache()
    speed(self._logger, round(total_time / len(test_dataloader), 3), 'batch (avg)')

    ious = iou_metric.value()
    miou = ious.mean()
    metric_dict = {}
    for i in range(len(ious)):
        metric_dict['iou_{}'.format(i)] = float(ious[i])
    metric_dict['miou'] = float(miou)

    self._logger.eval_log(metric_dict=metric_dict, step=self.checkpoint._global_step)