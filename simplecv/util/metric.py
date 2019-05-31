import torch


def th_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None):
    """

    Args:
        y_true: 1-D tensor of shape [n_samples], label value starts from 1 rather than 0
        y_pred: 1-D tensor of shape [n_samples]
        num_classes: scalar
    Returns:

    """
    size = [num_classes + 1, num_classes + 1] if num_classes is not None else None
    y_true = y_true.float()
    y_pred = y_pred.float()
    if size is None:
        cm = torch.sparse_coo_tensor(indices=torch.stack([y_true, y_pred], dim=0), values=torch.ones_like(y_pred))
    else:
        cm = torch.sparse_coo_tensor(indices=torch.stack([y_true, y_pred], dim=0), values=torch.ones_like(y_pred),
                                     size=size)

    return cm.to_dense()[1:, 1:] if cm.size(0) > 2 else cm.to_dense()


def th_overall_accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    return (y_true.int() == y_pred.int()).sum().float() / float(y_true.numel())


def th_average_accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None, return_accuracys=False):
    cm_th = th_confusion_matrix(y_true, y_pred, num_classes)
    cm_th = cm_th.float()
    aas = torch.diag(cm_th / (cm_th.sum(dim=1)[None, :] + 1e-6))
    if not return_accuracys:
        return aas.mean()
    else:
        return aas.mean(), aas


def th_cohen_kappa_score(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None):
    cm_th = th_confusion_matrix(y_true, y_pred, num_classes)
    cm_th = cm_th.float()
    n_classes = cm_th.size(0)
    sum0 = cm_th.sum(dim=0)
    sum1 = cm_th.sum(dim=1)
    expected = torch.ger(sum0, sum1) / torch.sum(sum0)
    w_mat = torch.ones([n_classes, n_classes], dtype=torch.float32)
    w_mat.view(-1)[:: n_classes + 1] = 0.
    k = torch.sum(w_mat * cm_th) / torch.sum(w_mat * expected)
    return 1. - k


def th_intersection_over_union_per_class(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None):
    cm_th = th_confusion_matrix(y_true, y_pred, num_classes)
    sum_over_row = cm_th.sum(dim=0)
    sum_over_col = cm_th.sum(dim=0)
    diag = cm_th.diag()
    denominator = sum_over_row + sum_over_col - diag

    iou_per_class = diag / denominator
    return iou_per_class


def th_mean_intersection_over_union(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None):
    iou_per_class = th_intersection_over_union_per_class(y_true, y_pred, num_classes)
    return iou_per_class.mean()
