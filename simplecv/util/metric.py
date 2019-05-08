import torch


def th_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None):
    """

    Args:
        y_true: 1-D tensor of shape [n_samples]
        y_pred: 1-D tensor of shape [n_samples]
        num_classes: scalar
    Returns:

    """
    size = [num_classes + 1, num_classes + 1] if num_classes is not None else None

    if size is None:
        cm = torch.sparse_coo_tensor(indices=torch.stack([y_true, y_pred], dim=0), values=torch.ones_like(y_pred))
    else:
        cm = torch.sparse_coo_tensor(indices=torch.stack([y_true, y_pred], dim=0), values=torch.ones_like(y_pred),
                                     size=size)
    return cm.to_dense()[1:, 1:]


def th_overall_accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    return (y_true == y_pred).sum().float() / y_true.numel()


def th_average_accuracy_score(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None):
    cm_th = th_confusion_matrix(y_true, y_pred, num_classes)
    cm_th = cm_th.float()
    return torch.diag(cm_th / cm_th.sum(dim=0)[None, :]).mean()


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