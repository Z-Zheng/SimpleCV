import simplecv._impl.metric.function as mF


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, y_true, y_pred, to_dense=True):
        return mF.th_confusion_matrix(y_true, y_pred, self.num_classes, to_dense=to_dense)

    def summary(self):
        # todo (visualize)
        return NotImplementedError
