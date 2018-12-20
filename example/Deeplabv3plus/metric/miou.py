import torch
import numpy as np


def miou_fn(self, test_dataloder):
    self._model.eval()

    for image, y_true in test_dataloder:
        y_pred = self._model(image)

        pass
