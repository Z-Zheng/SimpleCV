import torch
from PIL import Image
import numpy as np
import os
import torch.nn as nn


class VisualizeSegmm(nn.Module):
    def __init__(self, out_dir, palette):
        super(VisualizeSegmm, self).__init__()
        self.out_dir = out_dir
        self.palette = palette
        os.makedirs(self.out_dir, exist_ok=True)

    def forward(self, y_pred, filename):
        """

        Args:
            y_pred: 4-D array of shape [1, C, H, W]
            filename: str
        Returns:

        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1).astype(np.uint8)
        y_pred = y_pred.squeeze()
        color_y = Image.fromarray(y_pred)
        color_y.putpalette(self.palette)
        color_y.save(os.path.join(self.out_dir, filename))
