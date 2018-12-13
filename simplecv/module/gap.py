import torch
import torch.nn as nn



class GlobalAvgPool2D(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2D, self).__init__()

    def forward(self, x):
        x_pool = torch.mean(x.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)

        x_pool = x_pool.view(x.size(0), x.size(1), 1, 1).contiguous()
        return x_pool
