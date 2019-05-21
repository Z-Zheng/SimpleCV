import torch
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, RandomSampler
from torch.utils.data.dataset import Dataset
import math
from functools import reduce


class CrossValSamplerGenerator(object):
    def __init__(self, dataset: Dataset, seed=2333):
        self.num_samples = len(dataset)
        self.seed = seed

    def k_fold(self, k=5):
        g = torch.Generator()
        g.manual_seed(self.seed)

        indices = torch.randperm(self.num_samples, generator=g).tolist()
        total_size = int(math.ceil(len(indices) / k) * k)
        indices += indices[:(total_size - len(indices))]

        assert len(indices) == total_size

        # subsample
        sampler_pairs = []
        k_fold_indices = [indices[i:total_size:k] for i in range(k)]
        for i in range(k):
            cp = k_fold_indices.copy()
            val_indices = cp.pop(i)
            train_indices = reduce(lambda a, b: a + b, cp)
            assert len(val_indices) + len(train_indices) == total_size
            sampler_pairs.append((SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)))

        return sampler_pairs
