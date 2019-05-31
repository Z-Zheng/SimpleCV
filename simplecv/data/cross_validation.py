import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset
import math
from functools import reduce
from simplecv.data.distributed import StepDistributedRandomSubsetSampler


class CrossValSamplerGenerator(object):
    def __init__(self,
                 dataset: Dataset,
                 distributed=True,
                 seed=2333):
        """

        Args:
            dataset: a instance of torch.utils.data.dataset.Dataset
            distributed: whether to use distributed random sampler
            seed: random seed for torch.randperm

        Example::
            >>> CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            >>> sampler_pairs = CV.k_fold(5) # 5-fold CV
            >>> train_sampler, val_sampler = sampler_pairs[0] # 0-th as val, 1, 2, 3, 4 as train
        """
        self.num_samples = len(dataset)
        self.seed = seed
        self.distributed = distributed

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

            if self.distributed:
                sampler_pairs.append((StepDistributedRandomSubsetSampler(train_indices),
                                      StepDistributedRandomSubsetSampler(val_indices)))
            else:
                sampler_pairs.append((SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)))

        return sampler_pairs
