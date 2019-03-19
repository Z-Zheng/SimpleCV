from torch.utils.data.distributed import DistributedSampler


class Iterator(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader
        self._iterator = iter(self._data_loader)

    def next(self, forward_times=1, call_backs=None, is_master=True):
        data_list = []
        for _ in range(forward_times):
            try:
                data = next(self._iterator)
            except StopIteration:
                self.reset()
                if is_master:
                    if call_backs is not None:
                        for f in call_backs:
                            f()
                data = next(self._iterator)

            data_list.append(data)
        return data_list

    def reset(self):
        self._iterator = iter(self._data_loader)

    def iter(self, forward_times=1):
        """ a droplast iterator
        
        Args:
            forward_times: int

        Returns:
        
        """
        data_list = []
        for data in self._iterator:
            data_list.append(data)
            if len(data_list) == forward_times:
                yield data_list
                data_list = []

    def set_seed_for_dist_sampler(self, seed):
        if not isinstance(self._data_loader.sampler, DistributedSampler):
            return

        if self._data_loader.sampler is not None:
            if hasattr(self._data_loader.sampler, 'set_step'):
                self._data_loader.sampler.set_step(seed)
            elif hasattr(self._data_loader.sampler, 'set_epoch'):
                self._data_loader.sampler.set_epoch(seed)
