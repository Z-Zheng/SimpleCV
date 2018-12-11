class Iterator(object):
    def __init__(self, data_loader):
        self._data_loader = data_loader
        self._iterator = iter(self._data_loader)

    def next(self, forward_times=1, call_backs=None):
        data_list = []
        for _ in range(forward_times):
            try:
                data = next(self._iterator)
            except StopIteration:
                self.reset()
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
