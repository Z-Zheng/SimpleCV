import torch
from simplecv.core.transform_base import TestTransform


class DistPredictor(object):
    def __init__(self):
        self._model_list = list()
        self._trans_list = list()
        self._reduce_op = None

        self._callback_after_transforms = list()
        self._callback_after_forward = list()
        self._callback_after_reduce = list()

    def add_model(self, model):
        self._model_list.append(model)

    def add_transform(self, transform: TestTransform):
        if not isinstance(transform, TestTransform):
            raise ValueError('transform must inherit from simplecv.core.transform_base.TestTransfrom')

        self._trans_list.append(transform)

    def register_callback_after_transforms(self, callback):
        self._callback_after_transforms.append(callback)

    def register_callback_after_forward(self, callback):
        self._callback_after_forward.append(callback)

    def register_callback_after_reduce(self, callback):
        self._callback_after_reduce.append(callback)

    def register_reduce_op(self, reduce_op):
        self._reduce_op = reduce_op

    @property
    def models(self):
        return self._model_list

    def __call__(self, image, **kwargs):
        image_list = self.transforms(image, **kwargs)
        for func in self._callback_after_transforms:
            func(image_list)

        out_list = self._forward(image_list, **kwargs)
        out_list = self.inv_transforms(out_list, **kwargs)
        for func in self._callback_after_forward:
            func(out_list)

        out = self.reduce(out_list, **kwargs)
        for func in self._callback_after_reduce:
            func(out)
        return out

    def _forward(self, image_list, **kwargs):
        out_list = list()
        for image in image_list:
            for model in self._model_list:
                # todo: use multi-gpu
                out = model(image)
                out_list.append(out)
        return out_list

    def transforms(self, image, **kwargs):
        return [trans_op.transform(image) for trans_op in self._trans_list]

    def inv_transforms(self, image_list, **kwargs):
        assert len(image_list) == len(self._trans_list)
        return [trans_op.inv_transform(image) for image, trans_op in zip(image_list, self._trans_list)]

    def reduce(self, data_list, **kwargs):
        if self._reduce_op:
            return self._reduce_op(data_list, **kwargs)
        return self.default_reduce_op(data_list, **kwargs)

    def default_reduce_op(self, data_list, **kwargs):
        return torch.mean(torch.cat(data_list, dim=0), dim=0, keepdim=True)
