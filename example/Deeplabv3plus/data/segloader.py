from torch.utils.data.dataloader import DataLoader, default_collate
from data import seg_data
from simplecv import registry


@registry.DATALOADER.register('segdataloader')
class SegDataLoader(DataLoader):
    def __init__(self,
                 config):
        self.config = {}
        self.set_defalut_config()
        self.config.update(config)

        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = seg_data.SegData(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            image_format=self.image_format,
            mask_format=self.mask_format,
            training=self.training,
            filenameList_path=self.filenameList_path
        )
        super(SegDataLoader, self).__init__(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=None,
            batch_sampler=None,
            num_workers=self.num_workers,
            collate_fn=default_collate,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
            worker_init_fn=None,
        )

    def set_defalut_config(self):
        self.config.update(dict(
            image_dir='',
            mask_dir='',
            image_format='jpg',
            mask_format='png',
            training=True,
            filenameList_path=None,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            timeout=0,
        ))
