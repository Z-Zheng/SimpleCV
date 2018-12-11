# SimpleCV
This repo is aimed to simplify training, evaluation and prediction in Pytorch.
## Features
1. High performance parallel training using Pytorch 1.0
2. Dynamic module registration mechanism makes you customize components on the fly
3. Support tensorboard

## Installation
TODO
## Usage
### 1. Define your model using CVModule and register it with one line code.
```python
from interface.module import CVModule
from util import registry

# register your model
registry.MODEL.register('deeplabv3plus')
class Deeplabv3plus(CVModule):
    def __init__(self, config):
        super(Deeplabv3plus,self).__init__(config)
    
    def forward(self, *input):
        ...
```
### 2. Define your dataloader and register it with one line code.
```python
from torch.utils.data import DataLoader
from util import registry
registry.DATALOADER.register('my_data_loader')
class CustomDataLoader(DataLoader):
    def __init__(self,...):
        super(CustomDataLoader, self).__init__(...)
       
    ...
```
### 3. Learning rate schedule and Optimizer 
```python
# support all naive optimizers and learning rate schedules in Pytorch
import torch.optim
from util import registry

registry.OPT.register('sgd', torch.optim.SGD)
registry.OPT.register('adam', torch.optim.Adam)
```

### 4. Edit your Config file(*.py) using python
```python
config=dict(
    ...
)
```
### 5. run your model on the fly
```bash
export NUM_GPUS=2
config_path=PATH_TO_YOUR_CONFIG_FILE
model_dir=PATH_TO_STORE_LOG
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir}
```

### TODO

- [ ] add complete demos of segmentation, detecion and classification