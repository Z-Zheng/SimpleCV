## Usage
### 1. Define your model using CVModule and register it with one line code.
```python
from simplecv.interface.module import CVModule
from simplecv.util import registry

# register your model
@registry.MODEL.register('deeplabv3plus')
class Deeplabv3plus(CVModule):
    def __init__(self, config):
        super(Deeplabv3plus,self).__init__(config)
        # you can access your parameters by self.<param_name> as follows
        print(self.param1)
        # >> 1
        print(self.param2)
        # >> (2, )
        print(self.param3)
        # >> [3, ]
        
    def forward(self, *input):
        if self.training:
            # compute your loss functions and store them into a dict
            # The framework will automatically sum these losses when compute the gradients
            # **Note that the key of loss tensor have to suffix with "loss"**
            loss_dict={
                '1_loss': loss1,
                '2_loss': loss2,
                'other_information': other
            }
            return loss_dict
        else:
            return YOUR_MODEL_OUTPUT
        
    
    def set_defalut_config(self):
        # set the defalut value for your config
        self.config.update(dict(
            param1=1,
            param2=(2, ),
            param3=[3, ],
        ))
```
### 2. Define your dataloader and register it with one line code.
```python
from torch.utils.data import DataLoader
from simplecv import registry

@registry.DATALOADER.register('my_data_loader')
class CustomDataLoader(DataLoader):
    def __init__(self, config):
        # best practice
        # preset these items
        dataset = ...
        sampler = ...
        batch_sampler = ...
        
        super(CustomDataLoader, self).__init__(...)
       
    ...
```
### 3. Learning rate schedule and Optimizer 
```python
# support all naive optimizers and learning rate schedules in Pytorch
import torch.optim
from simplecv.util import registry

registry.OPT.register('sgd', torch.optim.SGD)
registry.OPT.register('adam', torch.optim.Adam)
```

### 4. Edit your Config file(*.py) using python
```python
config=dict(
    model=dict(),
    data=dict(
        train=dict(),
        test=dict(),
    ),
    ...
)
```
Refer to [retinanet_R_50_FPN_1x.py](https://github.com/Z-Zheng/simplecv/blob/master/config_demo/retinanet_R_50_FPN_1x.py)
to see more detail.
### 5. run your model on the fly
we provide two preset python script 
[dp_train.py](https://github.com/Z-Zheng/simplecv/blob/master/simplecv/dp_train.py)
and
[ddp_train.py](https://github.com/Z-Zheng/simplecv/blob/master/simplecv/ddp_train.py)
for training
#### Mode 1: use torch.nn.DataParallel
```bash
config_path=PATH_TO_YOUR_CONFIG_FILE
model_dir=PATH_TO_STORE_LOG
PATH_TO_SIMPLECV = ...
python ${PATH_TO_SIMPLECV}/dp_train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir}
```

If you want custom train script such as overriding a evaluate function and add arguments, 
you can do as follows:
```python
from simplecv import dp_train as train


def evaluate_fn(self, test_dataloader):
    pass

def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_fn)

if __name__ == '__main__':
    train.parser.add_argument('--example', default=None, type=str,
                    help='example')
    
    args = train.parser.parse_args()
    train.run(config_path=args.config_path,
              model_dir=args.model_dir,
              cpu_mode=args.cpu,
              after_construct_launcher_callbacks=[register_evaluate_fn])
```
And run your custom train script
```bash
config_path=PATH_TO_YOUR_CONFIG_FILE
model_dir=PATH_TO_STORE_LOG
python custom_train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    --example="example"

```

#### Mode 2: use torch.nn.parallel.DistributedDataParallel
```bash
export NUM_GPUS=2
config_path=PATH_TO_YOUR_CONFIG_FILE
model_dir=PATH_TO_STORE_LOG
PATH_TO_SIMPLECV = ...
python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${PATH_TO_SIMPLECV}/ddp_train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir}
```

#### Mode 3: use apex.parallel.DistributedDataParallel
This mode has been supported.
The instructions will be released.
```bash
todo
```