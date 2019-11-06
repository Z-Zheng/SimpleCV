## CVModule Usage

### 1. Define Main Model
```python
from simplecv.interface import CVModule
from simplecv import registry

@registry.MODEL.register('model name you love')
class Model(CVModule):
    def __init__(self, config):
        super(Model, self).__init__(config)
        # todo
        
    def forward(self, x, y=None):
        # todo
        pass
    
    def set_default_config(self):
        self.config.update(dict(
            # todo
        ))
```

### 2. Override `forward` function
```python
class Model(CVModule):
    ...
    
    def forward(self, x, y=None):
        def pseudo_loss(ytrue, ypred):
            pass
            
        ypred = ...
        if self.training:
            ytrue = ...
            
            mem_usage=  ...
            loss_dict=dict(
                xxx_loss=pseudo_loss(ytrue, ypred),
                ohter_info = mem_usage
            )
            # during training, console will log 'xxx_loss = {value}, other_info = {mem_usage}'
        else:
            # todo: eval mode
            pass
```

### 3. Override `set_default_config` function

This aims to configure your default hyperparameters of your model.

### Q&A
#### Q: 1. how to load weights from file
#### A: You can set `weight` item to the preset `GLOBAL` attribute in CVModule, such as
```python
from simplecv.interface import CVModule
# case 1: define a default `weight`
class Model(CVModule):
    def __init__(self,config):
        super(Model,self).__init__(config)
        
    def set_default_config(self):
        self.config.update(dict(
            GLOBAL=dict(
                weight=dict(
                    path='<path to weights file (*.pth)>. default None',
                    excepts='a regular expression, which matches parameters you want to drop. default None'
                )
            )
        ))
        
# case 2: dynamic initialization in anywhere
model = Model(dict())
model.config.GLOBAL = dict(weight=dict(
    path=None,
    excepts=None,
))
model.init_from_weightfile()

# case 3: configure it in your config file
config = dict(
    model=dict(
        type='FasterRCNN',
        params=dict(
            GLOBAL=dict(
                weight=dict(
                    path=None,
                    excepts=None,
                )
            )
        )
    )
)

```