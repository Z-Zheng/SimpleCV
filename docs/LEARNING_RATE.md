## Learning rate config

### 1. mutiple step learning rate
```python
learning_rate=dict(
    type='multistep',
    params=dict(
        base_lr=0.02,
        steps=(60000, 80000),
        gamma=0.1,
        warmup_step=None,
        warmup_init_lr=0.02),
)
```
### 2. polynomial learning rate
```python
learning_rate=dict(
    type='poly',
    params=dict(
        base_lr=0.007,
        power=0.9,
        max_iters=30000),
),
```