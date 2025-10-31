也可以在线加载


```
import deeplake
ds = deeplake.load('hub://activeloop/pascal-voc-2007-test')

dataloader = ds.pytorch(num_workers=0, batch_size=4, shuffle=False)
```