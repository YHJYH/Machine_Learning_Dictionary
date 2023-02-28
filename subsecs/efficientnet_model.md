---
layout: default
---

Model: `EfficientNetB0` <br>
Data input: `{num_class: 1000, img_size: (3, 224, 224)}` <br>
Example dataset: [Caltech256](./caltech256.md) <br>
```py
===============================================================================================
Total params: 7,171,288
Trainable params: 7,171,288
Non-trainable params: 0
Total mult-adds (G): 17.67
===============================================================================================
Input size (MB): 0.57
Forward/backward pass size (MB): 1980.78
Params size (MB): 27.36
Estimated Total Size (MB): 2008.71
===============================================================================================
```

Model: `EfficientNetCustomize` <br>
Data input: `{num_class: 10, img_size: (3, 32, 32)}` <br>
Example dataset: [CIFAR10](./cifar10.md) <br>
```py
===============================================================================================
Total params: 2,117,626
Trainable params: 2,117,626
Non-trainable params: 0
Total mult-adds (M): 285.04
===============================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 33.62
Params size (MB): 8.08
Estimated Total Size (MB): 41.71
===============================================================================================
```