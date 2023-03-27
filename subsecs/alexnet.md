---
layout: default
---

[back](../index.md)

## AlexNet

[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)<br>
Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton <br>
Year: 2012 <br>

### Main contribution
**Rectified linear units (ReLU)**:
f(x) = max(0, x) <br>
* non-linearity
* sparsity (only weights and bias of positive inputs are updated)
* non-saturation (do not flatten out for large/small input values)
* faster computing

### Architecture
