---
layout: default
---

## EfficientNet

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](http://proceedings.mlr.press/v97/tan19a.html)<br>
Aurthors: Mingxing Tan, Quoc Le <br>
Year: 2019 <br>

### Architecture
![efficientnet](../pics/The-architecture-of-EfficientNet-Block.png)<br>

EfficientNet is built using a series of building blocks called `MBConvBlocks`.

The name `MBConvBlocks` comes from [**MobileNet**](./mobilenet.md).

`MBConvBlocks` consists of `DepthwiseConvBlock`, `PointwiseConvBlock`, and `SEBlock`.

#### DepthwiseConvBlock

`DepthwiseConvBlock`: A depthwise convolution is applied to the input tensor. This is a type of convolution that applies a separate filter to each channel of the input tensor without mixing the channels. This allows the network to extract spatial information independently from different input channels. This helps reduce the number of parameters in the model and also helps capture more diverse features.

*   `DepthwiseConvBlock` has the same number of input and output channels.
*   `DepthwiseConvBlock` has kernel size 3x3.

#### PointwiseConvBlock

`PointwiseConvBlock`: A pointwise convolution is applied to the output of the depthwise convolution. This is a 1x1 convolution that applies a filter to each location in the output tensor. It is used to increase or decrease the number of channels in the output tensor. This is a convolutional operation that uses a 1x1 filter to combine the output channels of the depthwise convolution block. The pointwise convolution helps to increase the expressiveness of the network by allowing it to learn more complex representations of the input data. The pointwise convolution operation applies a C_out filters of size 1x1 to the input feature map, which essentially performs a linear combination of the output channels of the previous convolutional layer.

*   `PointwiseConvBlock` has different number of input and output channels.
*   `PointwiseConvBlock` has kernel size 1x1.

#### SEBlock

`SEBlock`: Squeeze-and-Excitation (SE) Block. The SE block is used to model the interdependencies between channels. It contains a global average pooling layer followed by two fully connected layers and a sigmoid activation function. The output of the sigmoid function is used to re-weight the input feature map before passing it to the next layer. These weights are used to scale the channels of the input tensor to emphasize the most important channels.


