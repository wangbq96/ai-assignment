# 使用的参数
Epoch: 20

Batch Size: 64

Learning Rate: 0.01

Momentum: 0.9

# 如何选择设备（CPU/GPU）

## 对于PyTorch, MXNet, Caffe

分别修改代码中`device`部分的代码

## 对于Tensorflow

引用官方文档

> TensorFlow code, and tf.keras models will transparently run on a single GPU with no code changes required.

似乎没有办法手动修改，可以使用virtualenv来安装不同版本的Tensorflow来实现切换