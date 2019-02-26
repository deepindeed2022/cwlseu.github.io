---
layout: post
title: "神经网络压缩之二值化"
categories: [blog ]
tags: [深度学习, 神经网络压缩]
description: "神经网络量化、二值化、设计package的神经网络模块"
---
* content
{:toc}

## 引言
![](https://cwlseu.github.io/images/bnn/53225027.png)
面对神经网络over-parametrization和enormous parameters, 工作者在这者方面下了不少功夫。我们也想在这方面分的一杯羹, 当前主要的方向有：
- shallow network
- compressing pre-trained network
- designed compact layers：可以去看看[PVANet](http://cwlseu.github.io/PVANet)
- quantizing parameters
- network binarization

关于神经网络二值化最早的一篇应该是是Bengio组的Binaryconnect模型，这个模型用符号函数把weight二值化了，从而把前向传播中大部分的单精度乘法换成了单精度加法。他们组之后的一篇binarynet进而把activation也二值化了，那么单精度加法进一步变成了xnor位运算操作。
> 二值网络最大的贡献主要在于三点:
1. 尽管模型精度降低了非常多，但是在训练效果却不比全精度的网络差，有的时候二值网络的训练效果甚至会超越全精度网络，因为二值化过程给神经网络带来了noise，像dropout一样，反而是一种regularization，可以部分避免网络的overfitting。
2. 二值化网络可以把单精度乘法变成位操作，这大大地减少了训练过程中的运算复杂度。这种位运算可以写成gpu kernel, 或者用fpga实现，会给神经网络训练速度带来提升。
3. 存储神经网络模型主要是存储weights. 二值化的weight只要一个bit就可以存下来了，相比之前的32bit，模型减小了32倍，那么把训练好的模型放在移动设备，比如手机上面做测试就比较容易了。当前还有一些二值化网络的变种，比如给二值加一个系数(xnor net)来更好地逼近全值网络。比如通过离散化梯度把后向传播中的乘法也变成加法。因为训练速度的提高和存储空间的减少，二值化网络的发展将会让深度神经网络在更多计算能力和存储空间相对比较弱的平台上得到作用，比如手机，嵌入式系统等。

## 二值网络

## XNOR

## Binary-Weight-Network


## CNNC：Quantized Convolutional Neural Networks for Mobile Devices
### 论文作者信息
- [Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, Jian Cheng](National Laboratory of Patter Recognition Institute of Automation, Chinese Academy of Sciences)

### 简介
目标是减少模型大小，同时提高计算速度。通过对卷积层的filter kernal和全连接层的权重矩阵同时进行量化，最小化每一层的错误估值进行训练。

### 问题
之前的工作很少有能够同时实现整个网络的显著的加速和压缩。

### 主要贡献
1. 加速和压缩于一体的神经网络
2. 提出了有效的训练方式来减少训练过程中的累积残差
3. 实现4~6倍的加速，同时15~20倍的压缩比。

## 二值神经网络相关文献
1. [Mohammad Rastegari Vicente Ordonez Joseph Redmon Ali Farhadi: XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks (2016)](https://arxiv.org/abs/1603.05279)
2. [Courbariaux, M., Bengio, Y., David, J.P.: Training deep neural networks with low precision multiplications. arXiv preprint arXiv:1412.7024 (2014) 4](https://arxiv.org/abs/1412.7024)
3. [Soudry, D., Hubara, I., Meir, R.: Expectation backpropagation: parameter-free training of multilayer neural networks with continuous or discrete weights. In: Advances in Neural Information Processing Systems. (2014) 963–971 4](http://papers.nips.cc/paper/5269-expectation-backpropagation-parameter-free-training-of-multilayer-neural-networks-with-continuous-or-discrete-weights.pdf)
4. [Esser, S.K., Appuswamy, R., Merolla, P., Arthur, J.V., Modha, D.S.: Backpropagation for energy-efficient neuromorphic computing. In: Advances in Neural Information Processing Systems. (2015) 1117–1125 4](https://papers.nips.cc/paper/5862-backpropagation-for-energy-efficient-neuromorphic-computing) 
5. [Courbariaux, M., Bengio, Y., David, J.P.: Binaryconnect: Training deep neural networks with binary weights during propagations. In: Advances in Neural Information Processing Systems. (2015) 3105–3113 4, 6, 10, 11](https://www.arxiv.org/abs/1511.00363) 
6. [Baldassi, C., Ingrosso, A., Lucibello, C., Saglietti, L., Zecchina, R.: Subdominant dense clusters allow for simple learning and high computational performance in neural networks with discrete synapses. Physical review letters 115(12) (2015) 128101 5](https://arxiv.org/abs/1509.05753v1) 
7. [Kim, M., Smaragdis, P.: Bitwise neural networks. arXiv preprint arXiv:1601.06071 (2016)](https://arxiv.org/abs/1601.06071)
8. [Hubara I, Soudry D, Yaniv R E. Binarized Neural Networks[J]. arXiv preprint arXiv:1602.02505, 2016.](https://arxiv.org/abs/1602.02505)
代码链接：https://github.com/MatthieuCourbariaux/BinaryNet
9. [](https://blog.csdn.net/stdcoutzyx/article/details/50926174)

## 三值神经网络
1. [Can FPGAs Beat GPUs in Accelerating Next-Generation Deep Neural Networks](http://jaewoong.org/pubs/fpga17-next-generation-dnns.pdf)
2. [Ternary Residual Networks](http://arxiv.org/pdf/1707.04679)