---
layout: post
title: "论文笔记：Inception Architecture for Computer Vision"
categories: [blog ]
tags: [CV算法]
description: Inception是在NIN中提出的一个神经网络模块，后来随着googLeNet中的成功被人们视为深度网络的法宝。
---
{:toc}

## 目的
2014年之后，深度CNN网络成为主流，其中出现了Inception之后，将神经网络由十几层加深到34层[^2], Inception作为后来深度神经网络中的重要组成模块，有对其中的原理和效果进行钻研学习一下。

## 论文
[Network in Network]<https://arxiv.org/abs/1312.4400> 

[Going Deeper with Convolutions]<https://arxiv.org/abs/1409.4842>

[Inception v3]<https://www.arxiv.org/abs/1512.00567>

[Inception v4]<https://arxiv.org/abs/1602.07261>

## Network In Network提出原因[^1]
### 提出原因

Generalized Linear Model使用的前提是假设语义空间是线性可分的。但是往往并不是假设的那样子，来自同一个概念的数据信息往往是非线性的，从而表示这些信息要使用输入参数X的非线性关系函数。

### 结构

![@MLPConv and Linear Conv](https://cwlseu.github.io/images/inception/NINBlock.jpg)
通过堆叠的MLPConv的方式实现了NIN的设计，最后的预测层使用Global Average Pooling替代全连接层。为什么呢？因为全连接层容易出现Overfitting。对最后的每一个特征层进行average pooling计算，对pooling后的向量直接作为softmax的输入。其中最后的输出特征层可以解释为每个类别的confidence map；同时，average pooling没有参数进行优化；而且average pooling的方式利用的是全局信息，对于空间信息更加robust

![@NIN](https://cwlseu.github.io/images/inception/NIN.jpg)

展示最后的feature maps 结果信息：
![@Visualization NIN](https://cwlseu.github.io/images/inception/VisualizationNIN.jpg)

## 从Inception设计遵循规则[^4]

#### 避免特征表示瓶颈

Avoid representational bottlenecks, especially early in the network. Feed-forward networks can be represented by an acyclic graph from the input layer(s) to the classifier or regressor. This defines a clear direction
for the information flow. For any cut separating the inputs from the outputs, one can access the amount of information passing though the cut. One should avoid bottlenecks with extreme compression. In general the representation size should gently decrease from the inputs to the outputs before reaching the final representation used for the task at hand. Theoretically, information content can not be assessed merely by the dimensionality of the representation as it discards important factors like correlation structure; the dimensionality merely provides a rough estimate of information content.

![@](https://cwlseu.github.io/images/inception/9.PNG)

#### 高纬度更容易处理局部

Higher dimensional representations are easier to process locally within a network. Increasing the activations per tile in a convolutional network allows for more disentangled features. The resulting networks will train faster.
![@](https://cwlseu.github.io/images/inception/7.PNG)

#### 通过低维嵌入的方式实现空间信息的聚合，能够减少特征表示的损失

Spatial aggregation can be done over lower dimensional embeddings without much or any loss in representational power. For example, before performing a more spread out (e.g. 3 × 3) convolution, one can reduce the dimension of the input representation before the spatial aggregation without expecting serious adverse effects. We hypothesize that the reason for that is the strong correlation between adjacent unit results in much less loss of information during dimension reduction, if the outputs are used in a spatial aggregation context. Given that these signals should be easily compressible, the dimension reduction even promotes faster learning.
![@](https://cwlseu.github.io/images/inception/5.PNG)
![@](https://cwlseu.github.io/images/inception/6.PNG)

#### 网络的宽度和深度的平衡

Balance the width and depth of the network. Optimal performance of the network can be reached by balancing the number of filters per stage and the depth of the network. Increasing both the width and the depth of the network can contribute to higher quality networks.
However, the optimal improvement for a constant amount of computation can be reached if both are increased in parallel. The computational budget should therefore be distributed in a balanced way between the depth and width of the network.



## GoogLeNet中的应用[^2]

## 参考文献
[^1]: [Network in Network]<https://arxiv.org/abs/1312.4400>

[^2]: [Going Deeper with Convolutions]<https://arxiv.org/abs/1409.4842>

[^3]: [Inception v3]<https://www.arxiv.org/abs/1512.00567>

[^4]: [Inception v4]<https://arxiv.org/abs/1602.07261>
