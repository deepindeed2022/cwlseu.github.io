---
layout: post
comments: true
title: "神经网络模型训练提升成果总结"
categories: [blog ]
tags: [detection]
description: 物体检测算法概览
---

* content
{:toc}

## 问题: foreground-background class imbalance

### OHEM：Training Region-based Object Detectors with Online Hard Example Mining
- intro: CVPR 2016 Oral. Online hard example mining (OHEM)
- arxiv: http://arxiv.org/abs/1604.03540
- paper: http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf
- github（Official）: https://github.com/abhi2610/ohem
- author page: http://abhinav-shrivastava.info/
paper里直接说的最初的思想就是想把Bootstrapping移植到dl当中来，所以产生了ohem。由于bootstrapping本身是迭代算法，直接应用到dl当中，会使dl的训练变得非常慢。为此，作者借用了其核心思想--让hard example起更大的作用，提出了ohem。
具体的方法就是先让所有样本通过dl计算出loss,以loss大为hard example的标准，从而选出hard example来，然后再组成一组batch,进行训练。
![@OHEM设计架构](https://cwlseu.github.io/images/detection/OHEM.png)

### Focal Loss for Dense Object Detection

* 论文地址: https://arxiv.org/abs/1708.02002
* 主要观点：
提出Single stage detector不好的原因完全在于：
    - 极度不平衡的正负样本比例: anchor近似于sliding window的方式会使正负样本接近1000：1，而且绝大部分负样本都是easy example，这就导致下面一个问题：
    - gradient被easy example dominant的问题：往往这些easy example虽然loss很低，但由于数量众多，对于loss依旧有很大贡献，从而导致收敛到不够好的一个结果。所以作者的解决方案也很直接：直接按照loss decay掉那些easy example的权重，这样使训练更加bias到更有意义的样本中去。
![@Focal Loss v.s. Cross Enctroy Loss train performance](https://cwlseu.github.io/images/detection/FocalLoss.png)

### Gradient Harmonized Single-stage Detector

- 论文地址：https://arxiv.org/pdf/1811.05181.pdf
- 作者：Buyu Li, Yu Liu and Xiaogang Wang
- 针对的的痛点：Huge difference in quantity between positive and negative examples as well as between easy and hard examples.
- Abstract
GHM can be easily embedded into both classification loss function like cross-entropy (CE) and regression
loss function like smooth-L1 (SL1) loss.two novel loss functions called GHM-C and GHM-R are designed
to balancing the gradient flow for anchor classification and bounding box refinement

### A-Fast-RCNN: Hard positive generation via adversary for object detection
从更好的利用数据的角度出发，OHEM和S-OHEM都是发现困难样本，而A-Fast-RCNN的方法则是通过GAN的方式在特征空间产生具有部分遮挡和形变的困难样本

## 其他参考链接

- [熟睡的孩子关于GHM文章的解读，来自知乎](https://zhuanlan.zhihu.com/p/50217821)
- [为什么说ohem是bootstrapping在dl中的应用呢？](https://www.zhihu.com/question/56092850/answer/216461322)
- [OHEM论文解读](https://zhuanlan.zhihu.com/p/58162337)

## 问题：训练时间长

### 初始化

### 学习率

### 网络节点输入值正则化

神经网络训练时，每一层的输入分布都在变化。不论输入值大还是小，我们的学习率都是相同的，这显然是很浪费效率的。而且当输入值很小时，为了保证对它的精细调整，学习率不能设置太大。那有没有办法让输入值标准化得落到某一个范围内，比如[0, 1]之间呢，这样我们就再也不必为太小的输入值而发愁了。

办法当然是有的，那就是正则化！由于我们学习的是输入的特征分布，而不是它的绝对值，故可以对每一个mini-batch数据内部进行标准化，使他们规范化到[0, 1]内。这就是Batch Normalization，简称BN。由大名鼎鼎的inception V2提出。它在每个卷积层后，使用一个BN层，从而使得学习率可以设定为一个较大的值。使用了BN的inceptionV2，只需要以前的1/14的迭代次数就可以达到之前的准确率，大大加快了收敛速度。

### 优化结构

训练速度慢，归根结底还是网络结构的参数量过多导致的。减少参数量，可以大大加快收敛速度。采用先进的网络结构，可以用更少的参数量达到更高的精度。如inceptionV1参数量仅仅为500万，是AlexNet的1/12, 但top-5准确率却提高了一倍多。如何使用较少的参数量达到更高的精度，一直是神经网络结构研究中的难点。目前大致有如下几种方式

- 使用小卷积核来代替大卷积核。VGGNet全部使用3x3的小卷积核，来代替AlexNet中11x11和5x5等大卷积核。小卷积核虽然参数量较少，但也会带来特征面积捕获过小的问题。inception net认为越往后的卷积层，应该捕获更多更高阶的抽象特征。因此它在靠后的卷积层中使用的5x5等大面积的卷积核的比率较高，而在前面几层卷积中，更多使用的是1x1和3x3的卷积核。
- 使用两个串联小卷积核来代替一个大卷积核。inceptionV2中创造性的提出了两个3x3的卷积核代替一个5x5的卷积核。在效果相同的情况下，参数量仅为原先的3x3x2 / 5x5 = 18/25
- 1x1卷积核的使用。1x1的卷积核可以说是性价比最高的卷积了，没有之一。它在参数量为1的情况下，同样能够提供线性变换，relu激活，输入输出channel变换等功能。VGGNet创造性的提出了1x1的卷积核
非对称卷积核的使用。inceptionV3中将一个7x7的卷积拆分成了一个1x7和一个7x1, 卷积效果相同的情况下，大大减少了参数量，同时还提高了卷积的多样性。
- depthwise卷积的使用。mobileNet中将一个3x3的卷积拆分成了串联的一个3x3 depthwise卷积和一个1x1正常卷积。对于输入channel为M，输出为N的卷积，正常情况下，每个输出channel均需要M个卷积核对输入的每个channel进行卷积，并叠加。也就是需要MxN个卷积核。而在depthwise卷积中，输出channel和输入相同，每个输入channel仅需要一个卷积核。而将channel变换的工作交给了1x1的卷积。这个方法在参数量减少到之前1/9的情况下，精度仍然能达到80%。
- 全局平均池化代替全连接层。这个才是大杀器！AlexNet和VGGNet中，全连接层几乎占据了90%的参数量。inceptionV1创造性的使用全局平均池化来代替最后的全连接层，使得其在网络结构更深的情况下（22层，AlexNet仅8层），参数量只有500万，仅为AlexNet的1/12
网络结构的推陈出新，先进设计思想的不断提出，使得减少参数量的同时提高准确度变为了现实。

## 问题：梯度弥散
https://yq.aliyun.com/articles/598429

## 问题：过拟合
### 什么是过拟合
过拟合就是训练模型的过程中，模型过度拟合训练数据，而不能很好的泛化到测试数据集上。出现over-fitting的原因是多方面的：

- 训练数据过少，数据量与数据噪声是成反比的，少量数据导致噪声很大
- 特征数目过多导致模型过于复杂

### 如何避免过拟合

- 控制特征的数目，可以通过特征组合，或者模型选择算法
- Regularization，保持所有特征，但是减小每个特征的参数向量$\theta$的大小，使其对分类y所做的贡献很小

### Lasso & 岭回归

http://www.cnblogs.com/ooon/p/4964441.html

## 问题： 模型大小与功耗
https://zhuanlan.zhihu.com/p/48269250

