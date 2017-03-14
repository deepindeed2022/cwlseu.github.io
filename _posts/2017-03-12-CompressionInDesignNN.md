---
layout: post
title: PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection
categories: [blog ]
tags: [CV, 深度学习]
description: 
---


## Abstract
1. 使用"Feature Extraction+Region Proposal+RoI Classification" 的结构，主要对Feature Extraction进行重新设计。因为，Region Proposal部分计算量不太大而且classification部分可以使用通用的技术(例如：Truncated SVD) 进行有效的压缩。
2. 设计原则：Less channels with more layers 和采用一些Building blocks （包括：串级的ReLU、Inception和HyperNet)

  结果 
  VOC2007—83.8\%mAP；VOC2012—82.5\%mAP，46ms/image在NVIDIA Titan X GPU；计算量是ResNet-101的12.3\% (理论上)

## Introduction

 准确率很高的检测算法有往往需要很大的计算量。现在压缩和量化技术的发展对减小网络的计算量很重要。这篇文章展示了我们用于目标检测的一个轻量级的特征提取的网络结构——PVANET

串级的ReLU(C.ReLU—Concatenated rectified linear unit)被用在我们的CNNs 的初期阶段来减少一半的计算数量而不损失精度。
Inception\cite{GoogLeNet}被用在剩下的生成feature的子网络中。一个Inception module 产生不同大小的感受野（receptive fields）的输出激活值，所以增加前一层感受野大小的变化。我们观察叠加的Inception modules可以比线性链式的CNNs更有效的捕捉大范围的大小变化的目标。
采用multi-scale representation的思想\cite{HyperNet}, 结合多个中间的输出，所以，这使得可以同时考虑多个level的细节和非线性。我们展示设计的网络deep and thin，在batch normalization、residual connections和基于plateau detection的learning rate的调整的帮助下进行有效地训练

##  Review Related Work

![@HyperNet的网络结构示意图,其中本文中主要利用其中对不同层的卷积特征的联合利用的想法](../images/pvanet/img/HyperNet.jpg)

![@Inception的网络结构示意图,其中的1x1的卷积核主要作用是用于特征降维和感受野设置为1](../images/pvanet/img/Inception.jpg)


## Nerual Network Design
### C.ReLU

    C.ReLU来源于CNN中间激活模式引发的。观察发现，输出节点倾向于是"配对的"，一个节点激活是另一个节点的相反面。

![@C.ReLU的设计结构](../images/pvanet/img/CReLU.jpg)

* 求同
    C.ReLU减少一半输出通数量，通过简单的连接相同的输出和negation 使其变成双倍，这使得2倍的速度提升而没有损失精度
* 存异
    同时，增加了scaling and shifting在concatenation之后，这允许每个channel 的斜率和激活阈值与其相反的channel不同。


### Inception
Inception是捕获图像中小目标和大目标的最具有成效的Building Blocks之一;
为了学习捕获大目标的视觉模式，CNN特征应该对应于足够大的感受野，这可以很容易的通过叠加 3x3或者更大的核卷积实现;
为了捕获小尺寸的物体，输出特征应该对应于足够小的感受野来精确定位小的感兴趣区域。
![@(Left) Our Inception building block. 5x5 convolution is replaced with two 3x3 convolutional layers for efficiency. (Right) Inception for reducing feature map size by half](../images/pvanet/img/PVANET_Inception.jpg)

1x1的conv扮演了关键的角色，保留上一层的感受野。只是增加输入模式的非线性，它减慢了一些输出特征的感受野的增长，使得可以精确地捕获小尺寸的目标。

![@Inception中的感受野的直观表示](../images/pvanet/ReceptionField.jpg)

## 整个网络的结构
![@The detailed structure of PVANET](../images/pvanet/PVANETDetails.jpg)
从中可以看出，在conv3\_4, conv4\_4, conv5\_4的输出特征通过下采样和上采样技术实现相同的size之后进行级联作为最后的卷积特征。
![@Comparisons between our network and some state-of-the-arts in the PASCAL VOC2012 leaderboard.](../images/pvanet/Result.jpg)

## Summary

1. C.ReLU减少训练过程中的网络大小
2. Inception是网络设计中的用于压缩网络的技巧
3. 1x1的使用相当于挖掘卷积结果中的冗余信息，从而减少channel个数

## reference
[1]. [PVANet: Lightweight Deep Neural Networks for Real-time Object Detection](https://www.arxiv.org/pdf/1608.08021v3.pdf)
[code:https://github.com/sanghoon/pva-faster-rcnn](https://github.com/sanghoon/pva-faster-rcnn)
[2] [HyperNet: Towards Accurate Region Proposal Generation and Joint](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kong_HyperNet_Towards_Accurate_CVPR_2016_paper.pdf)
[3] [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842v1.pdf)

