---
layout: post
title: 应用于自然场景文本检测与识别常用模块
tags: [计算机视觉] 
categories: [blog ]
notebook: 视觉算法
---

* content
{:toc}

## 序言
基于深度学习算法的的自然场景文本检测，经过几年的研究，针对解决实际问题中的某些问题，涌现出CTC, LSTM等大量的单元。本章节将针对模型设计中常用的
模块进行介绍。

## 问题
- 文本定位
- 序列不定长的问题
- 文字大小不一的问题

## HMM & CTC
针对问题：序列不定长的问题

## rlstm(Reverse LSTM)整体架构如下，其中需要用到Reverse这种Layer

![](../../images/ocr/shuffle_1.png)

## ChannelShuffle

![](../../images/ocr/shuffle_2.png)

一般的分组卷积(如ResNeXt的)仅对$3\times3$的层进行了分组操作，然而$1\times1$的pointwise卷积占据了绝大多数的乘加操作，在小模型中为了减少运算量只能减少通道数，然而减少通道数会大幅损害模型精度。作者提出了对$1\times1$也进行分组操作，但是如图１(a)所示，输出只由部分输入通道决定。为了解决这个问题，作者提出了图(c)中的通道混淆(channel shuffle)操作来分享组间的信息，假设一个卷基层有g groups，每个group有n个channel，因此shuffle后会有$g\times n$个通道，首先将输出的通道维度变形为(g, n)，然后转置(transpose)、展平(flatten)，shuffle操作也是可导的。

![](../../images/ocr/shuffle_3.png)


图２(a)是一个将卷积替换为depthwise卷积的residual block，(b)中将两个$1\times1$卷积都换为pointwise group convolution，然后添加了channel shuffle，为了简便，没有在第二个pointwise group convolution后面加channel shuffle。根据Xception的论文，depthwise卷积后面没有使用ReLU。(c)为stride > 1的情况，此时在shotcut path上使用$3\times3$的平均池化，并将加法换做通道concatenation来增加输出通道数(一般的设计中，stride=2的同时输出通道数乘2)。

对于$c \times h \times w$的输入大小，bottleneck channels为m，则ResNet unit需要$hw(2cm + 9m^2)FLOPs$，ResNeXt需要$hw(2cm + 9m^2/g)FLOPs$，ShuffleNet unit只需要$hw(2cm/g + 9m)FLOPs$，g表示卷积分组数。换句话说，在有限计算资源有限的情况下，ShuffleNet可以使用更宽的特征图，作者发现对于小型网络这很重要。

即使depthwise卷积理论上只有很少的运算量，但是在移动设备上的实际实现不够高效，和其他密集操作(dense operation)相比，depthwise卷积的computation/memory access ratio很糟糕。因此作者只在bottleneck里实现了depthwise卷积。

![](../../images/ocr/covertCNN5.jpg)

https://arxiv.org/pdf/1610.02357.pdf

## 场景文字检测—CTPN原理与实现

https://zhuanlan.zhihu.com/p/34757009

