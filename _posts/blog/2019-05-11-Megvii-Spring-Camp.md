---
layout: post
title: "Megvii:计算机视觉之物体检测与深度神经网络模型设计专场"
categories: [blog ]
tags: [计算机视觉]
description: "参加计算机视觉物体检测的一些想法"
---

* content
{:toc}

## 概述

今天主要围绕计算机视觉领域的一些任务中，如何提高score的方面进行了各种方面的讨论。研究领域包括：基本的场景的物体检测，开放领域的物体检测，不规则文本检测。涉及的相关技术有
* 在backbone设计中总结的一些tips；
* 如何在物体检测领域，通过分析对比各种方法，提出自己的解决方案；
* 物体检测领域中多尺度问题的重要性；
* anchor是什么东西，anchor free和anchor base之间的 vs
* attention在计算机视觉中的应用SENet

## Deform Convlayer v.s. Conv Layer
这个专题主要是为了改善遮挡和分割算法而做的相关研究。在普通的卷积中，加入旋转，scale等特性，使得能够对一个非正方凸区域进行卷积。具体有啥好处，我暂时没有听明白。

## Review Object Detection from anchor base to anchor-free
这个主要通过介绍anchor-base和anchor-free的发展路线，讨论两种方法的之间的优劣之处。
认为现在anchor-base的research空间已经有限，所以开始考虑向anchor-free开始进展。
anchor-base需要设置较多的超参数，落地的过程中存在较大的调参困难。而anchor-free相对来说，比较少。

## 当前scene text detection中的一些问题和研究进展

这个话题我还比较感兴趣的。因为我后面要做一些自然场景文本检测的一个项目，所以最近在突击这方面的论文。而这个论坛中关于自然场景文本检测的分享让我受益颇丰。我才发现原来要找的相关文献很多都是白翔老师这个组的工作。

> 白翔，华中科技大学电信学院教授，先后于华中大获得学士、硕士、博士学位。他的主要研究领域为计算机视觉与模式识别、深度学习应用技术。尤其在形状的匹配与检索、相似性度量与融合、场景OCR取得了一系列重要研究成果，入选2014、2015、2016年Elsevier中国高被引学者。他的研究工作曾获微软学者，首届国家自然科学基金优秀青年基金的资助。他已在相关领域一流国际期刊或会议如PAMI、IJCV、CVPR、ICCV、ECCV、NIPS、ICML、AAAI、IJCAI上发表论文40余篇。任国际期刊Pattern Recognition, Pattern Recognition Letters, Neurocomputing, Frontier of Computer Science编委，VALSE指导委员，曾任VALSE在线委员会(VOOC)主席, VALSE 2016大会主席, 是VALSE在线活动（VALSE Webinar）主要发起人之一。

现在还是没有入门，师傅已经带我走马观花了，剩下的就是我自己细细品读各个方法的奥秘了。
### 暂时定的路线是：

1. SWT，MSER两个经典方法和衍生方法
2. 引入CNN之后如何演变到CNN+LSTM的方法
3. 分割技术如何提升不规则排布文本检测的

### 需要思考的内容：

1. 常用数据集，和应用场景的实际数据之间的差别
2. 各个方法之间的改进侧略的出发点是什么，有什么收货
3. 水平文本检测性能、速度比较推荐的算法是什么，简单应用场景呢？
4. 常规检测算法的与文本检测算法之间能不能直接迁移应用，有什么gap，克服这个gap需要做什么工作
5. 分割算法应用到文本检测中有什么结果，为什么会对非规则文本检测有好处

## 相关文献及blog

- [Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition](https://arxiv.org/abs/1811.00751)
- [Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes](https://blog.csdn.net/dQCFKyQDXYm3F8rB0/article/details/81437413)
- https://blog.csdn.net/francislucien2017/article/details/88583219
